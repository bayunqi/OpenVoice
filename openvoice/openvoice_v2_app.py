import argparse
import base64
import glob
import html
import mimetypes
import os
import shutil
import site
import sys
import traceback
import time
import tempfile
import uuid
from functools import lru_cache


def bootstrap_cudnn_library_path():
    if "--no-cudnn" in sys.argv or os.environ.get("OPENVOICE_CUDNN_BOOTSTRAPPED") == "1":
        return

    candidate_roots = [
        os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
        *site.getsitepackages(),
    ]
    cudnn_lib_dirs = []
    for root in candidate_roots:
        cudnn_lib_dirs.extend(glob.glob(os.path.join(root, "nvidia", "cudnn", "lib")))
    cudnn_lib_dirs = [path for path in dict.fromkeys(cudnn_lib_dirs) if os.path.isdir(path)]
    if not cudnn_lib_dirs:
        return

    current_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    current_paths = [path for path in current_paths if path]
    if current_paths[: len(cudnn_lib_dirs)] == cudnn_lib_dirs:
        return

    env = os.environ.copy()
    env["OPENVOICE_CUDNN_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([*cudnn_lib_dirs, *current_paths])
    print(f">> Using PyTorch-bundled cuDNN from: {':'.join(cudnn_lib_dirs)}")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


bootstrap_cudnn_library_path()


import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import soundfile
import torch

from openvoice import se_extractor
from openvoice.api import ToneColorConverter


mimetypes.add_type("audio/wav", ".wav")


LANGUAGE_TEXT = {
    "EN_NEWEST": "Did you ever hear a folk tale about a giant turtle?",
    "EN": "Did you ever hear a folk tale about a giant turtle?",
    "ES": "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    "FR": "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    "ZH": "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    "JP": "彼は毎朝ジョギングをして体を健康に保っています。",
    "KR": "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}


parser = argparse.ArgumentParser(description="OpenVoice V2 local cloning demo")
parser.add_argument("--host", default="[::]", help="IPv6 host to bind. Defaults to [::]")
parser.add_argument("--port", type=int, default=9004, help="Port to bind. Defaults to 9004")
parser.add_argument("--checkpoint-dir", default="checkpoints_v2", help="OpenVoice V2 checkpoint directory")
parser.add_argument("--output-dir", default="outputs_v2/demo", help="Directory for generated audio")
parser.add_argument("--processed-dir", default="processed_v2/demo", help="Directory for extracted speaker embeddings")
parser.add_argument("--preload-language", default="EN_NEWEST", choices=list(LANGUAGE_TEXT.keys()), help="Language to preload at startup")
parser.add_argument("--no-cudnn", action="store_true", help="Disable cuDNN for emergency debugging.")
args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
if "cuda" in device and args.no_cudnn:
    torch.backends.cudnn.enabled = False
    print(">> cuDNN disabled for this demo.")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.processed_dir, exist_ok=True)


def _require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing required file: {path}. Download and extract checkpoints_v2 before launching this demo."
        )


def _load_melo_tts():
    try:
        from melo.api import TTS
    except ImportError as exc:
        raise RuntimeError(
            "MeloTTS is required for OpenVoice V2. Install it with "
            "`pip install git+https://github.com/myshell-ai/MeloTTS.git` and run `python -m unidic download`."
        ) from exc
    return TTS


@lru_cache(maxsize=1)
def get_converter():
    ckpt_converter = os.path.join(args.checkpoint_dir, "converter")
    config_path = os.path.join(ckpt_converter, "config.json")
    ckpt_path = os.path.join(ckpt_converter, "checkpoint.pth")
    _require_file(config_path)
    _require_file(ckpt_path)

    converter = ToneColorConverter(config_path, device=device)
    converter.load_ckpt(ckpt_path)
    return converter


@lru_cache(maxsize=len(LANGUAGE_TEXT))
def get_tts(language):
    TTS = _load_melo_tts()
    return TTS(language=language, device=device)


def speaker_choices(language):
    model = get_tts(language)
    return list(model.hps.data.spk2id.keys())


def pick_speaker(language, requested_speaker):
    speakers = speaker_choices(language)
    if requested_speaker:
        requested_speaker = requested_speaker.strip()
        if requested_speaker in speakers:
            return requested_speaker
        lowered = {speaker.lower(): speaker for speaker in speakers}
        if requested_speaker.lower() in lowered:
            return lowered[requested_speaker.lower()]
        raise ValueError(f"Unknown base speaker '{requested_speaker}'. Available speakers: {', '.join(speakers)}")
    return speakers[0]


def source_se_path(speaker_key):
    speaker_file = speaker_key.lower().replace("_", "-")
    return os.path.join(args.checkpoint_dir, "base_speakers", "ses", f"{speaker_file}.pth")


@lru_cache(maxsize=64)
def get_source_se(speaker_key):
    source_path = source_se_path(speaker_key)
    _require_file(source_path)
    return torch.load(source_path, map_location=device)


def preload_models():
    start_time = time.perf_counter()
    print(f">> Preloading OpenVoice V2 models for {args.preload_language} on {device}...")
    converter = get_converter()
    model = get_tts(args.preload_language)
    default_speaker = pick_speaker(args.preload_language, None)
    get_source_se(default_speaker)
    print(
        f">> Preload complete in {time.perf_counter() - start_time:.2f} seconds "
        f"(version={converter.version}, language={args.preload_language}, speaker={default_speaker})."
    )


def audio_duration_seconds(audio_path):
    info = soundfile.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def audio_data_uri(audio_path):
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    with open(audio_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def inline_audio_html(audio_path, download_name=None):
    if not audio_path or not os.path.isfile(audio_path):
        return ""
    data_uri = audio_data_uri(audio_path)
    player = f'<audio controls style="width:100%" src="{data_uri}"></audio>'
    if download_name:
        player += (
            '<div style="margin-top:8px">'
            f'<a download="{download_name}" href="{data_uri}">Download {download_name}</a>'
            "</div>"
        )
    return player


def refresh_speakers(language):
    try:
        speakers = speaker_choices(language)
    except Exception as exc:
        return [], f"[ERROR] {exc}"
    return speakers, f"Loaded {language} speakers."


def resolve_uploaded_file(uploaded_file):
    if isinstance(uploaded_file, str):
        return uploaded_file
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("name") or uploaded_file.get("path")
    if hasattr(uploaded_file, "name"):
        return uploaded_file.name
    if isinstance(uploaded_file, (list, tuple)) and uploaded_file:
        return resolve_uploaded_file(uploaded_file[0])
    return None


def clone_voice(text, language, base_speaker, reference_audio, speed):
    try:
        reference_audio = resolve_uploaded_file(reference_audio)
        if not reference_audio or not os.path.isfile(reference_audio):
            return "Please upload a reference audio file before cloning.", ""
        if not text or len(text.strip()) < 2:
            return "Please enter at least two characters of text.", ""

        text = text.strip()
        language = language or "EN_NEWEST"
        speed = float(speed or 1.0)

        converter = get_converter()
        model = get_tts(language)
        speaker_key = pick_speaker(language, base_speaker)
        speaker_id = model.hps.data.spk2id[speaker_key]
        source_se = get_source_se(speaker_key)

        request_id = uuid.uuid4().hex[:12]
        tmp_dir = tempfile.mkdtemp(prefix=f"openvoice_v2_{request_id}_", dir=args.output_dir)
        src_path = os.path.join(tmp_dir, "base.wav")
        output_path = os.path.join(tmp_dir, "output.wav")

        start_time = time.perf_counter()
        step_start_time = start_time
        print(f">> Clone request: language={language}, speaker={speaker_key}, reference={reference_audio}")
        target_se, audio_name = se_extractor.get_se(
            reference_audio,
            converter,
            target_dir=args.processed_dir,
            vad=True,
        )
        extract_time = time.perf_counter() - step_start_time

        if torch.backends.mps.is_available() and device == "cpu":
            torch.backends.mps.is_available = lambda: False

        step_start_time = time.perf_counter()
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        tts_time = time.perf_counter() - step_start_time

        step_start_time = time.perf_counter()
        converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@MyShell",
        )
        convert_time = time.perf_counter() - step_start_time
        total_time = time.perf_counter() - start_time
        output_duration = audio_duration_seconds(output_path)
        rtf = total_time / output_duration if output_duration > 0 else float("inf")

        print(f">> tone_color_extract_time: {extract_time:.2f} seconds")
        print(f">> base_tts_time: {tts_time:.2f} seconds")
        print(f">> tone_color_convert_time: {convert_time:.2f} seconds")
        print(f">> Total inference time: {total_time:.2f} seconds")
        print(f">> Generated audio length: {output_duration:.2f} seconds")
        print(f">> RTF: {rtf:.4f}")

        info = (
            "Generated successfully.\n"
            f"Language: {language}\n"
            f"Base speaker: {speaker_key}\n"
            f"Reference embedding: {audio_name}\n"
            f"Tone color extract time: {extract_time:.2f}s\n"
            f"Base TTS time: {tts_time:.2f}s\n"
            f"Tone color convert time: {convert_time:.2f}s\n"
            f"Total inference time: {total_time:.2f}s\n"
            f"Generated audio length: {output_duration:.2f}s\n"
            f"RTF: {rtf:.4f}"
        )
        return info, inline_audio_html(output_path, download_name=os.path.basename(output_path))
    except Exception as exc:
        traceback.print_exc()
        return f"[ERROR] {type(exc).__name__}: {exc}", ""


app = FastAPI(title="OpenVoice V2 Demo")


def render_index():
    language_options = "\n".join(
        f'<option value="{html.escape(language)}">{html.escape(language)}</option>'
        for language in LANGUAGE_TEXT
    )
    initial_text = html.escape(LANGUAGE_TEXT["EN_NEWEST"])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenVoice V2 Local Demo</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243042; background: #fafafa; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 32px; }}
    h1 {{ margin: 0 0 20px; font-size: 28px; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr); gap: 20px; }}
    .panel {{ border: 1px solid #d9dee7; border-radius: 8px; background: white; padding: 16px; }}
    label {{ display: block; font-size: 14px; font-weight: 600; margin: 14px 0 6px; }}
    select, textarea, input[type="number"], input[type="file"] {{ width: 100%; box-sizing: border-box; border: 1px solid #d2d8e2; border-radius: 6px; padding: 10px; font: inherit; background: white; }}
    textarea {{ min-height: 130px; resize: vertical; }}
    .row {{ display: flex; gap: 12px; align-items: center; margin-top: 16px; }}
    button {{ border: 0; border-radius: 6px; padding: 12px 16px; font-weight: 700; cursor: pointer; }}
    button.secondary {{ background: #eef1f6; color: #2b3545; }}
    button.primary {{ background: #ffb36b; color: #af3d00; flex: 1; }}
    button:disabled {{ opacity: 0.6; cursor: wait; }}
    pre {{ min-height: 180px; white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f7fb; border: 1px solid #d9dee7; border-radius: 6px; padding: 12px; }}
    #output audio {{ width: 100%; margin-top: 12px; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 20px; }} }}
  </style>
</head>
<body>
  <main>
    <h1>OpenVoice V2 Local Demo</h1>
    <div class="grid">
      <section class="panel">
        <label for="language">Language</label>
        <select id="language">{language_options}</select>

        <label for="text">Text</label>
        <textarea id="text">{initial_text}</textarea>

        <label for="speaker">Base speaker</label>
        <select id="speaker"><option value="">Use first available speaker</option></select>

        <label for="speed">Speed</label>
        <input id="speed" type="number" min="0.5" max="2" step="0.05" value="1">

        <label for="reference">Reference audio</label>
        <input id="reference" type="file" accept="audio/*">

        <div class="row">
          <button id="refresh" class="secondary" type="button">Refresh speakers</button>
          <button id="clone" class="primary" type="button">Clone voice</button>
        </div>
      </section>

      <section class="panel">
        <label>Status</label>
        <pre id="status">Ready.</pre>
        <label>Output audio</label>
        <div id="output"></div>
      </section>
    </div>
  </main>
  <script>
    const exampleText = {LANGUAGE_TEXT!r};
    const language = document.getElementById("language");
    const text = document.getElementById("text");
    const speaker = document.getElementById("speaker");
    const speed = document.getElementById("speed");
    const reference = document.getElementById("reference");
    const statusBox = document.getElementById("status");
    const output = document.getElementById("output");
    const cloneButton = document.getElementById("clone");
    const refreshButton = document.getElementById("refresh");

    function setBusy(isBusy) {{
      cloneButton.disabled = isBusy;
      refreshButton.disabled = isBusy;
    }}

    language.addEventListener("change", () => {{
      text.value = exampleText[language.value] || text.value;
      speaker.innerHTML = '<option value="">Use first available speaker</option>';
    }});

    refreshButton.addEventListener("click", async () => {{
      setBusy(true);
      statusBox.textContent = "Loading speakers...";
      try {{
        const response = await fetch(`/api/speakers?language=${{encodeURIComponent(language.value)}}`);
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || response.statusText);
        speaker.innerHTML = '<option value="">Use first available speaker</option>';
        for (const item of data.speakers) {{
          const option = document.createElement("option");
          option.value = item;
          option.textContent = item;
          speaker.appendChild(option);
        }}
        statusBox.textContent = data.message;
      }} catch (error) {{
        statusBox.textContent = `[ERROR] ${{error.message}}`;
      }} finally {{
        setBusy(false);
      }}
    }});

    cloneButton.addEventListener("click", async () => {{
      if (!reference.files.length) {{
        statusBox.textContent = "Please upload a reference audio file before cloning.";
        return;
      }}
      const form = new FormData();
      form.append("text", text.value);
      form.append("language", language.value);
      form.append("base_speaker", speaker.value);
      form.append("speed", speed.value);
      form.append("reference_audio", reference.files[0]);

      setBusy(true);
      output.innerHTML = "";
      statusBox.textContent = "Uploading reference audio and cloning...";
      try {{
        const response = await fetch("/api/clone", {{ method: "POST", body: form }});
        const data = await response.json();
        statusBox.textContent = data.info || "";
        output.innerHTML = data.audio_html || "";
        if (!response.ok || data.error) throw new Error(data.error || response.statusText);
      }} catch (error) {{
        statusBox.textContent = `[ERROR] ${{error.message}}`;
      }} finally {{
        setBusy(false);
      }}
    }});
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return render_index()


@app.get("/api/speakers")
def api_speakers(language: str):
    speakers, message = refresh_speakers(language)
    if message.startswith("[ERROR]"):
        return JSONResponse({"error": message, "speakers": []}, status_code=500)
    return {"speakers": speakers, "message": message}


@app.post("/api/clone")
async def api_clone(
    text: str = Form(...),
    language: str = Form("EN_NEWEST"),
    base_speaker: str = Form(""),
    speed: float = Form(1.0),
    reference_audio: UploadFile = File(...),
):
    request_id = uuid.uuid4().hex[:12]
    suffix = os.path.splitext(reference_audio.filename or "")[1] or ".wav"
    upload_dir = tempfile.mkdtemp(prefix=f"openvoice_upload_{request_id}_", dir=args.output_dir)
    upload_path = os.path.join(upload_dir, f"reference{suffix}")
    print(f">> Receiving upload: filename={reference_audio.filename}, content_type={reference_audio.content_type}, save_path={upload_path}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(reference_audio.file, f)
    print(f">> Upload saved: {upload_path}, size={os.path.getsize(upload_path)} bytes")

    info, audio_html = clone_voice(text, language, base_speaker, upload_path, speed)
    if info.startswith("[ERROR]"):
        return JSONResponse({"info": info, "audio_html": "", "error": info}, status_code=500)
    return {"info": info, "audio_html": audio_html}


def launch_demo():
    preload_models()
    host = args.host.strip("[]")
    print(f"Running on IPv6 URL:  http://[::1]:{args.port}")
    uvicorn.run(app, host=host, port=args.port)


if __name__ == "__main__":
    launch_demo()
