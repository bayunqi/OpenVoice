import argparse
import base64
import glob
import mimetypes
import os
import site
import sys
import tempfile
import time
import traceback
import uuid
from functools import lru_cache
from ipaddress import ip_address


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

    current_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    if current_paths[: len(cudnn_lib_dirs)] == cudnn_lib_dirs:
        return

    env = os.environ.copy()
    env["OPENVOICE_CUDNN_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([*cudnn_lib_dirs, *current_paths])
    print(f">> Using PyTorch-bundled cuDNN from: {':'.join(cudnn_lib_dirs)}")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


bootstrap_cudnn_library_path()


import gradio as gr
import requests
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
parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
parser.add_argument("--no-queue", action="store_true", help="Disable Gradio queue for local debugging.")
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


def ensure_nltk_resources():
    try:
        import nltk
    except ImportError:
        return

    resources = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/cmudict", "cmudict"),
    ]
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f">> Downloading NLTK resource: {package}")
            nltk.download(package)


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
    ensure_nltk_resources()
    start_time = time.perf_counter()
    print(f">> Preloading OpenVoice V2 models for {args.preload_language} on {device}...")
    converter = get_converter()
    get_tts(args.preload_language)
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


def update_example_text(language):
    return gr.update(value=LANGUAGE_TEXT[language])


def refresh_speakers(language):
    try:
        speakers = speaker_choices(language)
    except Exception as exc:
        return gr.update(choices=[], value=""), f"[ERROR] {exc}"
    return gr.update(choices=speakers, value=speakers[0] if speakers else ""), f"Loaded {language} speakers."


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


def _is_ipv6_literal(host):
    try:
        return ip_address(host.strip("[]")).version == 6
    except ValueError:
        return False


def _startup_check_host(host):
    host = host.strip("[]")
    if host in {"::", "0:0:0:0:0:0:0:0"}:
        return "[::1]"
    if _is_ipv6_literal(host):
        return f"[{host}]"
    return host


def _rewrite_gradio_local_url(url):
    if not _is_ipv6_literal(args.host):
        return url

    raw_host = args.host.strip("[]")
    bracketed_host = f"[{raw_host}]"
    invalid_authorities = (
        f"http://{args.host}:{args.port}",
        f"https://{args.host}:{args.port}",
        f"http://{raw_host}:{args.port}",
        f"https://{raw_host}:{args.port}",
        f"http://{bracketed_host}:{args.port}",
        f"https://{bracketed_host}:{args.port}",
    )
    valid_host = _startup_check_host(args.host)
    for invalid_authority in invalid_authorities:
        if url == invalid_authority or url.startswith(f"{invalid_authority}/"):
            scheme = invalid_authority.split("://", 1)[0]
            valid_authority = f"{scheme}://{valid_host}:{args.port}"
            return valid_authority + url[len(invalid_authority):]
    return url


def on_reference_audio_change(path):
    if not path or not os.path.isfile(path):
        return gr.update(value="", visible=False)
    return gr.update(value=inline_audio_html(path), visible=True)


REF_AUDIO_CSS = """
.ref-audio-noplayer .component-wrapper,
.ref-audio-noplayer audio { display: none !important; }
.ref-audio-noplayer .audio-container { height: auto !important; }
"""


with gr.Blocks(title="OpenVoice V2 Demo", analytics_enabled=False, css=REF_AUDIO_CSS) as demo:
    gr.Markdown("# OpenVoice V2 Local Demo")
    gr.Markdown("Upload reference audio, enter text, clone the voice, then play or download the generated output.")

    with gr.Row():
        with gr.Column():
            language_gr = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGE_TEXT.keys()),
                value="EN_NEWEST",
            )
            input_text_gr = gr.Textbox(
                label="Text",
                value=LANGUAGE_TEXT["EN_NEWEST"],
                lines=4,
            )
            base_speaker_gr = gr.Dropdown(
                label="Base speaker",
                choices=[],
                value=None,
                info="Leave empty or refresh after selecting language to use the first available speaker.",
            )
            speed_gr = gr.Slider(
                label="Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.05,
            )
            reference_gr = gr.Audio(
                label="Reference audio",
                elem_classes=["ref-audio-noplayer"],
                sources=["upload", "microphone"],
                type="filepath",
            )
            reference_preview_gr = gr.HTML(label="Reference preview", visible=False)
            with gr.Row():
                refresh_button = gr.Button("Refresh speakers")
                clone_button = gr.Button("Clone voice", variant="primary")

        with gr.Column():
            info_gr = gr.Textbox(label="Status", lines=8)
            output_audio_gr = gr.HTML(label="Output audio")

    language_gr.change(update_example_text, inputs=language_gr, outputs=input_text_gr)
    reference_gr.change(on_reference_audio_change, inputs=reference_gr, outputs=reference_preview_gr)
    reference_gr.clear(lambda: gr.update(value="", visible=False), outputs=reference_preview_gr)
    refresh_button.click(refresh_speakers, inputs=language_gr, outputs=[base_speaker_gr, info_gr])
    clone_button.click(
        clone_voice,
        inputs=[input_text_gr, language_gr, base_speaker_gr, reference_gr, speed_gr],
        outputs=[info_gr, output_audio_gr],
    )


def launch_demo():
    preload_models()
    if not args.no_queue:
        demo.queue(20)

    original_session_request = requests.sessions.Session.request

    def ipv6_safe_request(session, method, url, *request_args, **request_kwargs):
        url = _rewrite_gradio_local_url(url)
        return original_session_request(session, method, url, *request_args, **request_kwargs)

    requests.sessions.Session.request = ipv6_safe_request
    try:
        if _is_ipv6_literal(args.host):
            print(f"Running on IPv6 URL:  http://{_startup_check_host(args.host)}:{args.port}")
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=True,
            show_api=True,
            show_error=True,
        )
    finally:
        requests.sessions.Session.request = original_session_request


if __name__ == "__main__":
    launch_demo()
