import argparse
import os
import tempfile
import uuid
from functools import lru_cache

import gradio as gr
import torch

from openvoice import se_extractor
from openvoice.api import ToneColorConverter


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
parser.add_argument("--host", default="::", help="IPv6 host to bind. Defaults to ::")
parser.add_argument("--port", type=int, default=9004, help="Port to bind. Defaults to 9004")
parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
parser.add_argument("--checkpoint-dir", default="checkpoints_v2", help="OpenVoice V2 checkpoint directory")
parser.add_argument("--output-dir", default="outputs_v2/demo", help="Directory for generated audio")
parser.add_argument("--processed-dir", default="processed_v2/demo", help="Directory for extracted speaker embeddings")
args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
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


def update_example_text(language):
    return gr.update(value=LANGUAGE_TEXT[language])


def refresh_speakers(language):
    try:
        speakers = speaker_choices(language)
    except Exception as exc:
        return gr.update(choices=[], value=""), f"[ERROR] {exc}"
    return gr.update(choices=speakers, value=speakers[0] if speakers else ""), f"Loaded {language} speakers."


def clone_voice(text, language, base_speaker, reference_audio, speed):
    if not reference_audio:
        raise gr.Error("Please upload a reference audio file.")
    if not text or len(text.strip()) < 2:
        raise gr.Error("Please enter at least two characters of text.")

    text = text.strip()
    language = language or "EN_NEWEST"
    speed = float(speed or 1.0)

    converter = get_converter()
    model = get_tts(language)
    speaker_key = pick_speaker(language, base_speaker)
    speaker_id = model.hps.data.spk2id[speaker_key]

    source_path = source_se_path(speaker_key)
    _require_file(source_path)
    source_se = torch.load(source_path, map_location=device)

    request_id = uuid.uuid4().hex[:12]
    tmp_dir = tempfile.mkdtemp(prefix=f"openvoice_v2_{request_id}_", dir=args.output_dir)
    src_path = os.path.join(tmp_dir, "base.wav")
    output_path = os.path.join(tmp_dir, "output.wav")

    target_se, audio_name = se_extractor.get_se(
        reference_audio,
        converter,
        target_dir=args.processed_dir,
        vad=True,
    )

    if torch.backends.mps.is_available() and device == "cpu":
        torch.backends.mps.is_available = lambda: False

    model.tts_to_file(text, speaker_id, src_path, speed=speed)
    converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell",
    )

    info = (
        "Generated successfully.\n"
        f"Language: {language}\n"
        f"Base speaker: {speaker_key}\n"
        f"Reference embedding: {audio_name}"
    )
    return info, output_path, output_path


with gr.Blocks(title="OpenVoice V2 Demo", analytics_enabled=False) as demo:
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
                type="filepath",
                value="resources/example_reference.mp3",
            )
            with gr.Row():
                refresh_button = gr.Button("Refresh speakers")
                clone_button = gr.Button("Clone voice", variant="primary")

        with gr.Column():
            info_gr = gr.Textbox(label="Status", lines=5)
            output_audio_gr = gr.Audio(label="Output audio", type="filepath")
            download_gr = gr.File(label="Download output")

    language_gr.change(update_example_text, inputs=language_gr, outputs=input_text_gr)
    language_gr.change(refresh_speakers, inputs=language_gr, outputs=[base_speaker_gr, info_gr])
    demo.load(refresh_speakers, inputs=language_gr, outputs=[base_speaker_gr, info_gr])
    refresh_button.click(refresh_speakers, inputs=language_gr, outputs=[base_speaker_gr, info_gr])
    clone_button.click(
        clone_voice,
        inputs=[input_text_gr, language_gr, base_speaker_gr, reference_gr, speed_gr],
        outputs=[info_gr, output_audio_gr, download_gr],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True,
        show_api=True,
    )
