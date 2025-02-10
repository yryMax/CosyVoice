import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import argparse

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)


def synthesize(ref_audio_path, text, output_dir, prompt_text):
    prompt_speech_16k = load_wav(ref_audio_path, 16000)
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text , prompt_speech_16k, stream=False)):
        torchaudio.save(output_dir, j['tts_speech'], cosyvoice.sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice CLI TTS")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio file")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prompt_text", type=str, required=True, help="Prompt text")

    args = parser.parse_args()

    synthesize(args.ref_audio, args.text, args.output_dir, args.prompt_text)