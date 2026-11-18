"""
pipeline.py — Interface de linha de comando para transcrição de podcasts.

Uso:
    python pipeline.py audio.mp3
    python pipeline.py audio.mp3 --model large-v3 --speakers 2
    python pipeline.py audio.mp3 --model medium --lang en
"""

import os
import sys
import gc
import argparse
import torch

# --- CONFIGURAÇÃO DE DLL (Windows / NVIDIA) ---
def _fix_dll_path():
    nvidia_dirs = [
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    ]
    for path in nvidia_dirs:
        if os.path.exists(path):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

_fix_dll_path()
# ----------------------------------------------

from src.audio_utils import convert_to_wav
from src.transcriber import TranscriberEngine
from src.diarizer import DiarizerEngine
from src.merger import merge_transcription_diarization
from src.exporters import export_to_txt, export_to_srt, export_to_docx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcreve áudio/vídeo com identificação de falantes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Arquivo de entrada (mp3, mp4, ogg, wav, ...)")
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Tamanho do modelo Whisper",
    )
    parser.add_argument(
        "--speakers", type=int, default=0,
        help="Número de falantes (0 = automático)",
    )
    parser.add_argument(
        "--lang", default="pt",
        help="Código de idioma ISO (ex: pt, en, es). Use 'auto' para detecção.",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Dispositivo de processamento",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Diretório de saída (padrão: data/output/)",
    )
    return parser.parse_args()


def run_pipeline(input_file, model_size, num_speakers, language, device, output_dir):
    base_dir = os.getcwd()

    if not os.path.exists(input_file):
        # Tenta também dentro de data/input/
        candidate = os.path.join(base_dir, "data", "input", input_file)
        if os.path.exists(candidate):
            input_file = candidate
        else:
            print(f"Arquivo não encontrado: {input_file}")
            sys.exit(1)

    # Diretórios
    processing_dir = os.path.join(base_dir, "data", "processing")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "output")
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    processed_wav = os.path.join(processing_dir, "pipeline_temp.wav")
    lang_arg = None if language == "auto" else language

    try:
        # 1. Conversão
        print("\n=== ETAPA 1: CONVERSÃO DE ÁUDIO ===")
        convert_to_wav(input_file, processed_wav)

        # 2. Transcrição
        print("\n=== ETAPA 2: TRANSCRIÇÃO (WHISPER) ===")
        transcriber = TranscriberEngine(model_size=model_size, device=device)
        whisper_segments, _ = transcriber.transcribe(processed_wav, language=lang_arg)
        del transcriber
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("   Memória GPU liberada.")

        # 3. Diarização
        print("\n=== ETAPA 3: DIARIZAÇÃO (PYANNOTE) ===")
        diarizer = DiarizerEngine()
        n_spk = num_speakers if num_speakers > 0 else None
        diarization_segments = diarizer.process(processed_wav, num_speakers=n_spk)
        del diarizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Fusão
        print("\n=== ETAPA 4: FUSÃO ===")
        final_data = merge_transcription_diarization(whisper_segments, diarization_segments)

        # 5. Exportação
        print("\n=== ETAPA 5: EXPORTAÇÃO ===")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_base = os.path.join(output_dir, base_name)

        export_to_txt(final_data, output_base + ".txt")
        export_to_srt(final_data, output_base + ".srt")
        export_to_docx(final_data, output_base + ".docx")

        print(f"\n✅ Concluído! Arquivos em: {output_dir}")

    finally:
        if os.path.exists(processed_wav):
            try:
                os.remove(processed_wav)
            except OSError:
                pass


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_file=args.input,
        model_size=args.model,
        num_speakers=args.speakers,
        language=args.lang,
        device=args.device,
        output_dir=args.output_dir,
    )
