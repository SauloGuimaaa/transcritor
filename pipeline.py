import os
import sys
import gc
import torch

# --- 1. CONFIGURAÇÃO DE DLL (Amesma de sempre) ---
def fix_dll_path():
    try:
        base_path = sys.prefix 
        nvidia_paths = [
            os.path.join(base_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
            os.path.join(base_path, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        ]
        for path in nvidia_paths:
            if os.path.exists(path):
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
    except: pass
fix_dll_path()
# -------------------------------------------------

from src.audio_utils import convert_to_wav
from src.transcriber import TranscriberEngine
from src.diarizer import DiarizerEngine
from src.merger import merge_transcription_diarization
from src.exporters import export_to_txt, export_to_srt, export_to_docx

# --- CONFIGURAÇÃO ---
ARQUIVO_ENTRADA = "audio_whats.ogg" # <--- SEU ARQUIVO AQUI
# --------------------

def run_pipeline():
    base_dir = os.getcwd()
    input_file = os.path.join(base_dir, "data", "input", ARQUIVO_ENTRADA)
    processed_wav = os.path.join(base_dir, "data", "processing", "final_processed.wav")
    
    # 1. Conversão
    if not os.path.exists(input_file):
        print("❌ Arquivo de entrada não encontrado!")
        return
    
    convert_to_wav(input_file, processed_wav)

    # 2. Transcrição (Whisper na GPU)
    print("\n--- ETAPA 1: TRANSCRIÇÃO (WHISPER) ---")
    transcriber = TranscriberEngine(model_size="medium", device="cuda")
    # Precisamos capturar os segmentos brutos, não só o texto
    segments_gen, _ = transcriber.model.transcribe(processed_wav, language="pt", vad_filter=True)
    whisper_segments = list(segments_gen) # Converte gerador para lista
    
    # Limpeza de Memória VRAM (Crítico para quem tem GPU de 4GB/6GB)
    del transcriber
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 Memória da GPU liberada para a próxima etapa.")

    # 3. Diarização (Pyannote na CPU - Mais estável no Windows)
    print("\n--- ETAPA 2: DIARIZAÇÃO (PYANNOTE) ---")
    diarizer = DiarizerEngine()
    # Forçamos o uso da CPU dentro da classe (se necessário, edite o diarizer.py, 
    # mas o teste_cpu.py provou que CPU funciona bem)
    diarization_segments = diarizer.process(processed_wav)

    # 4. Fusão
    print("\n--- ETAPA 3: FUSÃO E EXPORTAÇÃO ---")
    final_data = merge_transcription_diarization(whisper_segments, diarization_segments)

    # 5. Exportação
    output_base = os.path.join(base_dir, "data", "output", os.path.splitext(ARQUIVO_ENTRADA)[0])
    
    export_to_txt(final_data, output_base + ".txt")
    export_to_srt(final_data, output_base + ".srt")
    export_to_docx(final_data, output_base + ".docx")

    print(f"\n✅ PROCESSO CONCLUÍDO! Arquivos gerados em data/output/")

if __name__ == "__main__":
    run_pipeline()