import os
import sys
import time
import gc
import torch
import streamlit as st
import yt_dlp
from datetime import datetime

# --- CONFIGURAÇÃO DE DLL ---
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
# ---------------------------

from src.audio_utils import convert_to_wav
from src.transcriber import TranscriberEngine
from src.diarizer import DiarizerEngine
from src.merger import merge_transcription_diarization
from src.exporters import export_to_txt, export_to_srt, export_to_docx

st.set_page_config(page_title="TranscreveLocal Pro", page_icon="🎙️", layout="centered")

# Função Auxiliar para Baixar YouTube
def download_youtube_audio(url, output_folder):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, 'youtube_%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        return filename, info.get('title', 'Video Youtube')

def main():
    st.title("🎙️ TranscreveLocal Pro")
    
    # Abas para escolher o método de entrada
    tab1, tab2 = st.tabs(["📁 Upload de Arquivo", "▶️ Link do YouTube"])
    
    file_path_to_process = None
    original_name = "transcricao"

    # ABA 1: ARQUIVO LOCAL
    with tab1:
        uploaded_file = st.file_uploader("Arraste áudio ou vídeo (mp4, mkv, mp3...)", type=["mp3", "wav", "ogg", "m4a", "mp4", "mov", "mkv"])
        if uploaded_file:
            base_dir = os.getcwd()
            input_filename = f"upload_{int(time.time())}_{uploaded_file.name}"
            file_path_to_process = os.path.join(base_dir, "data", "input", input_filename)
            original_name = uploaded_file.name
            # Salva apenas se ainda não existir
            if not os.path.exists(file_path_to_process):
                with open(file_path_to_process, "wb") as f:
                    f.write(uploaded_file.getbuffer())

    # ABA 2: YOUTUBE
    with tab2:
        yt_url = st.text_input("Cole o link do vídeo aqui:")
        if yt_url:
            original_name = "Youtube_Video"
            # O download real acontece só ao clicar no botão processar para não travar a tela

    # Configurações Globais
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox("Modelo Whisper", ["small", "medium", "large-v3"], index=1)
    with col2:
        num_speakers = st.number_input("Número de Falantes", min_value=0, max_value=10, value=0, help="0 = Detectar automático")

    if st.button("🚀 Iniciar Processamento", type="primary"):
        if not uploaded_file and not yt_url:
            st.warning("⚠️ Por favor, faça upload de um arquivo ou cole um link.")
            return

        status_box = st.status("Iniciando motores...", expanded=True)
        
        try:
            base_dir = os.getcwd()
            
            # Lógica específica do YouTube
            if not uploaded_file and yt_url:
                status_box.write("📥 Baixando áudio do YouTube...")
                try:
                    file_path_to_process, yt_title = download_youtube_audio(yt_url, os.path.join(base_dir, "data", "input"))
                    original_name = yt_title
                    status_box.write("✅ Download concluído!")
                except Exception as e:
                    status_box.update(label="Erro no Download", state="error")
                    st.error(f"Falha ao baixar do YouTube: {e}")
                    return

            # Pipeline Padrão
            status_box.write("⚙️ Normalizando áudio (FFmpeg)...")
            processed_wav = os.path.join(base_dir, "data", "processing", "temp_streamlit.wav")
            convert_to_wav(file_path_to_process, processed_wav)

            status_box.write(f"🧠 Whisper ({model_size}) trabalhando...")
            transcriber = TranscriberEngine(model_size=model_size, device="cuda")
            
            # Transcrição
            segments_gen, _ = transcriber.model.transcribe(processed_wav, language="pt", vad_filter=True)
            whisper_segments = list(segments_gen)
            
            # Limpeza
            del transcriber
            gc.collect()
            torch.cuda.empty_cache()
            
            # Diarização
            status_box.write("👥 Separando os falantes...")
            diarizer = DiarizerEngine()
            n_spk = num_speakers if num_speakers > 0 else None
            diarization_segments = diarizer.process(processed_wav, num_speakers=n_spk)

            # Fusão
            status_box.write("🔄 Organizando o texto final...")
            final_data = merge_transcription_diarization(whisper_segments, diarization_segments)

            # Exportação
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Limpa nome do arquivo para não dar erro no Windows
            safe_name = "".join([c for c in original_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            output_base = os.path.join(base_dir, "data", "output", f"{safe_name}_{timestamp}")
            
            export_to_txt(final_data, output_base + ".txt")
            export_to_srt(final_data, output_base + ".srt")
            export_to_docx(final_data, output_base + ".docx")

            status_box.update(label="✅ Finalizado!", state="complete", expanded=False)
            st.success(f"Pronto! Arquivo: {safe_name}")

            # Botões de Download
            c1, c2, c3 = st.columns(3)
            with open(output_base + ".docx", "rb") as f:
                c1.download_button("📄 Word", f, file_name=f"{safe_name}.docx")
            with open(output_base + ".srt", "rb") as f:
                c2.download_button("🎬 Legenda", f, file_name=f"{safe_name}.srt")
            with open(output_base + ".txt", "rb") as f:
                c3.download_button("📝 Texto", f, file_name=f"{safe_name}.txt")

        except Exception as e:
            st.error(f"Erro fatal: {e}")
            print(e)

if __name__ == "__main__":
    main()