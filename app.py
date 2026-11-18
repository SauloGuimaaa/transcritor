import os
import sys
import gc
import time
import uuid
import wave
import torch
import streamlit as st
from datetime import datetime

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

# Diretórios necessários criados na inicialização da app
_BASE_DIR = os.getcwd()
_DIRS = {
    "input":      os.path.join(_BASE_DIR, "data", "input"),
    "processing": os.path.join(_BASE_DIR, "data", "processing"),
    "output":     os.path.join(_BASE_DIR, "data", "output"),
}
for _d in _DIRS.values():
    os.makedirs(_d, exist_ok=True)


st.set_page_config(page_title="TranscreveLocal Pro", page_icon="🎙️", layout="centered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(name):
    """Remove caracteres inválidos para nomes de arquivo."""
    return "".join(c for c in name if c.isalnum() or c in " _-").strip() or "transcricao"


def _audio_duration_min(wav_path):
    """Retorna duração em minutos do WAV sem dependências extras."""
    try:
        with wave.open(wav_path) as w:
            return w.getnframes() / w.getframerate() / 60
    except Exception:
        return 0.0


def _download_youtube(url, folder):
    """Baixa áudio de uma URL do YouTube e retorna (caminho, título)."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp não está instalado. Execute: pip install yt-dlp")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder, "youtube_%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp muda a extensão após pós-processamento
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".mp3"
        return filename, info.get("title", "Youtube_Video")


def _release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Configurações testadas em ordem: da mais rápida/pesada para a mais leve.
# "bad allocation" (std::bad_alloc) = CTranslate2 não conseguiu alocar memória
# no meio da decodificação; nesse caso tentamos a próxima configuração.
_FALLBACK_CONFIGS = [
    ("cuda", "float16",      5),
    ("cuda", "int8_float16", 5),
    ("cuda", "int8_float16", 1),
    ("cpu",  "int8",         5),
]


def _is_memory_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in (
        "bad alloc", "bad_alloc", "out of memory", "cuda failed",
        "cublas", "cudnn", "memoryerror",
    )) or isinstance(exc, MemoryError)


def _transcribe_with_fallback(wav_path, model_size, status_box):
    last_exc = None
    for device, compute_type, beam in _FALLBACK_CONFIGS:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        transcriber = None
        try:
            transcriber = TranscriberEngine(
                model_size=model_size, device=device, compute_type=compute_type
            )
            segments, _ = transcriber.transcribe(wav_path, beam_size=beam)
            if (device, compute_type, beam) != _FALLBACK_CONFIGS[0]:
                status_box.write(
                    f"   ✅ Concluído no modo econômico ({device}, {compute_type}, beam={beam})"
                )
            return segments
        except Exception as e:
            if not _is_memory_error(e):
                raise
            last_exc = e
            print(f"AVISO: falta de memória com {device}/{compute_type}/beam={beam}: {e}")
            status_box.write(
                f"   ⚠️ Memória insuficiente ({device}, {compute_type}, beam={beam}) — "
                "tentando modo mais leve..."
            )
        finally:
            del transcriber
            _release_gpu()
    raise RuntimeError(f"Sem memória em todas as configurações: {last_exc}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _run_pipeline(file_path, original_name, model_size, num_speakers, status_box):
    """
    Executa o pipeline completo e retorna o caminho base dos arquivos exportados.
    Levanta exceção em qualquer falha; quem chama é responsável pela limpeza.
    """
    # 1. Conversão de áudio
    # Temp file com UUID evita colisões entre execuções simultâneas
    run_id = uuid.uuid4().hex[:8]
    processed_wav = os.path.join(_DIRS["processing"], f"run_{run_id}.wav")

    try:
        status_box.write("⚙️ Normalizando áudio (FFmpeg)...")
        convert_to_wav(file_path, processed_wav)

        duration_min = _audio_duration_min(processed_wav)
        status_box.write(
            f"   Duração do áudio: **{duration_min:.1f} min** "
            f"({'arquivo longo — pode demorar' if duration_min > 20 else 'OK'})"
        )

        # 2. Transcrição
        status_box.write(f"🧠 Transcrevendo com Whisper ({model_size})...")
        whisper_segments = _transcribe_with_fallback(processed_wav, model_size, status_box)
        _release_gpu()

        # 3. Diarização
        status_box.write("👥 Separando falantes (Pyannote)...")
        diarizer = DiarizerEngine()
        n_spk = num_speakers if num_speakers > 0 else None
        diarization_segments = diarizer.process(processed_wav, num_speakers=n_spk)
        del diarizer
        _release_gpu()

        # 4. Fusão
        status_box.write("🔄 Organizando texto final...")
        final_data = merge_transcription_diarization(whisper_segments, diarization_segments)

        # 5. Exportação
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = _safe_filename(original_name)
        output_base = os.path.join(_DIRS["output"], f"{safe_name}_{timestamp}")

        export_to_txt(final_data, output_base + ".txt")
        export_to_srt(final_data, output_base + ".srt")
        export_to_docx(final_data, output_base + ".docx")

        return output_base, safe_name

    finally:
        # Limpeza do WAV temporário independente de sucesso ou falha
        if os.path.exists(processed_wav):
            try:
                os.remove(processed_wav)
            except OSError:
                pass
        _release_gpu()


# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

def _render_downloads():
    """
    Renderiza os botões de download usando caminhos salvos no session_state.
    Chamado fora do bloco if-button para sobreviver ao rerun causado pelo clique
    no próprio botão de download (comportamento padrão do Streamlit).
    """
    output_base = st.session_state.get("output_base")
    safe_name = st.session_state.get("safe_name")

    if not output_base or not safe_name:
        return

    # Verifica se os arquivos ainda existem (proteção extra)
    if not os.path.exists(output_base + ".docx"):
        return

    st.success(f"Transcrição pronta: **{safe_name}**")
    c1, c2, c3 = st.columns(3)

    with open(output_base + ".docx", "rb") as f:
        c1.download_button(
            "📄 Word (.docx)", f.read(),
            file_name=f"{safe_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="dl_docx",
        )
    with open(output_base + ".srt", "rb") as f:
        c2.download_button(
            "🎬 Legenda (.srt)", f.read(),
            file_name=f"{safe_name}.srt",
            mime="text/plain",
            key="dl_srt",
        )
    with open(output_base + ".txt", "rb") as f:
        c3.download_button(
            "📝 Texto (.txt)", f.read(),
            file_name=f"{safe_name}.txt",
            mime="text/plain",
            key="dl_txt",
        )

    if st.button("🔄 Nova transcrição", key="btn_reset"):
        st.session_state.pop("output_base", None)
        st.session_state.pop("safe_name", None)
        st.rerun()


def main():
    st.title("🎙️ TranscreveLocal Pro")

    # Se já existe resultado de uma execução anterior, mostra os downloads
    # imediatamente (sobrevive ao rerun do botão de download).
    if st.session_state.get("output_base"):
        _render_downloads()
        return

    tab1, tab2 = st.tabs(["📁 Upload de Arquivo", "▶️ Link do YouTube"])

    file_path_to_process = None
    original_name = "transcricao"

    with tab1:
        uploaded_file = st.file_uploader(
            "Arraste áudio ou vídeo aqui",
            type=["mp3", "wav", "ogg", "m4a", "mp4", "mov", "mkv"],
        )
        if uploaded_file:
            input_filename = f"upload_{int(time.time())}_{uploaded_file.name}"
            file_path_to_process = os.path.join(_DIRS["input"], input_filename)
            original_name = os.path.splitext(uploaded_file.name)[0]
            if not os.path.exists(file_path_to_process):
                with open(file_path_to_process, "wb") as f:
                    f.write(uploaded_file.getbuffer())

    with tab2:
        yt_url = st.text_input("Cole o link do vídeo aqui:")
        if yt_url:
            original_name = "Youtube_Video"

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox(
            "Modelo Whisper",
            ["small", "medium", "large-v3"],
            index=1,
            help="large-v3 = mais preciso, mais lento e mais VRAM",
        )
    with col2:
        num_speakers = st.number_input(
            "Número de Falantes",
            min_value=0, max_value=20, value=0,
            help="0 = detecção automática",
        )

    if st.button("🚀 Iniciar Processamento", type="primary"):
        if not uploaded_file and not yt_url:
            st.warning("Por favor, faça upload de um arquivo ou cole um link do YouTube.")
            return

        status_box = st.status("Iniciando...", expanded=True)

        try:
            if not uploaded_file and yt_url:
                status_box.write("📥 Baixando áudio do YouTube...")
                file_path_to_process, yt_title = _download_youtube(yt_url, _DIRS["input"])
                original_name = yt_title
                status_box.write("   Download concluído!")

            output_base, safe_name = _run_pipeline(
                file_path=file_path_to_process,
                original_name=original_name,
                model_size=model_size,
                num_speakers=num_speakers,
                status_box=status_box,
            )

            status_box.update(label="✅ Processamento concluído!", state="complete", expanded=False)

            # Persiste os caminhos no session_state para sobreviver ao rerun
            st.session_state["output_base"] = output_base
            st.session_state["safe_name"] = safe_name

            # Força rerun para renderizar os botões de download fora do bloco if-button
            st.rerun()

        except Exception as e:
            status_box.update(label="Erro durante o processamento", state="error")
            st.error(f"Falha: {e}")
            _release_gpu()


if __name__ == "__main__":
    main()
