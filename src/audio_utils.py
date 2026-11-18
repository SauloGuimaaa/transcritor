import os
import ffmpeg


def convert_to_wav(input_path, output_path):
    """
    Converte qualquer arquivo de mídia para WAV 16kHz Mono usando FFmpeg.
    Otimizado para Whisper e Pyannote.

    Args:
        input_path: Arquivo de entrada (mp3, mp4, ogg, m4a, mkv, mov, wav, ...).
        output_path: Caminho de saída para o arquivo WAV.

    Returns:
        output_path (str)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    # Garante que o diretório de saída existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Convertendo áudio: {os.path.basename(input_path)} ...")

    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ac=1, ar=16000)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        print(f"   Conversão concluída: {os.path.basename(output_path)}")
        return output_path

    except ffmpeg.Error as e:
        raise RuntimeError(
            f"FFmpeg falhou ao converter '{os.path.basename(input_path)}':\n"
            + e.stderr.decode("utf-8", errors="replace")
        ) from e
