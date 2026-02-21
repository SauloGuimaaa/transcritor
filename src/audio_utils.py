import os
import ffmpeg

def convert_to_wav(input_path, output_path):
    """
    Converte qualquer arquivo de mídia para WAV 16kHz Mono
    usando o FFmpeg. Otimizado para o Whisper.
    """
    print(f"🔄 Processando áudio: {os.path.basename(input_path)}...")
    
    try:
        # Verifica se o arquivo existe
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

        # Configura o FFmpeg:
        # -y: Sobrescrever se existir
        # -ac 1: Mono (Audio Channel = 1) -> Whisper não precisa de estéreo
        # -ar 16000: Taxa de amostragem de 16kHz (Padrão do modelo)
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ac=1, ar=16000)
            .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        print(f"✅ Conversão concluída: {os.path.basename(output_path)}")
        return output_path

    except ffmpeg.Error as e:
        print(f"❌ Erro no FFmpeg: {e.stderr.decode('utf8')}")
        raise e
    except Exception as e:
        print(f"❌ Erro genérico: {str(e)}")
        raise e