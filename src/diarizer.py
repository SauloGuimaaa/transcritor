import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

class DiarizerEngine:
    def __init__(self):
        print("👥 Inicializando motor de Diarização (Pyannote)...")
        
        # Pega o token do arquivo .env
        auth_token = os.getenv("HF_TOKEN")
        if not auth_token:
            raise ValueError("❌ ERRO: Token HF_TOKEN não encontrado no arquivo .env!")

        try:
            # Baixa/Carrega o pipeline de diarização
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            
            # Envia para a GPU (Sua RTX 3050)
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("   ✅ Diarização rodando na GPU (CUDA).")
            else:
                print("   ⚠️ AVISO: GPU não detectada para diarização. Vai ser lento!")

        except Exception as e:
            print(f"❌ Erro ao carregar Pyannote: {e}")
            print("DICA: Verifique se aceitou os termos de uso em huggingface.co/pyannote/speaker-diarization-3.1")
            raise e

    def process(self, audio_path, num_speakers=None):
        """
        Processa o áudio e retorna quem falou e quando.
        Args:
            audio_path: Caminho do arquivo WAV
            num_speakers: (Opcional) Se você souber quantos falantes são, ajuda o modelo.
        Returns:
            Lista de segmentos: [(start, end, speaker), ...]
        """
        print(f"👥 Identificando falantes em: {os.path.basename(audio_path)}...")
        
        # Executa a diarização
        # min_speakers e max_speakers podem ser ajustados se necessário
        diarization = self.pipeline(audio_path, num_speakers=num_speakers)

        results = []
        # Itera sobre os segmentos detectados
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Armazena: Inicio, Fim, Nome do Falante (ex: SPEAKER_00)
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
            
        print(f"   ✅ Detectados {len(results)} segmentos de fala.")
        return results