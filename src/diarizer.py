import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()


class DiarizerEngine:
    def __init__(self):
        print("Inicializando motor de Diarização (Pyannote)...")

        auth_token = os.getenv("HF_TOKEN")
        if not auth_token:
            raise ValueError(
                "Token HF_TOKEN não encontrado no arquivo .env!\n"
                "Crie um arquivo .env com: HF_TOKEN=hf_seu_token_aqui"
            )

        try:
            # pyannote.audio 3.1.x usa `use_auth_token`; versões >= 3.2 usam `token`.
            # Tentamos o parâmetro novo primeiro; se falhar, caímos para o antigo.
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=auth_token,
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=auth_token,
                )

            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("   Diarização rodando na GPU (CUDA).")
            else:
                print("   AVISO: GPU não detectada. Diarização será lenta na CPU.")

        except Exception as e:
            print(f"Erro ao carregar Pyannote: {e}")
            print("DICA: Verifique se aceitou os termos em huggingface.co/pyannote/speaker-diarization-3.1")
            raise

    def process(self, audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
        """
        Processa o áudio e retorna quem falou e quando.

        Args:
            audio_path: Caminho do arquivo WAV.
            num_speakers: Número exato de falantes (sobrescreve min/max quando definido).
            min_speakers: Número mínimo de falantes (usado se num_speakers=None).
            max_speakers: Número máximo de falantes (usado se num_speakers=None).

        Returns:
            Lista de dicts: [{"start": float, "end": float, "speaker": str}, ...]
        """
        print(f"Identificando falantes em: {os.path.basename(audio_path)}...")

        kwargs = {}
        if num_speakers and num_speakers > 0:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers and min_speakers > 0:
                kwargs["min_speakers"] = min_speakers
            if max_speakers and max_speakers > 0:
                kwargs["max_speakers"] = max_speakers

        diarization = self.pipeline(audio_path, **kwargs)

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        speakers_found = {r["speaker"] for r in results}
        print(f"   {len(speakers_found)} falante(s) detectado(s), {len(results)} segmentos.")
        return results
