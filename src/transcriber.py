import os
import wave
import torch
from faster_whisper import WhisperModel


class TranscriberEngine:
    def __init__(self, model_size="medium", device="cuda", compute_type=None):
        """
        Inicializa o modelo Whisper.
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
            device: 'cuda' (GPU) ou 'cpu'
            compute_type: 'float16' para GPU, 'int8_float16' para GPU com VRAM limitada,
                          'int8' para CPU. None = detecta automaticamente.
        """
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"Carregando modelo '{model_size}' em {device.upper()} ({compute_type})...")

        # Tenta float16; se faltar VRAM, cai para int8_float16 automaticamente.
        for ct in ([compute_type] if compute_type != "float16" else ["float16", "int8_float16"]):
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=ct)
                self._compute_type = ct
                print(f"Modelo carregado com sucesso! (compute_type={ct})")
                return
            except Exception as e:
                if "out of memory" in str(e).lower() and ct == "float16":
                    print("AVISO: VRAM insuficiente para float16, tentando int8_float16...")
                    continue
                print(f"Erro ao carregar modelo: {e}")
                raise

    def transcribe(self, audio_path, beam_size=5, language="pt"):
        """
        Realiza a transcrição do arquivo WAV.

        Args:
            audio_path: Caminho do arquivo WAV (16kHz, mono).
            beam_size: Tamanho do beam search. Menor = mais rápido, menos preciso.
            language: Código ISO do idioma (ex: 'pt', 'en'). None = detecção automática.

        Returns:
            Tupla (segments, info) onde segments é uma lista de objetos NamedSegment
            com atributos .start, .end, .text, e info contém metadados da transcrição.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

        duration = self._get_duration(audio_path)
        print(f"Iniciando transcrição ({duration / 60:.1f} min de áudio)...")

        segments_gen, info = self.model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=True,
            # Evita loops de alucinação em áudios longos
            condition_on_previous_text=True,
            # Parâmetros VAD ajustados para podcasts (pausas mais longas são comuns)
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        print(f"   Idioma detectado: {info.language} (confiança: {info.language_probability:.2f})")

        segments = []
        for seg in segments_gen:
            print(f"   [{seg.start:7.2f}s -> {seg.end:7.2f}s] {seg.text.strip()}")
            segments.append(seg)

        print(f"   Total: {len(segments)} segmentos transcritos.")
        return segments, info

    @staticmethod
    def _get_duration(wav_path):
        """Retorna duração em segundos sem dependências externas."""
        try:
            with wave.open(wav_path) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return 0.0
