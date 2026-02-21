import os
import torch
from faster_whisper import WhisperModel

class TranscriberEngine:
    def __init__(self, model_size="medium", device="cuda", compute_type="float16"):
        """
        Inicializa o modelo Whisper.
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
            device: 'cuda' (GPU) ou 'cpu'
            compute_type: 'float16' para GPU (mais rápido), 'int8' para CPU.
        """
        print(f"🧠 Carregando modelo '{model_size}' na {device.upper()}...")
        
        # Ajuste automático para evitar erro de VRAM na 3050 (4GB/6GB)
        # Se der erro de memória, mudaremos compute_type para "int8"
        try:
            self.model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type
            )
            print("🚀 Modelo carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise e

    def transcribe(self, audio_path):
        """
        Realiza a transcrição do arquivo WAV.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Arquivo de áudio processado sumiu!")

        print("🎙️ Iniciando transcrição (pode demorar um pouco na 1ª vez)...")
        
        # segments é um gerador (não carrega tudo na memória de uma vez)
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5, 
            language="pt", # Força português para evitar detecção errada
            vad_filter=True # Filtro de Atividade de Voz (Ignora silêncio)
        )

        print(f"   Idioma detectado: {info.language} (Confiança: {info.language_probability:.2f})")
        
        full_text = []
        for segment in segments:
            # Exibe no terminal em tempo real (estilo Matrix)
            print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text.append(segment.text)

        return " ".join(full_text)