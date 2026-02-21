def merge_transcription_diarization(whisper_segments, diarization_segments):
    """
    Combina os segmentos de texto do Whisper com os segmentos de falantes do Pyannote.
    Lógica: Atribui o falante que esteve ativo pela maior parte do tempo do segmento de texto.
    """
    print("🔄 Cruzando dados de Texto (Whisper) com Falantes (Pyannote)...")
    
    final_segments = []

    for text_seg in whisper_segments:
        # Segmento do Whisper (Start, End, Text)
        w_start = text_seg.start
        w_end = text_seg.end
        w_text = text_seg.text

        # Encontrar qual falante estava ativo neste intervalo
        best_speaker = "Desconhecido"
        max_overlap = 0

        for dia_seg in diarization_segments:
            # Calcula a interseção (overlap) de tempo
            start_overlap = max(w_start, dia_seg['start'])
            end_overlap = min(w_end, dia_seg['end'])
            overlap_duration = max(0, end_overlap - start_overlap)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = dia_seg['speaker']

        final_segments.append({
            "start": w_start,
            "end": w_end,
            "text": w_text,
            "speaker": best_speaker
        })

    return final_segments