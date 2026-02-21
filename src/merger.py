def merge_transcription_diarization(whisper_segments, diarization_segments):
    """
    Combina segmentos de texto (Whisper) com segmentos de falantes (Pyannote).

    Algoritmo two-pointer O(n+m):
      - Ambas as listas são percorridas em ordem cronológica uma única vez.
      - Para cada segmento Whisper, avança o ponteiro de diarização apenas até
        onde é necessário, sem reiniciar do início a cada iteração.
      - Escala bem para podcasts de 1h+ sem degradação de performance.
    """
    print("Cruzando dados de Texto (Whisper) com Falantes (Pyannote)...")

    # Garante ordem cronológica (ambas já devem vir ordenadas, mas defensivo)
    dia_sorted = sorted(diarization_segments, key=lambda s: s["start"])
    n_dia = len(dia_sorted)

    final_segments = []
    dia_idx = 0  # ponteiro na lista de diarização

    for text_seg in whisper_segments:
        w_start = text_seg.start
        w_end = text_seg.end
        w_text = text_seg.text.strip()

        # Segmentos vazios (silêncio pós-VAD) são descartados
        if not w_text:
            continue

        # Avança o ponteiro descartando segmentos de diarização já terminados
        while dia_idx < n_dia and dia_sorted[dia_idx]["end"] <= w_start:
            dia_idx += 1

        # Percorre apenas os segmentos que podem se sobrepor com este segmento Whisper
        best_speaker = "Desconhecido"
        max_overlap = 0.0
        j = dia_idx

        while j < n_dia and dia_sorted[j]["start"] < w_end:
            dia_seg = dia_sorted[j]
            overlap = max(0.0, min(w_end, dia_seg["end"]) - max(w_start, dia_seg["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = dia_seg["speaker"]
            j += 1

        final_segments.append({
            "start": w_start,
            "end": w_end,
            "text": w_text,
            "speaker": best_speaker,
        })

    print(f"   Fusão concluída: {len(final_segments)} segmentos finais.")
    return final_segments
