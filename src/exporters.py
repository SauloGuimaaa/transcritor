import os
from docx import Document
import srt
from datetime import timedelta

def export_to_txt(segments, output_path):
    """Gera um arquivo de texto simples."""
    print(f"💾 Salvando TXT: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get('speaker', 'Desconhecido')
            text = seg.get('text', '').strip()
            # Formato: [SPEAKER_00] Texto...
            f.write(f"[{speaker}] {text}\n")

def export_to_srt(segments, output_path):
    """Gera legendas SRT compatíveis com YouTube/VLC."""
    print(f"💾 Salvando SRT: {output_path}")
    subs = []
    for idx, seg in enumerate(segments):
        start = timedelta(seconds=seg['start'])
        end = timedelta(seconds=seg['end'])
        text = f"[{seg['speaker']}] {seg['text']}"
        
        subs.append(srt.Subtitle(index=idx+1, start=start, end=end, content=text))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

def export_to_docx(segments, output_path):
    """Gera um arquivo Word formatado."""
    print(f"💾 Salvando DOCX: {output_path}")
    doc = Document()
    doc.add_heading('Transcrição Automática', 0)

    for seg in segments:
        speaker = seg.get('speaker', 'Desconhecido')
        text = seg.get('text', '')
        start = timedelta(seconds=seg['start'])
        
        # Formata o tempo ex: 00:01:25
        timestamp = str(start).split('.')[0] 

        p = doc.add_paragraph()
        # Negrito para o falante e tempo
        runner = p.add_run(f"[{timestamp}] {speaker}: ")
        runner.bold = True
        p.add_run(text)

    doc.save(output_path)