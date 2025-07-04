import os
import sys
import shutil
# Добавляем абсолютный путь к папке app
APP_DIR = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, APP_DIR)
from utils import transcription_with_diarization_pipeline, create_docx_document

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), 'recordings')


def main():
    mp3_files = [f for f in os.listdir(RECORDINGS_DIR) if f.lower().endswith('.mp3')]
    total = len(mp3_files)
    if not mp3_files:
        print('В папке recordings нет mp3-файлов.')
        return
    iterator = tqdm(mp3_files, desc='Обработка файлов') if use_tqdm else mp3_files
    for idx, mp3_file in enumerate(iterator, 1):
        if not use_tqdm:
            print(f'\n[{idx}/{total}] Обработка файла: {mp3_file}')
        else:
            iterator.set_postfix({'файл': mp3_file})
        audio_path = os.path.join(RECORDINGS_DIR, mp3_file)
        # Создаём временный файл без пробелов
        temp_audio_path = os.path.join(RECORDINGS_DIR, f'temp_audio_{idx}.mp3')
        shutil.copy(audio_path, temp_audio_path)
        try:
            transcription, analysis, error = transcription_with_diarization_pipeline(temp_audio_path)
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        if error:
            print(f'Ошибка: {error}')
            continue
        # Сохраняем в txt
        base = os.path.splitext(mp3_file)[0]
        txt_path = os.path.join(RECORDINGS_DIR, f'{base}_transcription.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f'Транскрипция сохранена в {txt_path}')
        # Сохраняем в docx
        docx_path = create_docx_document(transcription, analysis, audio_path)
        if docx_path:
            print(f'DOCX сохранён: {docx_path}')
        else:
            print('Ошибка при сохранении DOCX')

if __name__ == '__main__':
    main() 