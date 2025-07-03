import os
import subprocess
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from utils import process_audio_transcription, create_docx_document, transcription_with_diarization_pipeline
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'recordings')
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_RECORDING_HOURS'] = 24
app.config['FFMPEG_PATH'] = 'ffmpeg'  # Путь к ffmpeg (по умолчанию ищет в PATH)

# Создаем необходимые директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

print(f"📁 Папка для записей: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
print(f"📁 Временная папка: {os.path.abspath(app.config['TEMP_FOLDER'])}")

# Словарь для отслеживания статуса обработки
processing_status = {}

STATUS_FILE = os.path.join(os.path.dirname(__file__), 'processing_status.json')

def cleanup_old_files():
    """Очистка старых временных файлов (старше 1 часа)."""
    try:
        now = time.time()
        for filename in os.listdir(app.config['TEMP_FOLDER']):
            filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
            if os.path.getmtime(filepath) < now - 3600:
                try:
                    os.remove(filepath)
                except Exception as e:
                    app.logger.warning(f"Не удалось удалить файл {filepath}: {str(e)}")
    except Exception as e:
        app.logger.error(f"Cleanup error: {str(e)}")

def save_statuses_to_file():
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(processing_status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        app.logger.error(f"Ошибка при сохранении статусов: {str(e)}")

def load_statuses_from_file():
    global processing_status
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Сброс зависших статусов
                for k, v in loaded.items():
                    if v.get('status') == 'processing':
                        v['status'] = 'not_processed'
                        v['progress'] = 'Файл ожидает обработки (был прерван)'
                processing_status.update(loaded)
    except Exception as e:
        app.logger.error(f"Ошибка при загрузке статусов: {str(e)}")

def process_transcription_async(audio_path, filename):
    """Асинхронная обработка транскрибации и анализа аудио."""
    try:
        processing_status[filename] = {
            'status': 'processing',
            'progress': 'Начинаем обработку...',
            'transcription': None,
            'analysis': None,
            'doc_path': None,
            'error': None
        }
        save_statuses_to_file()
        transcription, analysis, error = transcription_with_diarization_pipeline(audio_path)
        if error:
            processing_status[filename] = {
                'status': 'error',
                'progress': 'Ошибка обработки',
                'transcription': None,
                'analysis': None,
                'doc_path': None,
                'error': error
            }
            save_statuses_to_file()
            return
        doc_path = None
        if transcription:
            doc_path = create_docx_document(transcription, analysis, audio_path)
        processing_status[filename] = {
            'status': 'completed',
            'progress': 'Обработка завершена',
            'transcription': transcription,
            'analysis': analysis,
            'doc_path': doc_path,
            'error': None
        }
        save_statuses_to_file()
    except Exception as e:
        processing_status[filename] = {
            'status': 'error',
            'progress': 'Ошибка обработки',
            'transcription': None,
            'analysis': None,
            'doc_path': None,
            'error': str(e)
        }
        save_statuses_to_file()
        app.logger.error(f"Ошибка в process_transcription_async: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    cleanup_old_files()
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_{timestamp}.webm"
        filename = f"recording_{timestamp}.mp3"
        
        return jsonify({
            'status': 'success',
            'temp_filename': temp_filename,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/save_recording', methods=['POST'])
def save_recording():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file'}), 400
        
        audio_file = request.files['audio']
        temp_filename = request.form.get('temp_filename')
        filename = request.form.get('filename')
        
        if not all([temp_filename, filename]):
            return jsonify({'status': 'error', 'message': 'Missing filename parameters'}), 400
        
        print(f"💾 Сохранение записи: {filename}")
        print(f"📁 Временный файл: {temp_filename}")
        
        # Сохраняем временный файл
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_filename)
        audio_file.save(temp_path)
        print(f"✅ Временный файл сохранен: {temp_path}")
        
        # Конвертируем в MP3
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"🔄 Конвертация в MP3: {output_path}")
        
        try:
            ffmpeg_command = [
                app.config['FFMPEG_PATH'],
                '-i', temp_path,
                '-acodec', 'libmp3lame',
                '-q:a', '2',  # Качество 2 (0-9, где 0 - лучшее)
                '-y',  # Перезаписывать без подтверждения
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60  # Таймаут 60 секунд на конвертацию
            )
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr.decode('utf-8')}")
            
            print(f"✅ MP3 файл сохранен: {output_path}")
            
        except subprocess.TimeoutExpired:
            raise Exception("FFmpeg conversion timeout")
        finally:
            # Всегда удаляем временный файл
            try:
                os.remove(temp_path)
                print(f"🗑️ Временный файл удален: {temp_path}")
            except:
                pass
        
        # Запускаем асинхронную обработку транскрибации
        thread = threading.Thread(
            target=process_transcription_async,
            args=(output_path, filename)
        )
        thread.daemon = True
        thread.start()
        save_statuses_to_file()
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'path': output_path,
            'transcription_started': True
        })
        
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {str(e)}")
        # В случае ошибки пробуем сохранить оригинальный файл как fallback
        error_filename = None
        try:
            error_filename = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
            error_path = os.path.join(app.config['UPLOAD_FOLDER'], error_filename)
            audio_file.save(error_path)
            print(f"💾 Fallback файл сохранен: {error_path}")
        except:
            pass
        
        return jsonify({
            'status': 'error',
            'message': str(e),
            'fallback_file': error_filename
        }), 500

@app.route('/transcription_status/<filename>')
def transcription_status(filename):
    """Получение статуса обработки транскрибации"""
    print(filename)
    print(processing_status)
    if filename in processing_status:
        return jsonify(processing_status[filename])
    else:
        # Fallback: если файл есть на диске, но нет статуса
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return jsonify({
                'status': 'not_processed',
                'progress': 'Файл ожидает обработки',
                'transcription': None,
                'analysis': None,
                'doc_path': None,
                'error': None
            })
        return jsonify({
            'status': 'not_found',
            'progress': 'Файл не найден',
            'transcription': None,
            'analysis': None,
            'doc_path': None,
            'error': None
        })

@app.route('/recordings/<filename>')
def get_recording(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/documents/<filename>')
def get_document(filename):
    """Скачивание документа Word"""
    try:
        doc_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        abs_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
        if os.path.exists(doc_path):
            return send_from_directory(
                abs_dir,
                filename,
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recordings', methods=['GET'])
def api_recordings():
    """
    Получить список всех записей и их статусов для фронтенда.
    Возвращает: [{filename, status, progress, doc_path, error, created_at, size}]
    """
    try:
        recordings_dir = app.config['UPLOAD_FOLDER']
        files = [f for f in os.listdir(recordings_dir) if f.endswith('.mp3')]
        result = []
        for f in sorted(files, reverse=True):
            path = os.path.join(recordings_dir, f)
            stat = os.stat(path)
            status = processing_status.get(f, {}).get('status', 'not_processed')
            progress = processing_status.get(f, {}).get('progress', '')
            doc_path = processing_status.get(f, {}).get('doc_path', None)
            error = processing_status.get(f, {}).get('error', None)
            result.append({
                'filename': f,
                'status': status,
                'progress': progress,
                'doc_path': doc_path,
                'error': error,
                'created_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'size': stat.st_size
            })
        return jsonify({'status': 'success', 'recordings': result})
    except Exception as e:
        app.logger.error(f"Ошибка получения списка записей: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Загрузка статусов при старте
load_statuses_from_file()

if __name__ == '__main__':
    print("🌐 Запуск Flask приложения...")
    print("Приложение будет доступно по адресу: http://127.0.0.1:5000")
    print("Нажмите Ctrl+C для остановки")
    app.run(debug=False, port=5000, host='127.0.0.1')