import os
import subprocess
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'recordings'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_RECORDING_HOURS'] = 24
app.config['FFMPEG_PATH'] = 'ffmpeg'  # Путь к ffmpeg (по умолчанию ищет в PATH)

# Создаем необходимые директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

def cleanup_old_files():
    """Очистка старых временных файлов"""
    try:
        now = time.time()
        for filename in os.listdir(app.config['TEMP_FOLDER']):
            filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
            if os.path.getmtime(filepath) < now - 3600:  # Удаляем файлы старше 1 часа
                try:
                    os.remove(filepath)
                except:
                    continue
    except Exception as e:
        app.logger.error(f"Cleanup error: {str(e)}")

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
        
        # Сохраняем временный файл
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_filename)
        audio_file.save(temp_path)
        
        # Конвертируем в MP3
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
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
            
        except subprocess.TimeoutExpired:
            raise Exception("FFmpeg conversion timeout")
        finally:
            # Всегда удаляем временный файл
            try:
                os.remove(temp_path)
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'path': output_path
        })
        
    except Exception as e:
        # В случае ошибки пробуем сохранить оригинальный файл как fallback
        error_filename = None
        try:
            error_filename = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
            error_path = os.path.join(app.config['UPLOAD_FOLDER'], error_filename)
            audio_file.save(error_path)
        except:
            pass
        
        return jsonify({
            'status': 'error',
            'message': str(e),
            'fallback_file': error_filename
        }), 500

@app.route('/recordings/<filename>')
def get_recording(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)