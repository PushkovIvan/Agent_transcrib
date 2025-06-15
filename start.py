#!/usr/bin/env python3
"""
Скрипт для запуска приложения с проверкой зависимостей
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_ffmpeg():
    """Проверка наличия ffmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ffmpeg найден")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ ffmpeg не найден")
    print("Установите ffmpeg:")
    print("  macOS: brew install ffmpeg")
    print("  Ubuntu: sudo apt install ffmpeg")
    print("  Windows: скачайте с https://ffmpeg.org/download.html")
    return False

def check_dependencies():
    """Проверка Python зависимостей"""
    required_packages = [
        'flask', 'whisper', 'torch', 'torchaudio', 'transformers',
        'librosa', 'soundfile', 'sklearn', 'numpy', 'tqdm',
        'langchain_gigachat', 'yaml', 'docx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'docx':
                importlib.import_module('docx')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("Установите их командой:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """Проверка конфигурации"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"❌ Файл конфигурации не найден: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config and 'token' in config and 'gigachat' in config['token']:
            print("✅ Конфигурация GigaChat найдена")
            return True
        else:
            print("❌ Токен GigaChat не найден в конфигурации")
            return False
    except Exception as e:
        print(f"❌ Ошибка при чтении конфигурации: {str(e)}")
        return False

def create_directories():
    """Создание необходимых директорий"""
    directories = ['recordings', 'app/temp']
    
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(__file__), directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 Создана директория: {directory}")
        else:
            print(f"✅ Директория существует: {directory}")

def main():
    """Основная функция"""
    print("=" * 60)
    print("ПРОВЕРКА СИСТЕМЫ ДЛЯ АУДИО РЕКОРДЕРА С ТРАНСКРИБАЦИЕЙ")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 4
    
    # Проверка версии Python
    if check_python_version():
        checks_passed += 1
    
    print()
    
    # Проверка ffmpeg
    if check_ffmpeg():
        checks_passed += 1
    
    print()
    
    # Проверка зависимостей
    if check_dependencies():
        checks_passed += 1
    
    print()
    
    # Проверка конфигурации
    if check_config():
        checks_passed += 1
    
    print()
    
    # Создание директорий
    create_directories()
    
    print()
    print("=" * 60)
    
    if checks_passed == total_checks:
        print("🎉 Все проверки пройдены! Запускаем приложение...")
        print("=" * 60)
        
        # Запуск приложения
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
            from app import app
            print("🌐 Приложение запущено на http://127.0.0.1:5000")
            print("Нажмите Ctrl+C для остановки")
            app.run(debug=True, port=5000, host='127.0.0.1')
        except KeyboardInterrupt:
            print("\n👋 Приложение остановлено")
        except Exception as e:
            print(f"❌ Ошибка при запуске: {str(e)}")
    else:
        print(f"❌ Пройдено {checks_passed} из {total_checks} проверок")
        print("Исправьте ошибки и попробуйте снова")
        sys.exit(1)

if __name__ == "__main__":
    main() 