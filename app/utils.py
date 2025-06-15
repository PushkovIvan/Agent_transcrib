import os
import whisper
import torch
import numpy as np
import torchaudio
import librosa
import soundfile as sf
from langchain_gigachat.chat_models import GigaChat
from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForAudioFrameClassification
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import yaml
from docx import Document
from docx.shared import Inches
from datetime import datetime

# Загрузка конфигурации
def get_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

CONFIG = get_config()

# Инициализация GigaChat
giga = GigaChat(
    scope='GIGACHAT_API_CORP',
    credentials=CONFIG["token"]["gigachat"],
    verify_ssl_certs=False,
    model="GigaChat-2-Max"
)

# Глобальные переменные для кэширования моделей
_whisper_model = None
_feature_extractor = None
_embedding_model = None

def load_models():
    """Загрузка моделей для транскрибации"""
    global _whisper_model, _feature_extractor, _embedding_model
    
    if _whisper_model is None:
        _whisper_model = whisper.load_model("large-v3")
    
    if _feature_extractor is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus")
    
    if _embedding_model is None:
        _embedding_model = UniSpeechSatForAudioFrameClassification.from_pretrained("microsoft/unispeech-sat-base-plus")
    
    return _whisper_model, _feature_extractor, _embedding_model

def generate_detailed_analysis_prompt(transcription_text):
    """Генерация промпта для анализа транскрипции"""
    prompt = f"""  
Обработай транскрипт рабочего совещания. Сначала попробуй определить участников по контексту (диаризация по наитию), группируя реплики по условным ролям или именам (например: "Иван", "Участник 1", "Менеджер проекта"). Затем выдели: 1) Основные темы/проблемы (кратко суть), 2) Поручения (что, кому/роли, сроки если есть), 3) Ключевые решения, 4) Важные детали (цифры, риски, контакты). Не используй markdown. Если списки - только текстовые обозначения типа "а)" или "-". Группируй информацию по смыслу. Структура ответа: сначала условные говорящие, затем остальные категории. Если диаризация невозможна - пропусти её и выдели только суть.

Транскрипт:
{transcription_text}  

""" 
    return prompt

def analyze_transcription(text):
    """Анализ транскрипции с помощью GigaChat"""
    try:
        prompt = generate_detailed_analysis_prompt(text)
        analysis_result = giga.invoke(prompt)
        return analysis_result.content
    except Exception as e:
        print(f"Ошибка при анализе текста: {str(e)}")
        return None

def cleanup_files(*files):
    """Очистка временных файлов"""
    for file in files:
        try:
            if file and os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Не удалось удалить файл {file}: {str(e)}")

def extract_embeddings(audio_path, feature_extractor, model, sr=16000):
    """Извлечение эмбеддингов с помощью UniSpeechSat"""
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        
        if orig_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_sr, sr)
            waveform = resampler(waveform)
        
        inputs = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        
        return embeddings
    except Exception as e:
        print(f"Ошибка при извлечении эмбеддингов: {str(e)}")
        return None

def segment_audio(audio_path, segment_length=3, overlap=1, sr=16000):
    """Сегментация аудио на фрагменты"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        duration = len(y) / sr
        segments = []
        
        start = 0.0
        while start + segment_length <= duration:
            end = start + segment_length
            segment = y[int(start*sr):int(end*sr)]
            segments.append((start, end, segment))
            start += segment_length - overlap
        
        return segments
    except Exception as e:
        print(f"Ошибка при сегментации аудио: {str(e)}")
        return None

def cluster_speakers(embeddings, n_speakers=None):
    """Кластеризация говорящих"""
    try:
        if n_speakers is None:
            n_speakers = min(5, max(2, len(embeddings) // 3))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric='cosine',
            linkage='average'
        ).fit(embeddings)
        
        return clustering.labels_
    except Exception as e:
        print(f"Ошибка при кластеризации: {str(e)}")
        return None

def transcribe_with_speakers(whisper_model, audio_path, segments, labels):
    """Транскрипция с определением говорящих"""
    try:
        result = whisper_model.transcribe(audio_path, language="ru")
        tagged_segments = []
        for segment in result["segments"]:
            seg_start = segment["start"]
            seg_end = segment["end"]
            speaker = "Unknown"
            for (start, end, _), label in zip(segments, labels):
                if not (seg_end <= start or seg_start >= end):
                    speaker = f"Speaker {label+1}"
                    break
            
            tagged_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": segment["text"],
                "speaker": speaker
            })
        
        return tagged_segments
    except Exception as e:
        print(f"Ошибка при транскрипции: {str(e)}")
        return None

def format_transcription(tagged_segments):
    """Форматирование результата транскрипции"""
    formatted = []
    current_speaker = None
    
    for seg in tagged_segments:
        if seg["speaker"] != current_speaker:
            formatted.append(f"\n\n**{seg['speaker']}:**\n")
            current_speaker = seg["speaker"]
        formatted.append(seg["text"] + " ")
    
    return "".join(formatted).strip()

def create_docx_document(transcription_text, analysis_text, filename):
    """Создание документа Word с транскрипцией и анализом"""
    try:
        doc = Document()
        
        # Заголовок
        title = doc.add_heading('Транскрипция совещания', 0)
        title.alignment = 1  # По центру
        
        # Информация о файле
        doc.add_paragraph(f'Файл: {filename}')
        doc.add_paragraph(f'Дата создания: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}')
        doc.add_paragraph('')
        
        # Раздел с транскрипцией
        doc.add_heading('Транскрипция', level=1)
        doc.add_paragraph(transcription_text)
        
        # Раздел с анализом
        if analysis_text:
            doc.add_heading('Анализ совещания', level=1)
            doc.add_paragraph(analysis_text)
        
        # Сохранение документа
        doc_filename = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        doc_path = os.path.join(os.path.dirname(filename), doc_filename)
        doc.save(doc_path)
        
        return doc_path
    except Exception as e:
        print(f"Ошибка при создании документа: {str(e)}")
        return None

def process_audio_transcription(audio_path):
    """Основная функция обработки аудио для транскрибации"""
    try:
        print("Загружаем модели...")
        whisper_model, feature_extractor, embedding_model = load_models()
        
        print("Сегментируем аудио...")
        segments = segment_audio(audio_path)
        
        if not segments:
            return None, None, "Ошибка при сегментации аудио"
        
        print("Извлекаем эмбеддинги...")
        embeddings = []
        for start, end, segment in segments:
            segment_path = f"temp_segment_{start:.1f}_{end:.1f}.wav"
            sf.write(segment_path, segment, 48000, 'PCM_24')
            
            emb = extract_embeddings(segment_path, feature_extractor, embedding_model)
            if emb is not None:
                embeddings.append(emb)
            
            cleanup_files(segment_path)
        
        if not embeddings:
            return None, None, "Не удалось извлечь эмбеддинги"
        
        print("Кластеризуем спикеров...")
        labels = cluster_speakers(np.array(embeddings))
        
        if labels is None:
            return None, None, "Ошибка при кластеризации спикеров"
        
        print("Транскрибируем со спикерами...")
        tagged_segments = transcribe_with_speakers(
            whisper_model, audio_path, segments, labels
        )
        
        if not tagged_segments:
            return None, None, "Ошибка при транскрибации"
        
        print("Формируем транскрибацию...")
        transcription = format_transcription(tagged_segments)
        
        print("Вызываем GigaChat для анализа...")
        analysis = analyze_transcription(transcription)
        
        return transcription, analysis, None
        
    except Exception as e:
        print(f"Ошибка при обработке аудио: {str(e)}")
        return None, None, str(e) 