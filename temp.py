import streamlit as st
import whisper
import os
import torch
import traceback
import tempfile
import numpy as np
import torchaudio
import librosa
import soundfile as sf
from st_audiorec import st_audiorec
from langchain_gigachat.chat_models import GigaChat
from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForAudioFrameClassification
from sklearn.cluster import AgglomerativeClustering
from config import CONFIG

giga = GigaChat(
    scope='GIGACHAT_API_CORP',
    credentials=CONFIG["token"]["gigachat"],
    verify_ssl_certs=False,
    model="GigaChat-2-Max"
)

torch.classes.__path__ = []

def generate_detailed_analysis_prompt(transcription_text):
    prompt = f"""
Анализ текста разговора. Выполни следующие задачи:
1. Выдели основные темы/вопросы, которые обсуждались
2. Для каждой темы приведи 1-2 ключевых тезиса
3. Определи общий тон разговора (нейтральный, позитивный, конфликтный и т.д.)
4. Отметь важные решения или выводы, если они есть

Текст для анализа:
{transcription_text}

Формат ответа:
**Основные темы:**
1. [Тема] 
   - [Ключевые моменты]
   - [Контекст/Детали]

**Общий тон:** [описание тона]

**Ключевые выводы/решения:**
- [вывод 1]
- [вывод 2]
"""
    return prompt

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("large")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus")
    embedding_model = UniSpeechSatForAudioFrameClassification.from_pretrained("microsoft/unispeech-sat-base-plus")
    
    return whisper_model, feature_extractor, embedding_model

def analyze_transcription(text):
    try:
        prompt = generate_detailed_analysis_prompt(text)
        analysis_result = giga.invoke(prompt)
        return analysis_result.content
    except Exception as e:
        st.error(f"Ошибка при анализе текста: {str(e)}")
        return None

def save_audio(audio_bytes, filename):
    try:
        if not audio_bytes or len(audio_bytes) < 1024:
            st.warning("Аудиоданные слишком короткие или отсутствуют")
            return None
            
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        
        if os.path.getsize(filename) < 1024:
            st.warning("Созданный аудиофайл слишком мал")
            return None
            
        return filename
    except Exception as e:
        st.error(f"Ошибка при сохранении аудио: {str(e)}")
        return None
    
def cleanup_files(*files):
    for file in files:
        try:
            if file and os.path.exists(file):
                os.remove(file)
        except Exception as e:
            st.warning(f"Не удалось удалить файл {file}: {str(e)}")

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
        st.error(f"Ошибка при извлечении эмбеддингов: {str(e)}")
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
        st.error(f"Ошибка при сегментации аудио: {str(e)}")
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
        st.error(f"Ошибка при кластеризации: {str(e)}")
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
        st.error(f"Ошибка при транскрипции: {str(e)}")
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

def main():
    st.title("Умный помощник по транскрибации переговоров (Whisper + UniSpeechSAT)")
    whisper_model, feature_extractor, embedding_model = load_models()
    st.write("Нажмите на Start Recording, чтобы начать запись:")
    audio_bytes = st_audiorec()
    
    if audio_bytes and len(audio_bytes) > 1024:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                audio_path = tmp_file.name
            
            # st.audio(audio_bytes, format="audio/wav")
            
            with st.spinner("Анализ аудио..."):
                segments = segment_audio(audio_path)
                
                if segments:
                    embeddings = []
                    for start, end, segment in segments:
                        segment_path = f"temp_segment_{start:.1f}_{end:.1f}.wav"
                        sf.write(segment_path, segment, 48000, 'PCM_24')
                        
                        emb = extract_embeddings(segment_path, feature_extractor, embedding_model)
                        if emb is not None:
                            embeddings.append(emb)
                        
                        cleanup_files(segment_path)
                    
                    if embeddings:
                        labels = cluster_speakers(np.array(embeddings))
                        
                        if labels is not None:
                            tagged_segments = transcribe_with_speakers(
                                whisper_model, audio_path, segments, labels
                            )
                            
                            if tagged_segments:
                                transcription = format_transcription(tagged_segments)
                                with st.spinner("Анализируем основные темы..."):
                                    analysis = analyze_transcription(transcription)

                                st.subheader("Результат транскрибации")
                                st.markdown(transcription)  

                                st.subheader("Результат анализа гигачатом:")
                                st.markdown(analysis)
                                
                                st.download_button(
                                    label="Скачать транскрипцию",
                                    data=transcription,
                                    file_name="transcription.txt",
                                    mime="text/plain"
                                )

                                st.download_button(
                                    label="Скачать отчет Гигачата",
                                    data=analysis,
                                    file_name="analysis.txt",
                                    mime="text/plain"
                                )
            
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            st.text(traceback.format_exc())
        finally:
            cleanup_files(audio_path)

if __name__ == "__main__":
    main()