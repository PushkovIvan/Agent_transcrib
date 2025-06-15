document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const timerDisplay = document.querySelector('.timer');
    const statusDisplay = document.querySelector('.status');
    const diskSpaceDisplay = document.querySelector('.disk-space');
    const recordingsList = document.getElementById('recordingsList');
    
    let mediaRecorder;
    let audioChunks = [];
    let startTime;
    let timerInterval;
    let tempFilename;
    let finalFilename;
    
    // Проверка поддержки MediaRecorder
    if (!window.MediaRecorder) {
        statusDisplay.textContent = 'Ваш браузер не поддерживает запись аудио. Пожалуйста, используйте последнюю версию Safari или Chrome.';
        recordButton.disabled = true;
        return;
    }
    
    // Инициализация
    loadRecordings();
    updateDiskSpace();
    setInterval(updateDiskSpace, 60000); // Обновлять место на диске каждую минуту
    
    // Обработчики событий
    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    
    // Функции
    async function startRecording() {
        try {
            statusDisplay.textContent = 'Запрашиваю разрешение на микрофон...';
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                }
            });
            
            // Получаем имена файлов с сервера
            const response = await fetch('/start_recording', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.status !== 'success') {
                throw new Error(data.message || 'Ошибка сервера');
            }
            
            tempFilename = data.temp_filename;
            finalFilename = data.filename;
            
            // Настройка MediaRecorder
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 128000
            });
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                try {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await saveRecording(audioBlob);
                } catch (error) {
                    console.error('Ошибка при сохранении:', error);
                    statusDisplay.textContent = `Ошибка сохранения: ${error.message}`;
                } finally {
                    audioChunks = [];
                    stream.getTracks().forEach(track => track.stop());
                }
            };
            
            mediaRecorder.onerror = (event) => {
                console.error('Ошибка записи:', event.error);
                statusDisplay.textContent = `Ошибка записи: ${event.error}`;
                stopRecording();
            };
            
            // Старт записи
            mediaRecorder.start(5000); // Получаем данные каждые 5 секунд
            startTime = Date.now();
            updateTimer();
            timerInterval = setInterval(updateTimer, 1000);
            
            recordButton.disabled = true;
            stopButton.disabled = false;
            statusDisplay.textContent = 'Идет запись...';
            
        } catch (error) {
            console.error('Ошибка при запуске записи:', error);
            statusDisplay.textContent = `Ошибка: ${error.message}`;
            recordButton.disabled = false;
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            clearInterval(timerInterval);
            recordButton.disabled = false;
            stopButton.disabled = true;
            statusDisplay.textContent = 'Завершаю сохранение...';
        }
    }
    
    function updateTimer() {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const hours = Math.floor(elapsed / 3600).toString().padStart(2, '0');
        const minutes = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        timerDisplay.textContent = `${hours}:${minutes}:${seconds}`;
    }
    
    async function saveRecording(audioBlob) {
        try {
            statusDisplay.textContent = 'Сохранение MP3...';
            
            const formData = new FormData();
            formData.append('audio', audioBlob, tempFilename);
            formData.append('temp_filename', tempFilename);
            formData.append('filename', finalFilename);
            
            const response = await fetch('/save_recording', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status !== 'success') {
                throw new Error(data.message || 'Ошибка сохранения');
            }
            
            statusDisplay.textContent = 'Запись успешно сохранена как MP3';
            loadRecordings();
            updateDiskSpace();
            
        } catch (error) {
            console.error('Ошибка при сохранении:', error);
            let errorMsg = `Ошибка: ${error.message}`;
            
            if (error.response) {
                const errorData = await error.response.json();
                if (errorData.fallback_file) {
                    errorMsg += ` (оригинал сохранен как ${errorData.fallback_file})`;
                }
            }
            
            statusDisplay.textContent = errorMsg;
        }
    }
    
    async function loadRecordings() {
        try {
            // В реальном приложении здесь должен быть запрос к API
            // Для демо просто очищаем список
            recordingsList.innerHTML = '<li>Записи будут отображаться здесь</li>';
        } catch (error) {
            console.error('Ошибка загрузки списка:', error);
        }
    }
    
    async function updateDiskSpace() {
        try {
            // В реальном приложении запрашиваем у сервера свободное место
            diskSpaceDisplay.textContent = 'Место на диске: достаточно';
        } catch (error) {
            console.error('Ошибка проверки места:', error);
        }
    }
    
    // Обработка закрытия страницы
    window.addEventListener('beforeunload', (event) => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            event.preventDefault();
            event.returnValue = 'У вас есть активная запись. Вы уверены, что хотите уйти?';
            return event.returnValue;
        }
    });
});