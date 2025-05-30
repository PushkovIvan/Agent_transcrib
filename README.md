# Agent_transcrib

Агент, позволяющий получать транскрибацию с диаризацией участников разговора.
После отработки пайплайна есть возможность скачать как транскрибацию, так и саммари от Гигачата.



Указываем ключ к Гигачату в конфиге:

    token:
        gigachat: OD...

Базовые зависимости:

    pip install -r req.txt
    pip install nemo_toolkit
    pip install git+https://github.com/openai/whisper.git
    brew install ffmpeg

Первым делом лучше зайти в test.ipynb и выполнить все ячейки, подгрузить модель whisper и проверить гигу.

Затем запускаем:

    streamlit run temp.py