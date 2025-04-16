install:
    pip install -r requirements.txt

ingest:
    python src/ingest.py

build:
    python src/build_index.py data/

run:
    streamlit run src/app.py
