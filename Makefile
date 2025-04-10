ingest:
	python src/ingest.py

build:
	python src/build_index.py

serve:
	streamlit run src/app.py
