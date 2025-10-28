FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY pipeline_v1.bin .
COPY predict_api.py .

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

EXPOSE 8000


CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
