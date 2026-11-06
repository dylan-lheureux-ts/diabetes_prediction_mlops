FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Copy only the files required by the Streamlit application.
COPY src/ ./src/
COPY metrics/best_model_selection.json ./metrics/best_model_selection.json
COPY models/best_model.pkl ./models/best_model.pkl

# Run the application as a non-root user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
