FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Install all project dependencies, including DVC with S3 support.
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Copy the application and the metadata needed at runtime.
COPY src/ ./src/
COPY metrics/best_model_selection.json ./metrics/best_model_selection.json

# Copy only the DVC configuration and model pointer.
# The actual model is downloaded automatically when the
# container starts.
COPY .dvc/config ./.dvc/config
COPY models/best_model.pkl.dvc ./models/best_model.pkl.dvc

# Container startup script.
COPY docker-entrypoint.sh ./docker-entrypoint.sh

# Run the application as a non-root user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app \
    && chmod +x /app/docker-entrypoint.sh

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
