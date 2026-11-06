#!/bin/sh
set -eu

MODEL_PATH="/app/models/best_model.pkl"
MODEL_POINTER="/app/models/best_model.pkl.dvc"

echo "Starting Diabetes Prediction Assistant..."

if [ ! -f "$MODEL_PATH" ]; then
    echo "Selected model not found locally."
    echo "Restoring model from the configured DVC S3 remote..."

    if ! dvc pull "$MODEL_POINTER"; then
        echo ""
        echo "ERROR: Unable to restore the selected model from DVC."
        echo "Make sure the container has AWS credentials with access"
        echo "to the project's S3 DVC remote."
        echo ""
        echo "Required environment variables are typically:"
        echo "  AWS_ACCESS_KEY_ID"
        echo "  AWS_SECRET_ACCESS_KEY"
        echo "  AWS_DEFAULT_REGION"
        echo ""
        exit 1
    fi
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: DVC completed but the model file was not restored."
    exit 1
fi

echo "Model is ready."
echo "Launching Streamlit..."

exec python -m streamlit run src/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501
