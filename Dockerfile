# Supply Chain Inventory Rebalancer — Hugging Face Spaces
FROM python:3.11-slim

# HF Spaces requires non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install deps as user (avoids pip root warning)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user server/ ./server/
COPY --chown=user baseline.py .
COPY --chown=user inference.py .
COPY --chown=user openenv.yaml .
COPY --chown=user pyproject.toml .
COPY --chown=user uv.lock .

# HuggingFace token will be injected via Space Secrets at runtime
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

EXPOSE 7860

LABEL org.opencontainers.image.title="Supply Chain Inventory Rebalancer"
LABEL org.opencontainers.image.description="OpenEnv RL benchmark — LLM planning over warehouse logistics"
LABEL space_sdk="docker"

# Serve FastAPI app + dashboard on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
