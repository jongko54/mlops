# linux/amd64 CUDA runtime. The digest was resolved from the official image.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69
WORKDIR /app
COPY requirements-container.txt .
RUN python -m pip install --no-cache-dir -r requirements-container.txt
COPY src/ /app/src/
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8
USER 10001:10001
ENTRYPOINT ["python", "-m", "mlops_lab.train"]
