FROM python:3.12-slim

# System deps: build toolchain for dlib (compiled from source), plus the
# runtime libs opencv/onnxruntime need that aren't in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ArcFace (buffalo_l) model pack into the image at a fixed path so
# containers don't re-download ~300MB from the model zoo on every cold start.
ENV INSIGHTFACE_HOME=/app/.insightface
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(640, 640))"

COPY . .
RUN chmod +x entrypoint.sh

RUN python manage.py collectstatic --noinput

ENV PORT=8000
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
