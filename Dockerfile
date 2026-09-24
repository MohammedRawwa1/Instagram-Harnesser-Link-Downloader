FROM python:3.12-slim

# ffmpeg is bundled via imageio-ffmpeg, but keep system ffmpeg as a fallback
# and for anything that expects it on PATH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY groq ./groq

RUN mkdir -p /data/instagram-groq
ENV WORK_DIR=/data/instagram-groq

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]