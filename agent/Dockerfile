FROM python:3.11-slim

WORKDIR /app

COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

COPY agent/ ./agent/
COPY evaluation/ ./evaluation/
COPY storage.py ./
COPY scheduler/ ./scheduler/
COPY .env* ./

CMD ["python", "agent/run_all.py", "start"]
