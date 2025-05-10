FROM python:3.12-slim

WORKDIR /app

COPY requirement.txt ./
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "route:app", "--host", "0.0.0.0", "--port", "8000"]