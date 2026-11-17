FROM python:3.11-slim
WORKDIR /app
COPY requirements-fase5.txt .
RUN pip install --no-cache-dir -r requirements-fase5.txt
COPY fase5 ./fase5
RUN python -m fase5.train && useradd --create-home guardia && chown -R guardia:guardia /app
USER guardia
EXPOSE 8050
CMD ["uvicorn", "fase5.app:app", "--host", "0.0.0.0", "--port", "8050"]
