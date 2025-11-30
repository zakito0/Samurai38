FROM python:3.11-slim

WORKDIR /app

COPY requirements_autonomous.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "autonomous_app.py"]
