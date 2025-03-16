FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY test_images test_images
COPY test_masks test_masks

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Démarrer l’application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
