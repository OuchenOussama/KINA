FROM python:3.9-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -m nltk.downloader stopwords punkt

COPY . /usr/src/app/

EXPOSE 5000

CMD ["python", "src/app.py"]