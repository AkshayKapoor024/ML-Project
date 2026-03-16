FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app/

RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list \
 && sed -i 's|security.debian.org|archive.debian.org|g' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y awscli build-essential gcc

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3","application.py"]