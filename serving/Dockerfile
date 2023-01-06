FROM python:latest

COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY preprocess.py .
COPY train.py .

COPY config.json .
ADD encoders /encoders
ADD models /models

COPY app.py .
COPY utils.py .
# CMD ["python3","app.py"]
CMD gunicorn --bind 0.0.0.0:3000 --timeout=150 app:app -w 5
EXPOSE 3000
