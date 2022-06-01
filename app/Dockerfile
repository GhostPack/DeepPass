FROM python:3.8-slim-buster
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
#run python -c "import nltk; nltk.download('punkt')"
CMD ["flask", "run"]