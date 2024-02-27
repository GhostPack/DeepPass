FROM python:3.13.0a3-slim
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
#run python -c "import nltk; nltk.download('punkt')"
CMD ["flask", "run"]