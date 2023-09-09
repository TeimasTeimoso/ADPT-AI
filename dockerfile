FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "src/main.py"]