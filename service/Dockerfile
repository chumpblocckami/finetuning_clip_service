FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /app

COPY requirements.txt requirements.txt
COPY create_dataset.py .
COPY dataset.py .
COPY finetuning.py .
COPY run.sh .

RUN pip install -r requirements.txt

CMD ["sh","run.sh"]