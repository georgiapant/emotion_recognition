FROM python:3.7


RUN apt-get update && apt-get install nano


RUN mkdir ./emotional-index

WORKDIR /emotional-index

COPY ./ /emotional-index

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.7 -m pip install -r ./src/requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

CMD ["python3", "./src/emotion_main.py"]