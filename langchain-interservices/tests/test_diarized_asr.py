from langchain_openvino_diarize import OpenVINOSpeechDiarizeLoader
from langchain_openvino_asr import OpenVINOSpeechToTextLoader
from langchain.docstore.document import Document
#import numpy as np
#import sys
#import json
import requests
from langchain_core.load import loads

print("Diarized ASR Test Started")
# Env variable set manually
INTERSERVICES_DIARIZED_ASR_API_BASE = "http://127.0.0.1:8002/diarized_asr"

request = { 'file': open("./audio.mp3", 'rb')}
response = requests.post(url=INTERSERVICES_DIARIZED_ASR_API_BASE, files=request)
docs = loads(response.content)  # Document objects
print(docs)
