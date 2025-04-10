from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
import argparse
import sys
from typing import List
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document
from langchain_openvino_asr import OpenVINOSpeechToTextLoader
from langchain_core.load import dumpd, dumps

app = FastAPI()
asr_loader = None

@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Hello. Use /diarize to perform diarization"
    }

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), diarize: bool = False):
    """
    Endpoint for calling trasncribe. Input is path to a file
    """
    #print("Microservice transcribe for file: ", file.filename)
    #print("Diarize requested: ", diarize)
    docs = []
    asr_loader.file_path = file.filename

    try:
        docs = asr_loader.load()
    except:
        print("Error occurred calling asr module.")

    return dumpd(docs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
    parser.add_argument("--device", nargs="?", default="GPU")
    parser.add_argument("--asr_batch_size", default=1, type=int)
    parser.add_argument("--asr_load_in_8bit", default=False, action="store_true")
    parser.add_argument("--endpoint_uri", nargs="?", default="")
    parser.add_argument("--endpoint_port", default=0, type=int)
    args = parser.parse_args()

    if args.endpoint_uri == "" and args.endpoint_port == 0:
        sys.exit("Error: endpoint_uri and endpoint_port must be specified.")

    asr_loader = OpenVINOSpeechToTextLoader(
        file_path = "",
        model_id = args.asr_model_id,
        device = args.device,
        return_timestamps = True,
        return_language = "en",
        chunk_length_s = 30,
        load_in_8bit = args.asr_load_in_8bit,
        batch_size = args.asr_batch_size)

    uvicorn.run(app, host=args.endpoint_uri, port=args.endpoint_port)
