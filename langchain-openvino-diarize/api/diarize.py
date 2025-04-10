from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
import argparse
import sys
from typing import List
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document
from langchain_openvino_diarize import OpenVINOSpeechDiarizeLoader
from langchain_core.load import dumpd, dumps

app = FastAPI()
loader = None

@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Hello. Use /diarize to perform diarization"
    }

@app.post("/diarize")
def diarize(file: UploadFile = File(...)):
    """
    Endpoint for calling diarize. Input is path to a file
    """
    docs = []
    loader.file_path = file.filename

    try:
        docs = loader.load()
    except Exception as exc:
        raise Exception(
            "Error occurred calling diarize load."
        ) from exc

    #print("done microservice")
    return dumpd(docs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--model_id", nargs="?", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--device", nargs="?", default="GPU")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--endpoint_uri", nargs="?", default="")
    parser.add_argument("--endpoint_port", default=0, type=int)
    args = parser.parse_args()

    if args.endpoint_uri == "" and args.endpoint_port == 0:
        sys.exit("Error: endpoint_uri and endpoint_port must be specified.")

    loader = OpenVINOSpeechDiarizeLoader(
        file_path = "",
        model_id = args.model_id,
        device = args.device,
        batch_size = args.batch_size)

    uvicorn.run(app, host=args.endpoint_uri, port=args.endpoint_port)
