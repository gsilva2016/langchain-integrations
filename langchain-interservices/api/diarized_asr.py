from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
import argparse
import sys
from typing import List
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document
from langchain_openvino_diarize import OpenVINOSpeechDiarizeLoader
from langchain_openvino_asr import OpenVINOSpeechToTextLoader
from langchain_core.load import dumpd, dumps, loads
import numpy as np

app = FastAPI()
asr_loader = None
diarize_loader = None

@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Hello."
    }

@app.post("/diarized_asr")
def diarized_asr(file: UploadFile = File(...)):
    """
    Endpoint for calling diarize. Input is path to a file
    """
    diarize_docs = asr_docs = []
    asr_loader.file_path = file.filename
    diarize_loader.file_path = file.filename

    try:
        asr_docs = loads(asr_loader.load())
        diarize_docs = loads(diarize_loader.load())
    except Exception as exc:
        raise Exception(
            "Error occurred calling load."
        ) from exc

    return dumpd(post_process(asr_docs, diarize_docs))

    #print("done microservice")
    #return dumpd(docs)

def post_process(transcript, diarize):
    end_timestamps = []
    for i, chunk in enumerate(transcript):
        ts = eval(chunk.metadata["timestamp"])
        if ts is None:
            end_timestamps.append(sys.float_info.max)
        else:
            end_timestamps.append(ts[1])
    end_timestamps = np.array(end_timestamps)

    segmented_preds = []

    for segment in diarize:
        end_time = eval(segment.metadata["stop"])
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        for i in range(upto_idx + 1):
            segmented_preds.append( Document(
                page_content=f"speaker: {segment.metadata['speaker']} stated: {transcript[i].page_content}",
                metadata = {
                    "speaker": segment.metadata['speaker'],
                    "text": transcript[i].page_content
                }
            ))

        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break

    return segmented_preds

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--diarize_model_id", nargs="?", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
    parser.add_argument("--device", nargs="?", default="GPU")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--endpoint_uri", nargs="?", default="")
    parser.add_argument("--endpoint_port", default=0, type=int)
    args = parser.parse_args()

    if args.endpoint_uri == "" and args.endpoint_port == 0:
        sys.exit("Error: endpoint_uri and endpoint_port must be specified.")

    OPENVINO_ASR_API_BASE = "http://127.0.0.1:8000/transcribe"
    asr_loader = OpenVINOSpeechToTextLoader(
        api_base = OPENVINO_ASR_API_BASE,
        file_path = "",
        model_id = ""
    )

    OPENVINO_DIARIZE_API_BASE = "http://127.0.0.1:8001/diarize"
    diarize_loader = OpenVINOSpeechDiarizeLoader(
        api_base = OPENVINO_DIARIZE_API_BASE,
        file_path = "",
        model_id = ""
    )

    uvicorn.run(app, host=args.endpoint_uri, port=args.endpoint_port)
