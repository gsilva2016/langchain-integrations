from langchain_openvino_diarize import OpenVINOSpeechDiarizeLoader

# Env variable set manually
OPENVINO_DIARIZE_API_BASE = "http://127.0.0.1:8001/diarize"
print("Diarize Test Started")
loader = OpenVINOSpeechDiarizeLoader(
    api_base = OPENVINO_DIARIZE_API_BASE,
    file_path = "./audio.mp3",
    # below are ignored for API requests
    model_id = "pyannote/speaker-diarization-3.1",
    device = "GPU", # GPU
    batch_size = 1,
)

docs = loader.load()
print(docs)
