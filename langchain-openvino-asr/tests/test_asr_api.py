from langchain_openvino_asr import OpenVINOSpeechToTextLoader

# Env variable set manually
OPENVINO_ASR_API_BASE = "http://127.0.0.1:8000/transcribe"
print("ASR Test Started")
loader = OpenVINOSpeechToTextLoader(
    api_base = OPENVINO_ASR_API_BASE,
    file_path = "./audio.mp3",
    # below are ignored for API requests
    model_id = "distil-whisper/distil-small.en",
    device = "GPU", # GPU
    return_timestamps = True,
    return_language = "en",
    chunk_length_s = 30,
    load_in_8bit = True,
    batch_size = 1,
)

docs = loader.load()
print(docs)
