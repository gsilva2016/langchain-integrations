# langchain-summarymerge-score

This package contains the LangChain integration with OpenvinoMinicpmv26.

## Installation

```bash
pip install -U langchain-openvino-minicpmv26
```

And you should configure credentials by setting the following environment variables:

* Set HF_ACCESS_TOKEN via `export HF_ACCESS_TOKEN=<YOUR_ACCESS_TOKEN>`

## Using the ChatOpenvinoMinicpmv26 Model

The `ChatOpenvinoMinicpmv26` object allows you to process video and text inputs to generate responses using the OpenvinoMinicpmv26 model.

### Example Usage

```python
from langchain_openvino_minicpmv26 import ChatOpenvinoMinicpmv26
from langchain_core.messages import HumanMessage

# Initialize the model
chat_model = ChatOpenvinoMinicpmv26(
    model_name="MiniCPM_INT8",
    device="GPU",
    max_new_tokens=500,
    max_num_frames=32,
    resolution=[480, 270]
)

# Prepare input messages
messages = [
    HumanMessage(content="video.mp4, What is happening in this video?"),
    HumanMessage(content=", Translate 'I love programming' to French."),
]

# Invoke the model
responses = chat_model.invoke(messages)

# Print the responses
for response in responses.generations:
    print(response.message.content)
```

### Key Parameters

- `model_name`: Name or path to the MiniCPMV_2_6 model (default: `"MiniCPM_INT8"`).
- `device`: Device to run the model on (default: `"GPU"`).
- `max_new_tokens`: Maximum number of tokens to generate (default: `500`).
- `max_num_frames`: Maximum number of video frames to process (default: `32`).
- `resolution`: Resolution of video frames as `[width, height]` (default: `[480, 270]`).

### Example Output

```plaintext
"The video shows a person walking in a park."
"J'aime programmer."
```
