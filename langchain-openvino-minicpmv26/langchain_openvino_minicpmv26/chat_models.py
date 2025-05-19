"""OpenvinoMinicpmv26 chat models."""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from langchain_core.messages import BaseMessage, AIMessage

from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import openvino_genai
from decord import VideoReader, cpu
from openvino import Tensor

def encode_video(video_path: str,
                 max_num_frames: int = 64,
                 resolution: list = []) -> list:
    def uniform_sample(l: list, n: int) -> list:
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if len(resolution) != 0:
        vr = VideoReader(video_path, width=resolution[0],
                         height=resolution[1], ctx=cpu(0))
    else:
        vr = VideoReader(video_path, ctx=cpu(0))

    frame_idx = [i for i in range(0, len(vr), max(1, int(len(vr) / max_num_frames)))]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()

    frames = [Tensor(v.astype('uint8')) for v in frames]
    print('Num frames sampled:', len(frames))
    return frames

def streamer(subword: str) -> bool:
    '''
    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return False".

class ChatOpenvinoMinicpmv26(BaseChatModel):
    """OpenvinoMinicpmv26 chat model integration.

    The default implementation processes video and text inputs to generate responses using the OpenvinoMinicpmv26 model.

    Setup:
        Install ``langchain-openvino-minicpmv26`` and set the environment variable ``OPENVINOMINICPMV26_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-openvino-minicpmv26
            export OPENVINOMINICPMV26_API_KEY="your-api-key"

    Key init args — completion params:
        model_name: str
            Name or path to MiniCPMV_2_6 model to use. Default is "MiniCPM_INT8".
        device: str
            Device to run the model on. Default is "GPU".
        max_new_tokens: Optional[int]
            Maximum number of tokens to generate. Default is 500.
        max_num_frames: Optional[int]
            Maximum number of video frames to process. Default is 32.
        resolution: Optional[list[int]]
            Resolution of video frames as [width, height]. Default is [480, 270].

    Key init args — client params:
        api_key: Optional[str]
            OpenvinoMinicpmv26 API key. If not passed, it will be read from the environment variable ``OPENVINOMINICPMV26_API_KEY``.

    Instantiate:
        .. code-block:: python

            from langchain_openvino_minicpmv26 import ChatOpenvinoMinicpmv26

            llm = ChatOpenvinoMinicpmv26(
                model_name="MiniCPM_INT8",
                device="GPU",
                max_new_tokens=500,
                max_num_frames=32,
                resolution=[480, 270]
            )

    Invoke:
        .. code-block:: python

            from langchain_core.messages import HumanMessage

            messages = [
                HumanMessage(content="video.mp4, What is happening in this video?"),
                HumanMessage(content=", Translate 'I love programming' to French."),
            ]
            ai_responses = llm.invoke(messages)
            for response in ai_responses.generations:
                print(response.message.content)

            # Example output:
            # "The video shows a person walking in a park."
            # "J'aime programmer."

    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the weather in this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            print(ai_msg.generations[0].message.content)

            # Example output:
            # "The weather in the image appears sunny with clear skies."
    """  # noqa: E501

    model_name: str = "MiniCPM_INT8"
    device: str = "GPU"
    # : Optional[int] = 500
    max_num_frames: Optional[int] = 32
    resolution: Optional[list[int]] = [480, 270]
    ovpipe: object = None
    config: object = None
    def __init__(self, model_name: str, device: str = "GPU",
                 max_new_tokens: Optional[int] = 500,
                 max_num_frames: Optional[int] = 32,
                 resolution: Optional[list[int]] = [480, 270]):        
        super().__init__()

        # Start ov genai pipeline
        enable_compile_cache = dict()
        if "GPU" == device:
            enable_compile_cache["CACHE_DIR"] = "vlm_cache"
        self.ovpipe = openvino_genai.VLMPipeline(model_name, device, **enable_compile_cache)
        
        # Set variables for inference 
        self.model_name = model_name
        self.config = openvino_genai.GenerationConfig()
        self.config.max_new_tokens = max_new_tokens
        self.max_num_frames = max_num_frames
        self.resolution = resolution

        print(f"Running model: {model_name} on device: {device} max_new_tokens: {max_new_tokens}")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-openvino-minicpmv26"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
        """
        self.ovpipe.start_chat()
        results = []
        for message in messages:
            # Extract content from the message
            content = message.content
            video_fh, question = content.split(',', 1)

            # Process text only
            if video_fh.strip() == '':
                generated_text = self.ovpipe.generate(
                    question.strip(),
                    generation_config=self.config,
                    streamer=streamer
                )
            # Process video and text
            else:
                frames = encode_video(
                    video_fh.strip(),
                    self.max_num_frames,
                    resolution=self.resolution
                )
                generated_text = self.ovpipe.generate(
                    question.strip(),
                    images=frames,
                    generation_config=self.config,
                    streamer=streamer
                )
            print("Generated text:")
            print(str(generated_text))
            ai_message = AIMessage(content=str(generated_text))
            results.append(ai_message)

        # Combine results into a single ChatResult
        self.ovpipe.finish_chat()
        return ChatResult(generations=[ChatGeneration(message=result) for result in results])
