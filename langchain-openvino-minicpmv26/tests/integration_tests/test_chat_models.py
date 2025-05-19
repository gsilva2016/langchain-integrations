"""Test ChatOpenvinoMinicpmv26 chat model."""
import sys
import os
import argparse
from langchain_core.messages import HumanMessage

# Add langchain_openvino_minicpmv26 to the path
resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r"..\..\langchain_openvino_minicpmv26"))
sys.path.insert(0, resolved_path)
from chat_models import ChatOpenvinoMinicpmv26

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ChatOpenvinoMinicpmv26 chat model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--device", type=str, default="GPU", help="Device to run the model on (default: GPU).")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum number of tokens to generate.")
    parser.add_argument("--max_num_frames", type=int, default=1, help="Maximum number of video frames to process.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 270], help="Resolution of video frames as [width, height].")
    parser.add_argument("--prompt", type=str, default="Summarize this video.", help="Prompt to summarize the video.")

    args = parser.parse_args()

    chat_model = ChatOpenvinoMinicpmv26(
        model_name=args.model_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_num_frames=args.max_num_frames,
        resolution=args.resolution
    )
    messages = [HumanMessage(content=f"{args.video}, {args.prompt}")]
    res = chat_model.invoke(messages)
    print(res)