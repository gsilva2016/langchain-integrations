"""Test OpenvinoClip embeddings."""
from langchain_openvino_multimodal import OpenVINOClipEmbeddings, OpenVINOBgeEmbeddings
from PIL import Image
import numpy as np

if __name__ == "__main__":
    
    # Test OpenVINO CLIP Embeddings
    image_1_path = "cats.jpg"
    image_2_path = "mountain.jpg"
    
    ov_clip = OpenVINOClipEmbeddings(
        model_id="openai/clip-vit-base-patch32",
        device="GPU",
    )
    
    text = "a photo of my two cats with a dog and many other things"
    text_embedding = ov_clip.embed_query(text)
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Embed single image
    image_embedding = ov_clip.embed_image(image_1_path)
    print(f"Image embedding shape: {image_embedding.shape}")
    
    # Embed multiple images
    image_embeddings = ov_clip.embed_images([image_1_path, image_2_path])
    print(f"Embedded {len(image_embeddings)} images")
    
    # Convert image to numpy array
    image_1 = Image.open(image_1_path)
    image_1 = np.array(image_1)
    
    image_2 = Image.open(image_2_path)
    image_2 = np.array(image_2)
    
    # Embed input: numpy array
    image_embedding_arr = ov_clip.embed_image(image_1)
    # Embed input: list of numpy arrays
    image_embeddings_arr = ov_clip.embed_images([image_1, image_2])
    
    # Check if the embeddings are the same
    assert np.array_equal(image_embeddings_arr[0], image_embeddings[0]), "Cat Image embeddings do not match"   
    assert np.array_equal(image_embeddings_arr[1], image_embeddings[1]), "Mountain Image embeddings do not match" 
    
    # Test OpenVINO BGE Embeddings
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "CPU"}
    encode_kwargs = {"normalize_embeddings": True}
    ov_embeddings = OpenVINOBgeEmbeddings(
        model_name_or_path=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    embedding = ov_embeddings.embed_query("hi this is harrison")
    print(f"Embedding shape: {len(embedding)}")
    assert len(embedding) == 384, "Embedding length is not 384"


