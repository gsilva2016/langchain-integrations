"""Test embedding model integration."""

from typing import Type

from langchain_openvino_clip.embeddings import OpenVINOClipEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[OpenVINOClipEmbeddings]:
        return OpenVINOClipEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
