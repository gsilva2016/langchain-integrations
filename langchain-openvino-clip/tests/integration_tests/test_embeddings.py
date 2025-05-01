"""Test OpenvinoClip embeddings."""

from typing import Type

from langchain_openvino_clip.embeddings import OpenvinoClipEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OpenvinoClipEmbeddings]:
        return OpenvinoClipEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
