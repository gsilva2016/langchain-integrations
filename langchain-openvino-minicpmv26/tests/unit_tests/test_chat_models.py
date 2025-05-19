"""Test chat model integration."""

from typing import Type

from langchain_openvino_minicpmv26.chat_models import ChatOpenvinoMinicpmv26
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatOpenvinoMinicpmv26Unit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOpenvinoMinicpmv26]:
        return ChatOpenvinoMinicpmv26

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
