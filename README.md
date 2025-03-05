# LangChain* Integrations

Integrations are a core component of LangChain and this repository provides several that are crucial for building LLM/VLM/LMM applications. For more information on LangChain* Integrations please review the [contributing integrations guide.](https://python.langchain.com/docs/contributing/how_to/integrations/)

| Name  | Description |
| ------------- | ------------- |
| [OpenAIText2SpeechTool](langchain-openai-tts)  |  Performs Text-To-Speech using OpenAI's Speech API. Capable of file playback or live streaming as  LangChain* tool.  |
| [OpenVINOSpeechToTextLoader](langchain-openvino-asr) | A document loader for performing Speech-To-Text using OpenVINO™ and loads the transcript into documents. |
| [VideoChunkLoader](langchain-videochunk)  | A document loader which performs chunking of videos which can be used by VLMs. Two chunking strategies are possible using either `specific intervals` or `sliding window`.   |
