# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI plugin for LiveKit Agents

Support for OpenAI Realtime API, LLM, TTS, and STT APIs.

Also includes support for a large number of OpenAI-compatible APIs including Azure OpenAI, Cerebras,
Fireworks, Perplexity, Telnyx, xAI, Ollama, and DeepSeek.

See https://docs.livekit.io/agents/integrations/openai/ and
https://docs.livekit.io/agents/integrations/llm/ for more information.
"""

from . import realtime
from .embeddings import EmbeddingData, create_embeddings
from .llm import LLM, LLMStream
from .models import STTModels, TTSModels, TTSVoices
from .stt import STT
from .tts import TTS
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "LLM",
    "LLMStream",
    "STTModels",
    "TTSModels",
    "TTSVoices",
    "create_embeddings",
    "EmbeddingData",
    "realtime",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class OpenAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(OpenAIPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
