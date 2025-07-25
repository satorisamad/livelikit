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

"""Contextually-aware turn detection for LiveKit Agents

See https://docs.livekit.io/agents/build/turns/turn-detector/ for more information.
"""

from livekit.agents import Plugin

from .log import logger
from .version import __version__

__all__ = ["english", "multilingual", "__version__"]


class EOUPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        from transformers import AutoTokenizer  # type: ignore

        from .base import _download_from_hf_hub
        from .models import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME

        for revision in MODEL_REVISIONS.values():
            AutoTokenizer.from_pretrained(HG_MODEL, revision=revision)
            _download_from_hf_hub(HG_MODEL, ONNX_FILENAME, subfolder="onnx", revision=revision)
            _download_from_hf_hub(HG_MODEL, "languages.json", revision=revision)


Plugin.register_plugin(EOUPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
