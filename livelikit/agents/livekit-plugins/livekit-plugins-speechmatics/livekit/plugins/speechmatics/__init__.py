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

"""Speechmatics STT plugin for LiveKit Agents

See https://docs.livekit.io/agents/integrations/stt/speechmatics/ for more information.
"""

from .log import logger
from .stt import STT, SpeechStream
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "logger",
    "__version__",
]

from livekit.agents import Plugin


class SpeechmaticsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)


Plugin.register_plugin(SpeechmaticsPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
