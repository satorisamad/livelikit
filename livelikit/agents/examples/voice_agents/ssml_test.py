import logging
import time
from datetime import datetime
import re
import asyncio

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.job import JobProcess
from livekit.protocol.agent import JobType
from livekit.agents.llm import ChatContext
from livekit.agents.voice import Agent, AgentSession, agent, events
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("livekit.agents")

load_dotenv()




def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    class MyAgent(Agent):
        def __init__(self) -> None:
            super().__init__(
                llm=openai.LLM(model="gpt-4o-mini"),
                instructions='''You are a voice assistant created by LiveKit.
Your interface with users is voice. Be conversational and concise.

Use the following tags to control the TTS voice:
- `[SPEED:value]text` where `value` can be `slowest`, `slow`, `normal`, `fast`, or `fastest`.
- `[EMOTION:name:level,...]text` where `name` can be `anger`, `positivity`, `surprise`, `sadness`, `curiosity` and `level` can be `low`, `high`, `highest`.
- `[RESET]text` to reset to the default voice.

Example: `[SPEED:fast]I can talk very quickly! [EMOTION:surprise:high]Isn't that amazing?`
'''
            )

        async def on_reply(self, reply):
            full_response = reply.text
            logger.info("LLM response with tags: %s", full_response)

            segments = self._parse_tags(full_response)

            for segment in segments:
                if segment["text"]:
                    await self._tts(segment["text"], ssml_wrap=segment["ssml"])

        def _parse_tags(self, text: str) -> list[dict]:
            segments = []
            pattern = r'(\[(SPEED|EMOTION|RESET):?[^\]]*\])'
            matches = list(re.finditer(pattern, text))

            start_index = 0
            for match in matches:
                pre_text = text[start_index:match.start()]
                if pre_text:
                    segments.append({"text": pre_text, "ssml": False})

                tag = match.group(1)
                if '[RESET]' in tag:
                    segments.append({"text": "", "ssml": "<prosody rate='medium'>"})
                elif '[SPEED:' in tag:
                    try:
                        speed_val = tag.split(':')[1].strip(']').lower()
                        rate_map = {
                            'slowest': 'x-slow',
                            'slow': 'slow',
                            'normal': 'medium',
                            'fast': 'fast',
                            'fastest': 'x-fast'
                        }
                        rate = rate_map.get(speed_val, 'medium')
                        segments.append({"text": "", "ssml": f"<prosody rate='{rate}'>"})
                    except IndexError:
                        logger.warning("Malformed SPEED tag: %s", tag)
                elif '[EMOTION:' in tag:
                    try:
                        emotion_str = tag.split(':')[1].strip(']').lower()
                        # Extreme test parameters
                        if 'sad' in emotion_str:
                            segments.append({"text": "", "ssml": "<prosody rate='x-slow' pitch='x-low'>"})
                        elif any(e in emotion_str for e in ['happy', 'excite']):
                            segments.append({"text": "", "ssml": "<prosody rate='x-fast' pitch='+30st'>"})
                        elif 'anger' in emotion_str:
                            segments.append({"text": "", "ssml": "<prosody rate='fast' pitch='+50st'>"})
                        else:
                            segments.append({"text": "", "ssml": "<prosody rate='medium'>"})
                    except IndexError:
                        logger.warning("Malformed EMOTION tag: %s", tag)

                start_index = match.end()

            remaining_text = text[start_index:]
            if remaining_text:
                segments.append({"text": remaining_text, "ssml": False})

            return segments

        async def _tts(self, text: str, ssml_wrap: str = ""):
            if ssml_wrap:
                ssml_text = f"<speak>{ssml_wrap}{text}</prosody></speak>"
                logger.debug(f"Sending SSML to TTS: {ssml_text}")
                await self.session.tts.synthesize(ssml_text)
            else:
                logger.debug(f"Sending plain text to TTS: {text}")
                await self.session.tts.synthesize(text)

        async def say(self, text: str):
            segments = self._parse_tags(text)
            for segment in segments:
                if segment["text"]:
                    await self._tts(segment["text"], ssml_wrap=segment["ssml"])

    session = AgentSession(
        turn_detection=MultilingualModel(unlikely_threshold=0.05),
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),
        tts=deepgram.TTS(
            model="aura-2-andromeda-en",
            encoding="linear16",
            sample_rate=24000
        ),
    )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: events.UserInputTranscribedEvent):
        if event.is_final:
            session.generate_reply()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))