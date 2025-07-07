import logging
import time
import os
import csv
import json
from datetime import datetime
import asyncio

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.job import JobProcess
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, openai, silero, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("baseline-voice-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Baseline Configuration (No Optimizations)
BASELINE_CONFIG = {
    "name": "Baseline Voice Agent",
    "description": "Basic VAD + non-streaming STT + simple LLM + basic TTS",
    
    # LLM Configuration
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.7,
    
    # STT Configuration (Basic Deepgram)
    "stt_model": "nova-2",
    "stt_language": "en-US",
    "stt_interim_results": False,  # No streaming for baseline
    
    # TTS Configuration (Basic Cartesia)
    "tts_model": "sonic-2",
    "tts_voice": "79a125e8-cd45-4c13-8a67-188112f4dd22",
    "tts_speed": 1.0,  # Normal speed
    
    # Turn Detection (Basic VAD)
    "min_endpointing_delay": 0.5,
    "max_endpointing_delay": 6.0,
    "allow_interruptions": True,
}

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class BaselineVoiceAgent(Agent):
    """
    Baseline voice agent implementation for benchmarking.
    Uses basic configuration without optimizations.
    """
    
    def __init__(self) -> None:
        logger.info("[BASELINE] Initializing baseline voice agent")
        
        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[BASELINE] Successfully loaded prompt from prompt.txt")
        except Exception as e:
            logger.error(f"[BASELINE] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
        
        super().__init__(
            llm=openai.LLM(
                model=BASELINE_CONFIG["llm_model"],
                temperature=BASELINE_CONFIG["llm_temperature"],
            ),
            instructions=instructions,
        )
        
        # Initialize metrics collection
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.metrics_file = os.path.join(os.path.dirname(__file__), f"baseline_metrics_{self.session_id}.csv")
        self.metrics_data = []
        
        # Initialize CSV file
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "config_mode",
            "user_transcript", "agent_response", "end_to_end_latency_ms",
            "mic_to_transcript_ms", "llm_processing_ms", "tts_synthesis_ms",
            "response_length_chars", "response_length_words"
        ]
        
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        logger.info(f"[BASELINE] Metrics file initialized: {self.metrics_file}")
    
    def log_interaction_metrics(self, user_transcript: str, agent_response: str, 
                              end_to_end_latency: float, processing_times: dict = None):
        """Log metrics for a single interaction"""
        self.interaction_counter += 1
        
        if processing_times is None:
            processing_times = {}
        
        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "config_mode": "baseline",
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "end_to_end_latency_ms": f"{end_to_end_latency:.2f}",
            "mic_to_transcript_ms": f"{processing_times.get('mic_to_transcript', 0):.2f}",
            "llm_processing_ms": f"{processing_times.get('llm_processing', 0):.2f}",
            "tts_synthesis_ms": f"{processing_times.get('tts_synthesis', 0):.2f}",
            "response_length_chars": len(agent_response),
            "response_length_words": len(agent_response.split()),
        }
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        logger.info(f"[BASELINE] Logged interaction {self.interaction_counter}: {end_to_end_latency:.2f}ms")


async def entrypoint(ctx: JobContext):
    """
    Baseline voice agent entrypoint.
    Creates a simple, unoptimized voice agent for benchmarking.
    """
    await ctx.connect()
    
    logger.info("[BASELINE] Starting baseline voice agent")
    logger.info(f"[BASELINE] Configuration: {BASELINE_CONFIG['description']}")
    
    # Create baseline agent
    agent = BaselineVoiceAgent()
    
    # Create session with basic configuration
    session = AgentSession(
        # Basic VAD turn detection
        vad=silero.VAD.load(),
        
        # Basic STT (no streaming)
        stt=deepgram.STT(
            model=BASELINE_CONFIG["stt_model"],
            language=BASELINE_CONFIG["stt_language"],
            interim_results=BASELINE_CONFIG["stt_interim_results"],
        ),
        
        # Basic TTS
        tts=cartesia.TTS(
            model=BASELINE_CONFIG["tts_model"],
            voice=BASELINE_CONFIG["tts_voice"],
            speed=BASELINE_CONFIG["tts_speed"],
        ),
        
        # Basic endpointing
        min_endpointing_delay=BASELINE_CONFIG["min_endpointing_delay"],
        max_endpointing_delay=BASELINE_CONFIG["max_endpointing_delay"],
        allow_interruptions=BASELINE_CONFIG["allow_interruptions"],
    )
    
    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    
    logger.info("[BASELINE] Baseline voice agent started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
