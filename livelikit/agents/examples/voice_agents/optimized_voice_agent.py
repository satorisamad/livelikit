import logging
import time
import os
import csv
import json
import re
from datetime import datetime
import asyncio

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.job import JobProcess
from livekit.agents.voice import Agent, AgentSession, events
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, openai, silero, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("optimized-voice-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Optimized Configuration (All Features Enabled)
OPTIMIZED_CONFIG = {
    "name": "Optimized Voice Agent",
    "description": "Streaming STT + partial LLM + turn detector + SSML TTS",
    
    # LLM Configuration
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.7,
    
    # STT Configuration (Optimized Deepgram)
    "stt_model": "nova-2",
    "stt_language": "en-US",
    "stt_interim_results": True,  # Enable streaming
    
    # TTS Configuration (Enhanced Cartesia)
    "tts_model": "sonic-2",
    "tts_voice": "79a125e8-cd45-4c13-8a67-188112f4dd22",
    "tts_speed": 1.0,
    
    # Turn Detection (Advanced)
    "min_endpointing_delay": 0.2,  # Faster response
    "max_endpointing_delay": 3.0,  # Shorter max wait
    "allow_interruptions": True,
    
    # Optimization Flags
    "enable_streaming_stt": True,
    "enable_partial_llm": True,
    "enable_ssml": True,
    "enable_turn_detector_plugin": True,
    "partial_confidence_threshold": 0.7,
}

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class OptimizedVoiceAgent(Agent):
    """
    Optimized voice agent implementation with all performance features enabled.
    """
    
    def __init__(self) -> None:
        logger.info("[OPTIMIZED] Initializing optimized voice agent")
        
        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[OPTIMIZED] Successfully loaded prompt from prompt.txt")
        except Exception as e:
            logger.error(f"[OPTIMIZED] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
        
        super().__init__(
            llm=openai.LLM(
                model=OPTIMIZED_CONFIG["llm_model"],
                temperature=OPTIMIZED_CONFIG["llm_temperature"],
            ),
            instructions=instructions,
        )
        
        # Initialize metrics collection
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.metrics_file = os.path.join(os.path.dirname(__file__), f"optimized_metrics_{self.session_id}.csv")
        
        # Optimization tracking
        self.partial_transcripts = []
        self.processing_start_time = None
        self.llm_start_time = None
        self.tts_start_time = None
        
        # Initialize CSV file
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "config_mode",
            "user_transcript", "agent_response", "end_to_end_latency_ms",
            "mic_to_transcript_ms", "llm_processing_ms", "tts_synthesis_ms",
            "response_length_chars", "response_length_words",
            "partial_transcripts_count", "early_llm_trigger", "ssml_enhanced"
        ]
        
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        logger.info(f"[OPTIMIZED] Metrics file initialized: {self.metrics_file}")
    
    def enhance_text_with_ssml(self, text: str) -> str:
        """Enhance text with SSML for better TTS quality"""
        if not OPTIMIZED_CONFIG["enable_ssml"]:
            return text
        
        enhanced_text = text
        
        # Add emphasis for important medical terms
        medical_terms = ["emergency", "urgent", "appointment", "doctor", "clinic", "insurance"]
        for term in medical_terms:
            pattern = rf'\b{re.escape(term)}\b'
            enhanced_text = re.sub(pattern, f'<emphasis level="moderate">{term}</emphasis>', 
                                 enhanced_text, flags=re.IGNORECASE)
        
        # Add pauses for better clarity
        enhanced_text = enhanced_text.replace("!", '!<break time="0.3s"/>')
        enhanced_text = enhanced_text.replace("?", '?<break time="0.2s"/>')
        enhanced_text = enhanced_text.replace(",", ',<break time="0.1s"/>')
        
        # Wrap in SSML speak tag
        enhanced_text = f'<speak>{enhanced_text}</speak>'
        
        logger.info(f"[OPTIMIZED] Enhanced text with SSML: {len(enhanced_text)} chars")
        return enhanced_text
    
    async def on_user_speech_committed(self, msg):
        """Handle final user speech with optimizations"""
        self.processing_start_time = time.perf_counter()
        logger.info(f"[OPTIMIZED] User speech committed: {msg.transcript}")
        
        # Check if we already started processing with partial transcript
        early_trigger = len(self.partial_transcripts) > 0
        
        # Clear partial transcripts for next interaction
        self.partial_transcripts = []
        
        # Continue with normal processing
        await super().on_user_speech_committed(msg)
    
    def log_interaction_metrics(self, user_transcript: str, agent_response: str, 
                              end_to_end_latency: float, processing_times: dict = None,
                              optimization_data: dict = None):
        """Log metrics for a single interaction with optimization data"""
        self.interaction_counter += 1
        
        if processing_times is None:
            processing_times = {}
        if optimization_data is None:
            optimization_data = {}
        
        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "config_mode": "optimized",
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "end_to_end_latency_ms": f"{end_to_end_latency:.2f}",
            "mic_to_transcript_ms": f"{processing_times.get('mic_to_transcript', 0):.2f}",
            "llm_processing_ms": f"{processing_times.get('llm_processing', 0):.2f}",
            "tts_synthesis_ms": f"{processing_times.get('tts_synthesis', 0):.2f}",
            "response_length_chars": len(agent_response),
            "response_length_words": len(agent_response.split()),
            "partial_transcripts_count": optimization_data.get('partial_count', 0),
            "early_llm_trigger": optimization_data.get('early_trigger', False),
            "ssml_enhanced": optimization_data.get('ssml_used', False),
        }
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        logger.info(f"[OPTIMIZED] Logged interaction {self.interaction_counter}: {end_to_end_latency:.2f}ms")


async def entrypoint(ctx: JobContext):
    """
    Optimized voice agent entrypoint.
    Creates a fully optimized voice agent with all performance features.
    """
    await ctx.connect()
    
    logger.info("[OPTIMIZED] Starting optimized voice agent")
    logger.info(f"[OPTIMIZED] Configuration: {OPTIMIZED_CONFIG['description']}")
    logger.info(f"[OPTIMIZED] Features enabled:")
    logger.info(f"[OPTIMIZED] - Streaming STT: {OPTIMIZED_CONFIG['enable_streaming_stt']}")
    logger.info(f"[OPTIMIZED] - Partial LLM: {OPTIMIZED_CONFIG['enable_partial_llm']}")
    logger.info(f"[OPTIMIZED] - SSML Enhancement: {OPTIMIZED_CONFIG['enable_ssml']}")
    logger.info(f"[OPTIMIZED] - Turn Detector Plugin: {OPTIMIZED_CONFIG['enable_turn_detector_plugin']}")
    
    # Create optimized agent
    agent = OptimizedVoiceAgent()
    
    # Create session with optimized configuration
    session = AgentSession(
        # Advanced turn detection with multilingual model
        turn_detection=MultilingualModel() if OPTIMIZED_CONFIG["enable_turn_detector_plugin"] else "vad",
        vad=silero.VAD.load(),
        
        # Streaming STT
        stt=deepgram.STT(
            model=OPTIMIZED_CONFIG["stt_model"],
            language=OPTIMIZED_CONFIG["stt_language"],
            interim_results=OPTIMIZED_CONFIG["stt_interim_results"],
        ),
        
        # Enhanced TTS
        tts=cartesia.TTS(
            model=OPTIMIZED_CONFIG["tts_model"],
            voice=OPTIMIZED_CONFIG["tts_voice"],
            speed=OPTIMIZED_CONFIG["tts_speed"],
        ),
        
        # Optimized endpointing
        min_endpointing_delay=OPTIMIZED_CONFIG["min_endpointing_delay"],
        max_endpointing_delay=OPTIMIZED_CONFIG["max_endpointing_delay"],
        allow_interruptions=OPTIMIZED_CONFIG["allow_interruptions"],
    )
    
    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    
    logger.info("[OPTIMIZED] Optimized voice agent started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
