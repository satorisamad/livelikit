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
from livekit.agents.voice import Agent, AgentSession, events
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, openai, silero, deepgram

logger = logging.getLogger("vad-tuning-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# VAD/Endpointing Parameter Configurations for Testing
VAD_CONFIGURATIONS = {
    "fast_response": {
        "name": "Fast Response",
        "description": "Minimal delays for fastest response",
        "min_endpointing_delay": 0.2,
        "max_endpointing_delay": 3.0,
        "turn_detection": "vad",
    },
    
    "balanced": {
        "name": "Balanced",
        "description": "Balanced speed vs accuracy",
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
        "turn_detection": "vad",
    },
    
    "patient": {
        "name": "Patient",
        "description": "Longer delays for natural pauses",
        "min_endpointing_delay": 1.0,
        "max_endpointing_delay": 8.0,
        "turn_detection": "vad",
    },
    
    "very_patient": {
        "name": "Very Patient",
        "description": "Maximum patience for complex speech",
        "min_endpointing_delay": 2.0,
        "max_endpointing_delay": 10.0,
        "turn_detection": "vad",
    },
    
    "stt_based": {
        "name": "STT-Based",
        "description": "Use STT confidence for turn detection",
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
        "turn_detection": "stt",
    }
}

# Current configuration to test (change this to test different settings)
CURRENT_VAD_CONFIG = "balanced"  # Change to test different configurations

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class VADTuningAgent(Agent):
    """
    Voice agent for testing different VAD/endpointing parameters.
    Collects detailed metrics on false triggers and missed endpoints.
    """
    
    def __init__(self, vad_config_name: str) -> None:
        self.vad_config = VAD_CONFIGURATIONS[vad_config_name]
        self.vad_config_name = vad_config_name
        
        logger.info(f"[VAD_TUNING] Initializing with config: {self.vad_config['name']}")
        logger.info(f"[VAD_TUNING] Description: {self.vad_config['description']}")
        logger.info(f"[VAD_TUNING] Min delay: {self.vad_config['min_endpointing_delay']}s")
        logger.info(f"[VAD_TUNING] Max delay: {self.vad_config['max_endpointing_delay']}s")
        
        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[VAD_TUNING] Successfully loaded prompt from prompt.txt")
        except Exception as e:
            logger.error(f"[VAD_TUNING] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
        
        super().__init__(
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            instructions=instructions,
        )
        
        # Initialize metrics collection
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.metrics_file = os.path.join(os.path.dirname(__file__), 
                                       f"vad_tuning_{vad_config_name}_{self.session_id}.csv")
        
        # VAD-specific tracking
        self.speech_start_times = []
        self.speech_end_times = []
        self.false_triggers = 0
        self.missed_endpoints = 0
        self.natural_pauses_detected = 0
        
        # Initialize CSV file
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "vad_config",
            "user_transcript", "agent_response", "end_to_end_latency_ms",
            "speech_detection_delay_ms", "endpointing_delay_ms", 
            "false_trigger", "missed_endpoint", "natural_pause_detected",
            "min_endpointing_delay", "max_endpointing_delay", "turn_detection_mode",
            "response_length_chars", "response_length_words"
        ]
        
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        logger.info(f"[VAD_TUNING] Metrics file initialized: {self.metrics_file}")
    
    async def on_user_speech_committed(self, msg):
        """Track speech detection and endpointing metrics"""
        current_time = time.perf_counter()
        
        # Calculate speech detection timing
        speech_detection_delay = 0  # Would need to track from actual speech start
        endpointing_delay = 0  # Would need to track from speech end to commitment
        
        # Analyze for false triggers and missed endpoints
        # This is simplified - in practice you'd need more sophisticated detection
        transcript_length = len(msg.transcript.strip())
        word_count = len(msg.transcript.split())
        
        # Heuristics for detecting issues
        false_trigger = transcript_length < 3  # Very short transcripts might be false triggers
        missed_endpoint = word_count > 50  # Very long transcripts might indicate missed endpoints
        natural_pause_detected = "..." in msg.transcript or msg.transcript.endswith(",")
        
        if false_trigger:
            self.false_triggers += 1
            logger.warning(f"[VAD_TUNING] Possible false trigger: '{msg.transcript}'")
        
        if missed_endpoint:
            self.missed_endpoints += 1
            logger.warning(f"[VAD_TUNING] Possible missed endpoint: {word_count} words")
        
        if natural_pause_detected:
            self.natural_pauses_detected += 1
            logger.info(f"[VAD_TUNING] Natural pause detected in transcript")
        
        # Store timing for metrics
        self.current_speech_metrics = {
            "speech_detection_delay_ms": speech_detection_delay,
            "endpointing_delay_ms": endpointing_delay,
            "false_trigger": false_trigger,
            "missed_endpoint": missed_endpoint,
            "natural_pause_detected": natural_pause_detected,
            "transcript": msg.transcript
        }
        
        logger.info(f"[VAD_TUNING] Speech committed: '{msg.transcript}' "
                   f"(len={transcript_length}, words={word_count})")
        
        # Continue with normal processing
        await super().on_user_speech_committed(msg)
    
    def log_interaction_metrics(self, user_transcript: str, agent_response: str, 
                              end_to_end_latency: float):
        """Log VAD tuning metrics for a single interaction"""
        self.interaction_counter += 1
        
        # Get speech metrics from the last speech event
        speech_metrics = getattr(self, 'current_speech_metrics', {})
        
        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "vad_config": self.vad_config_name,
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "end_to_end_latency_ms": f"{end_to_end_latency:.2f}",
            "speech_detection_delay_ms": f"{speech_metrics.get('speech_detection_delay_ms', 0):.2f}",
            "endpointing_delay_ms": f"{speech_metrics.get('endpointing_delay_ms', 0):.2f}",
            "false_trigger": speech_metrics.get('false_trigger', False),
            "missed_endpoint": speech_metrics.get('missed_endpoint', False),
            "natural_pause_detected": speech_metrics.get('natural_pause_detected', False),
            "min_endpointing_delay": self.vad_config["min_endpointing_delay"],
            "max_endpointing_delay": self.vad_config["max_endpointing_delay"],
            "turn_detection_mode": self.vad_config["turn_detection"],
            "response_length_chars": len(agent_response),
            "response_length_words": len(agent_response.split()),
        }
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        logger.info(f"[VAD_TUNING] Logged interaction {self.interaction_counter}: "
                   f"{end_to_end_latency:.2f}ms, FT:{speech_metrics.get('false_trigger', False)}, "
                   f"ME:{speech_metrics.get('missed_endpoint', False)}")
    
    def get_session_summary(self) -> dict:
        """Get summary of VAD tuning session"""
        return {
            "vad_config": self.vad_config_name,
            "config_details": self.vad_config,
            "session_id": self.session_id,
            "total_interactions": self.interaction_counter,
            "false_triggers": self.false_triggers,
            "missed_endpoints": self.missed_endpoints,
            "natural_pauses_detected": self.natural_pauses_detected,
            "false_trigger_rate": self.false_triggers / max(self.interaction_counter, 1),
            "missed_endpoint_rate": self.missed_endpoints / max(self.interaction_counter, 1),
        }


async def entrypoint(ctx: JobContext):
    """
    VAD tuning agent entrypoint.
    Tests specific VAD/endpointing parameters and collects detailed metrics.
    """
    await ctx.connect()
    
    config = VAD_CONFIGURATIONS[CURRENT_VAD_CONFIG]
    
    logger.info(f"[VAD_TUNING] Starting VAD tuning with config: {config['name']}")
    logger.info(f"[VAD_TUNING] Configuration: {config['description']}")
    logger.info(f"[VAD_TUNING] Parameters:")
    logger.info(f"[VAD_TUNING] - Min endpointing delay: {config['min_endpointing_delay']}s")
    logger.info(f"[VAD_TUNING] - Max endpointing delay: {config['max_endpointing_delay']}s")
    logger.info(f"[VAD_TUNING] - Turn detection: {config['turn_detection']}")
    
    # Create VAD tuning agent
    agent = VADTuningAgent(CURRENT_VAD_CONFIG)
    
    # Create session with specific VAD configuration
    session = AgentSession(
        # Turn detection based on config
        turn_detection=config["turn_detection"],
        vad=silero.VAD.load(),
        
        # Standard STT
        stt=deepgram.STT(
            model="nova-2",
            language="en-US",
            interim_results=False,  # Keep simple for VAD testing
        ),
        
        # Standard TTS
        tts=cartesia.TTS(
            model="sonic-2",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
            speed=1.0,
        ),
        
        # Test-specific endpointing parameters
        min_endpointing_delay=config["min_endpointing_delay"],
        max_endpointing_delay=config["max_endpointing_delay"],
        allow_interruptions=True,
    )
    
    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    
    logger.info(f"[VAD_TUNING] VAD tuning agent started successfully with {config['name']} configuration")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
