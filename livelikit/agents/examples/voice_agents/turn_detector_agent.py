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
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("turn-detector-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Turn Detection Configurations for Comparison
TURN_DETECTOR_CONFIGS = {
    "basic_vad": {
        "name": "Basic VAD",
        "description": "Simple voice activity detection",
        "turn_detection": "vad",
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
    },
    
    "stt_based": {
        "name": "STT-Based",
        "description": "STT confidence-based turn detection",
        "turn_detection": "stt",
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
    },
    
    "multilingual_model": {
        "name": "Multilingual Model",
        "description": "Advanced model-based turn detection",
        "turn_detection": "multilingual_model",
        "min_endpointing_delay": 0.3,
        "max_endpointing_delay": 4.0,
    },
    
    "multilingual_sensitive": {
        "name": "Multilingual Sensitive",
        "description": "More sensitive multilingual model",
        "turn_detection": "multilingual_model",
        "min_endpointing_delay": 0.2,
        "max_endpointing_delay": 3.0,
    },
    
    "multilingual_patient": {
        "name": "Multilingual Patient",
        "description": "More patient multilingual model",
        "turn_detection": "multilingual_model",
        "min_endpointing_delay": 0.8,
        "max_endpointing_delay": 8.0,
    }
}

# Current configuration to test
CURRENT_TURN_CONFIG = "multilingual_model"  # Change to test different configurations

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class TurnDetectorAgent(Agent):
    """
    Voice agent for testing different turn detection methods.
    Compares VAD, STT-based, and MultilingualModel turn detection.
    """
    
    def __init__(self, turn_config_name: str) -> None:
        self.turn_config = TURN_DETECTOR_CONFIGS[turn_config_name]
        self.turn_config_name = turn_config_name
        
        logger.info(f"[TURN_DETECTOR] Initializing with config: {self.turn_config['name']}")
        logger.info(f"[TURN_DETECTOR] Description: {self.turn_config['description']}")
        logger.info(f"[TURN_DETECTOR] Turn detection: {self.turn_config['turn_detection']}")
        
        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[TURN_DETECTOR] Successfully loaded prompt from prompt.txt")
        except Exception as e:
            logger.error(f"[TURN_DETECTOR] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
        
        super().__init__(
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            instructions=instructions,
        )
        
        # Initialize metrics collection
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.metrics_file = os.path.join(os.path.dirname(__file__), 
                                       f"turn_detector_{turn_config_name}_{self.session_id}.csv")
        
        # Turn detection specific tracking
        self.interruptions_detected = 0
        self.natural_pauses_respected = 0
        self.premature_interruptions = 0
        self.conversation_flow_score = 0
        
        # Initialize CSV file
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "turn_config",
            "user_transcript", "agent_response", "end_to_end_latency_ms",
            "turn_detection_method", "interruption_detected", "natural_pause_respected",
            "premature_interruption", "conversation_flow_score", "user_speech_duration_ms",
            "min_endpointing_delay", "max_endpointing_delay",
            "response_length_chars", "response_length_words"
        ]
        
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        logger.info(f"[TURN_DETECTOR] Metrics file initialized: {self.metrics_file}")
    
    async def on_user_speech_committed(self, msg):
        """Track turn detection behavior and conversation flow"""
        current_time = time.perf_counter()
        
        # Analyze conversation flow characteristics
        transcript = msg.transcript.strip()
        
        # Detect natural conversation patterns
        has_natural_pause = any(phrase in transcript.lower() for phrase in [
            "um", "uh", "let me think", "i think", "well", "you know", "actually"
        ])
        
        ends_with_pause = transcript.endswith(("...", ",", "and", "but", "so"))
        
        # Heuristics for conversation flow quality
        interruption_detected = len(transcript) < 5  # Very short might indicate interruption
        natural_pause_respected = has_natural_pause and len(transcript) > 10
        premature_interruption = interruption_detected and not transcript.endswith(("?", ".", "!"))
        
        # Calculate conversation flow score (0-10)
        flow_score = 5  # Base score
        if natural_pause_respected:
            flow_score += 2
        if not premature_interruption:
            flow_score += 2
        if len(transcript.split()) > 3:  # Complete thoughts
            flow_score += 1
        if interruption_detected:
            flow_score -= 3
        
        flow_score = max(0, min(10, flow_score))  # Clamp to 0-10
        
        # Update counters
        if interruption_detected:
            self.interruptions_detected += 1
        if natural_pause_respected:
            self.natural_pauses_respected += 1
        if premature_interruption:
            self.premature_interruptions += 1
        
        self.conversation_flow_score += flow_score
        
        # Store metrics for this interaction
        self.current_turn_metrics = {
            "interruption_detected": interruption_detected,
            "natural_pause_respected": natural_pause_respected,
            "premature_interruption": premature_interruption,
            "conversation_flow_score": flow_score,
            "user_speech_duration_ms": 0,  # Would need to track actual speech duration
            "transcript": transcript
        }
        
        logger.info(f"[TURN_DETECTOR] Speech committed: '{transcript}' "
                   f"(flow_score={flow_score}, interruption={interruption_detected})")
        
        # Continue with normal processing
        await super().on_user_speech_committed(msg)
    
    def log_interaction_metrics(self, user_transcript: str, agent_response: str, 
                              end_to_end_latency: float):
        """Log turn detection metrics for a single interaction"""
        self.interaction_counter += 1
        
        # Get turn detection metrics from the last speech event
        turn_metrics = getattr(self, 'current_turn_metrics', {})
        
        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "turn_config": self.turn_config_name,
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "end_to_end_latency_ms": f"{end_to_end_latency:.2f}",
            "turn_detection_method": self.turn_config["turn_detection"],
            "interruption_detected": turn_metrics.get('interruption_detected', False),
            "natural_pause_respected": turn_metrics.get('natural_pause_respected', False),
            "premature_interruption": turn_metrics.get('premature_interruption', False),
            "conversation_flow_score": turn_metrics.get('conversation_flow_score', 5),
            "user_speech_duration_ms": turn_metrics.get('user_speech_duration_ms', 0),
            "min_endpointing_delay": self.turn_config["min_endpointing_delay"],
            "max_endpointing_delay": self.turn_config["max_endpointing_delay"],
            "response_length_chars": len(agent_response),
            "response_length_words": len(agent_response.split()),
        }
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        logger.info(f"[TURN_DETECTOR] Logged interaction {self.interaction_counter}: "
                   f"{end_to_end_latency:.2f}ms, flow_score={turn_metrics.get('conversation_flow_score', 5)}")
    
    def get_session_summary(self) -> dict:
        """Get summary of turn detection session"""
        avg_flow_score = self.conversation_flow_score / max(self.interaction_counter, 1)
        
        return {
            "turn_config": self.turn_config_name,
            "config_details": self.turn_config,
            "session_id": self.session_id,
            "total_interactions": self.interaction_counter,
            "interruptions_detected": self.interruptions_detected,
            "natural_pauses_respected": self.natural_pauses_respected,
            "premature_interruptions": self.premature_interruptions,
            "average_conversation_flow_score": avg_flow_score,
            "interruption_rate": self.interruptions_detected / max(self.interaction_counter, 1),
            "natural_pause_rate": self.natural_pauses_respected / max(self.interaction_counter, 1),
            "premature_interruption_rate": self.premature_interruptions / max(self.interaction_counter, 1),
        }


async def entrypoint(ctx: JobContext):
    """
    Turn detector agent entrypoint.
    Tests different turn detection methods and measures conversation flow quality.
    """
    await ctx.connect()
    
    config = TURN_DETECTOR_CONFIGS[CURRENT_TURN_CONFIG]
    
    logger.info(f"[TURN_DETECTOR] Starting turn detector test with: {config['name']}")
    logger.info(f"[TURN_DETECTOR] Configuration: {config['description']}")
    logger.info(f"[TURN_DETECTOR] Turn detection method: {config['turn_detection']}")
    
    # Create turn detector agent
    agent = TurnDetectorAgent(CURRENT_TURN_CONFIG)
    
    # Configure turn detection based on the test configuration
    session_config = {
        "vad": silero.VAD.load(),
        "stt": deepgram.STT(model="nova-2", language="en-US", interim_results=False),
        "tts": cartesia.TTS(model="sonic-2", voice="79a125e8-cd45-4c13-8a67-188112f4dd22", speed=1.0),
        "min_endpointing_delay": config["min_endpointing_delay"],
        "max_endpointing_delay": config["max_endpointing_delay"],
        "allow_interruptions": True,
    }
    
    # Set turn detection method
    if config["turn_detection"] == "multilingual_model":
        session_config["turn_detection"] = MultilingualModel()
        logger.info("[TURN_DETECTOR] Using MultilingualModel for turn detection")
    elif config["turn_detection"] == "stt":
        session_config["turn_detection"] = "stt"
        logger.info("[TURN_DETECTOR] Using STT-based turn detection")
    else:
        session_config["turn_detection"] = "vad"
        logger.info("[TURN_DETECTOR] Using VAD-based turn detection")
    
    # Create session
    session = AgentSession(**session_config)
    
    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    
    logger.info(f"[TURN_DETECTOR] Turn detector agent started with {config['name']} configuration")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
