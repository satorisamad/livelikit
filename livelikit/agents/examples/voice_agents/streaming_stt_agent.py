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

logger = logging.getLogger("streaming-stt-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Streaming STT Configurations for Testing
STREAMING_CONFIGS = {
    "no_streaming": {
        "name": "No Streaming",
        "description": "Traditional non-streaming STT",
        "interim_results": False,
        "enable_partial_llm": False,
        "confidence_threshold": 0.0,
    },
    
    "basic_streaming": {
        "name": "Basic Streaming",
        "description": "Streaming STT without partial processing",
        "interim_results": True,
        "enable_partial_llm": False,
        "confidence_threshold": 0.0,
    },
    
    "partial_processing": {
        "name": "Partial Processing",
        "description": "Streaming STT with partial LLM prompting",
        "interim_results": True,
        "enable_partial_llm": True,
        "confidence_threshold": 0.7,
    },
    
    "aggressive_partial": {
        "name": "Aggressive Partial",
        "description": "Early partial processing at 60% confidence",
        "interim_results": True,
        "enable_partial_llm": True,
        "confidence_threshold": 0.6,
    },
    
    "conservative_partial": {
        "name": "Conservative Partial",
        "description": "Conservative partial processing at 80% confidence",
        "interim_results": True,
        "enable_partial_llm": True,
        "confidence_threshold": 0.8,
    }
}

# Current configuration to test
CURRENT_STREAMING_CONFIG = "partial_processing"  # Change to test different configurations

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class StreamingSTTAgent(Agent):
    """
    Voice agent for testing streaming STT and partial hypotheses processing.
    Measures impact on latency and transcript accuracy.
    """
    
    def __init__(self, streaming_config_name: str) -> None:
        self.streaming_config = STREAMING_CONFIGS[streaming_config_name]
        self.streaming_config_name = streaming_config_name
        
        logger.info(f"[STREAMING_STT] Initializing with config: {self.streaming_config['name']}")
        logger.info(f"[STREAMING_STT] Description: {self.streaming_config['description']}")
        logger.info(f"[STREAMING_STT] Interim results: {self.streaming_config['interim_results']}")
        logger.info(f"[STREAMING_STT] Partial LLM: {self.streaming_config['enable_partial_llm']}")
        
        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[STREAMING_STT] Successfully loaded prompt from prompt.txt")
        except Exception as e:
            logger.error(f"[STREAMING_STT] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
        
        super().__init__(
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            instructions=instructions,
        )
        
        # Initialize metrics collection
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.metrics_file = os.path.join(os.path.dirname(__file__), 
                                       f"streaming_stt_{streaming_config_name}_{self.session_id}.csv")
        
        # Streaming STT specific tracking
        self.partial_transcripts = []
        self.partial_llm_triggers = 0
        self.early_processing_gains = []
        self.transcript_accuracy_scores = []
        self.first_partial_time = None
        self.final_transcript_time = None
        
        # Initialize CSV file
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "streaming_config",
            "user_transcript", "agent_response", "end_to_end_latency_ms",
            "partial_transcripts_count", "partial_llm_triggered", "early_processing_gain_ms",
            "first_partial_to_final_ms", "transcript_accuracy_score", "confidence_threshold",
            "interim_results_enabled", "enable_partial_llm",
            "response_length_chars", "response_length_words"
        ]
        
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        logger.info(f"[STREAMING_STT] Metrics file initialized: {self.metrics_file}")
    
    async def on_user_speech_interim(self, msg):
        """Handle interim/partial transcript results"""
        if not self.streaming_config["interim_results"]:
            return
        
        current_time = time.perf_counter()
        
        # Track first partial transcript timing
        if self.first_partial_time is None:
            self.first_partial_time = current_time
        
        # Store partial transcript
        partial_data = {
            "transcript": msg.transcript,
            "confidence": getattr(msg, 'confidence', 0.0),
            "timestamp": current_time,
            "length": len(msg.transcript)
        }
        self.partial_transcripts.append(partial_data)
        
        logger.info(f"[STREAMING_STT] Partial transcript #{len(self.partial_transcripts)}: "
                   f"'{msg.transcript}' (confidence: {partial_data['confidence']:.2f})")
        
        # Check if we should trigger partial LLM processing
        if (self.streaming_config["enable_partial_llm"] and 
            partial_data['confidence'] >= self.streaming_config["confidence_threshold"] and
            len(msg.transcript) > 10):  # Minimum length for meaningful processing
            
            logger.info(f"[STREAMING_STT] Triggering partial LLM processing "
                       f"(confidence: {partial_data['confidence']:.2f} >= {self.streaming_config['confidence_threshold']:.2f})")
            
            self.partial_llm_triggers += 1
            
            # In a real implementation, you would start LLM processing here
            # For now, we just track that it would have been triggered
            
    async def on_user_speech_committed(self, msg):
        """Handle final transcript and calculate streaming metrics"""
        current_time = time.perf_counter()
        self.final_transcript_time = current_time
        
        # Calculate streaming-specific metrics
        partial_count = len(self.partial_transcripts)
        partial_llm_triggered = self.partial_llm_triggers > 0
        
        # Calculate early processing gain
        early_processing_gain = 0
        if self.first_partial_time and partial_llm_triggered:
            early_processing_gain = (current_time - self.first_partial_time) * 1000
            self.early_processing_gains.append(early_processing_gain)
        
        # Calculate first partial to final time
        first_partial_to_final = 0
        if self.first_partial_time:
            first_partial_to_final = (current_time - self.first_partial_time) * 1000
        
        # Calculate transcript accuracy score (simplified)
        accuracy_score = self.calculate_transcript_accuracy(msg.transcript)
        self.transcript_accuracy_scores.append(accuracy_score)
        
        # Store metrics for this interaction
        self.current_streaming_metrics = {
            "partial_transcripts_count": partial_count,
            "partial_llm_triggered": partial_llm_triggered,
            "early_processing_gain_ms": early_processing_gain,
            "first_partial_to_final_ms": first_partial_to_final,
            "transcript_accuracy_score": accuracy_score,
            "transcript": msg.transcript
        }
        
        logger.info(f"[STREAMING_STT] Final transcript: '{msg.transcript}' "
                   f"(partials: {partial_count}, early_gain: {early_processing_gain:.2f}ms)")
        
        # Reset for next interaction
        self.partial_transcripts = []
        self.partial_llm_triggers = 0
        self.first_partial_time = None
        
        # Continue with normal processing
        await super().on_user_speech_committed(msg)
    
    def calculate_transcript_accuracy(self, final_transcript: str) -> float:
        """Calculate a simple accuracy score based on transcript characteristics"""
        # This is a simplified accuracy measure
        # In practice, you'd compare against ground truth transcripts
        
        score = 5.0  # Base score
        
        # Penalize very short transcripts (might be incomplete)
        if len(final_transcript) < 5:
            score -= 2.0
        
        # Reward complete sentences
        if final_transcript.endswith(('.', '!', '?')):
            score += 1.0
        
        # Penalize obvious transcription errors
        error_indicators = ['[inaudible]', '[unclear]', '***', '...']
        for indicator in error_indicators:
            if indicator in final_transcript.lower():
                score -= 1.0
        
        # Reward proper capitalization and punctuation
        if final_transcript[0].isupper() if final_transcript else False:
            score += 0.5
        
        return max(0.0, min(10.0, score))  # Clamp to 0-10
    
    def log_interaction_metrics(self, user_transcript: str, agent_response: str, 
                              end_to_end_latency: float):
        """Log streaming STT metrics for a single interaction"""
        self.interaction_counter += 1
        
        # Get streaming metrics from the last speech event
        streaming_metrics = getattr(self, 'current_streaming_metrics', {})
        
        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "streaming_config": self.streaming_config_name,
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "end_to_end_latency_ms": f"{end_to_end_latency:.2f}",
            "partial_transcripts_count": streaming_metrics.get('partial_transcripts_count', 0),
            "partial_llm_triggered": streaming_metrics.get('partial_llm_triggered', False),
            "early_processing_gain_ms": f"{streaming_metrics.get('early_processing_gain_ms', 0):.2f}",
            "first_partial_to_final_ms": f"{streaming_metrics.get('first_partial_to_final_ms', 0):.2f}",
            "transcript_accuracy_score": f"{streaming_metrics.get('transcript_accuracy_score', 5.0):.1f}",
            "confidence_threshold": self.streaming_config["confidence_threshold"],
            "interim_results_enabled": self.streaming_config["interim_results"],
            "enable_partial_llm": self.streaming_config["enable_partial_llm"],
            "response_length_chars": len(agent_response),
            "response_length_words": len(agent_response.split()),
        }
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        logger.info(f"[STREAMING_STT] Logged interaction {self.interaction_counter}: "
                   f"{end_to_end_latency:.2f}ms, gain: {streaming_metrics.get('early_processing_gain_ms', 0):.2f}ms")
    
    def get_session_summary(self) -> dict:
        """Get summary of streaming STT session"""
        avg_accuracy = sum(self.transcript_accuracy_scores) / max(len(self.transcript_accuracy_scores), 1)
        avg_early_gain = sum(self.early_processing_gains) / max(len(self.early_processing_gains), 1)
        
        return {
            "streaming_config": self.streaming_config_name,
            "config_details": self.streaming_config,
            "session_id": self.session_id,
            "total_interactions": self.interaction_counter,
            "average_transcript_accuracy": avg_accuracy,
            "average_early_processing_gain_ms": avg_early_gain,
            "total_partial_llm_triggers": sum(self.early_processing_gains) > 0,
        }


async def entrypoint(ctx: JobContext):
    """
    Streaming STT agent entrypoint.
    Tests streaming STT with partial hypotheses and measures performance impact.
    """
    await ctx.connect()
    
    config = STREAMING_CONFIGS[CURRENT_STREAMING_CONFIG]
    
    logger.info(f"[STREAMING_STT] Starting streaming STT test: {config['name']}")
    logger.info(f"[STREAMING_STT] Configuration: {config['description']}")
    logger.info(f"[STREAMING_STT] Features:")
    logger.info(f"[STREAMING_STT] - Interim results: {config['interim_results']}")
    logger.info(f"[STREAMING_STT] - Partial LLM: {config['enable_partial_llm']}")
    logger.info(f"[STREAMING_STT] - Confidence threshold: {config['confidence_threshold']}")
    
    # Create streaming STT agent
    agent = StreamingSTTAgent(CURRENT_STREAMING_CONFIG)
    
    # Create session with streaming configuration
    session = AgentSession(
        vad=silero.VAD.load(),
        
        # Configure STT with streaming settings
        stt=deepgram.STT(
            model="nova-2",
            language="en-US",
            interim_results=config["interim_results"],
        ),
        
        tts=cartesia.TTS(
            model="sonic-2",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
            speed=1.0,
        ),
        
        # Standard endpointing for fair comparison
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0,
        allow_interruptions=True,
    )
    
    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    
    logger.info(f"[STREAMING_STT] Streaming STT agent started with {config['name']} configuration")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
