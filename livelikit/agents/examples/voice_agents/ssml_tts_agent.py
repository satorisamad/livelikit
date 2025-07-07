import logging
import time
import os
import csv
import json
import re
import wave
from datetime import datetime
import asyncio

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.job import JobProcess
from livekit.agents.voice import Agent, AgentSession, events
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, openai, silero, deepgram

logger = logging.getLogger("ssml-tts-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# SSML Enhancement Configurations
SSML_CONFIGS = {
    "no_ssml": {
        "name": "No SSML",
        "description": "Basic TTS without SSML enhancements",
        "enable_ssml": False,
        "speaking_rate": "medium",
        "emphasis_level": "none",
    },
    
    "basic_ssml": {
        "name": "Basic SSML",
        "description": "Basic SSML with pauses and emphasis",
        "enable_ssml": True,
        "speaking_rate": "medium",
        "emphasis_level": "moderate",
    },
    
    "healthcare_ssml": {
        "name": "Healthcare SSML",
        "description": "Healthcare-optimized SSML with medical terms",
        "enable_ssml": True,
        "speaking_rate": "slow",
        "emphasis_level": "strong",
    },
    
    "expressive_ssml": {
        "name": "Expressive SSML",
        "description": "Highly expressive SSML with varied prosody",
        "enable_ssml": True,
        "speaking_rate": "medium",
        "emphasis_level": "strong",
    },
    
    "multilingual_ssml": {
        "name": "Multilingual SSML",
        "description": "SSML optimized for multilingual content",
        "enable_ssml": True,
        "speaking_rate": "slow",
        "emphasis_level": "moderate",
    }
}

# Current configuration to test
CURRENT_SSML_CONFIG = "healthcare_ssml"  # Change to test different configurations

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class SSMLTTSAgent(Agent):
    """
    Voice agent for testing SSML-enhanced TTS.
    Implements comprehensive SSML features and measures speech quality.
    """
    
    def __init__(self, ssml_config_name: str) -> None:
        self.ssml_config = SSML_CONFIGS[ssml_config_name]
        self.ssml_config_name = ssml_config_name
        
        logger.info(f"[SSML_TTS] Initializing with config: {self.ssml_config['name']}")
        logger.info(f"[SSML_TTS] Description: {self.ssml_config['description']}")
        logger.info(f"[SSML_TTS] SSML enabled: {self.ssml_config['enable_ssml']}")
        
        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[SSML_TTS] Successfully loaded prompt from prompt.txt")
        except Exception as e:
            logger.error(f"[SSML_TTS] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
        
        super().__init__(
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            instructions=instructions,
        )
        
        # Initialize metrics collection
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.metrics_file = os.path.join(os.path.dirname(__file__), 
                                       f"ssml_tts_{ssml_config_name}_{self.session_id}.csv")
        
        # SSML-specific tracking
        self.ssml_enhancements_used = []
        self.audio_quality_scores = []
        self.speech_naturalness_scores = []
        
        # Initialize CSV file
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "ssml_config",
            "user_transcript", "agent_response", "enhanced_ssml_text",
            "end_to_end_latency_ms", "tts_synthesis_ms", "audio_file_path",
            "ssml_enhancements_count", "emphasis_tags", "prosody_tags", "break_tags",
            "speaking_rate", "emphasis_level", "audio_duration_ms", "file_size_bytes",
            "estimated_mos_score", "naturalness_score", "clarity_score",
            "response_length_chars", "response_length_words"
        ]
        
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        logger.info(f"[SSML_TTS] Metrics file initialized: {self.metrics_file}")
    
    def enhance_text_with_ssml(self, text: str) -> str:
        """Apply comprehensive SSML enhancements to text"""
        if not self.ssml_config["enable_ssml"]:
            return text
        
        enhanced_text = text
        enhancements_used = []
        
        # Healthcare-specific enhancements
        if self.ssml_config_name == "healthcare_ssml":
            # Emphasize medical terms
            medical_terms = [
                "emergency", "urgent", "appointment", "doctor", "clinic", "insurance",
                "prescription", "medication", "symptoms", "diagnosis", "treatment",
                "hospital", "surgery", "therapy", "vaccine", "blood pressure"
            ]
            
            for term in medical_terms:
                pattern = rf'\b{re.escape(term)}\b'
                if re.search(pattern, enhanced_text, re.IGNORECASE):
                    enhanced_text = re.sub(pattern, 
                                         f'<emphasis level="{self.ssml_config["emphasis_level"]}">{term}</emphasis>', 
                                         enhanced_text, flags=re.IGNORECASE)
                    enhancements_used.append(f"emphasis:{term}")
            
            # Add pauses for clarity in medical context
            enhanced_text = enhanced_text.replace("!", '!<break time="0.5s"/>')
            enhanced_text = enhanced_text.replace("?", '?<break time="0.3s"/>')
            enhanced_text = enhanced_text.replace(",", ',<break time="0.2s"/>')
            enhancements_used.append("medical_pauses")
        
        # Expressive SSML enhancements
        elif self.ssml_config_name == "expressive_ssml":
            # Vary speaking rate for emphasis
            important_phrases = ["important", "please note", "remember", "urgent", "critical"]
            for phrase in important_phrases:
                if phrase in enhanced_text.lower():
                    pattern = rf'\b{re.escape(phrase)}\b'
                    enhanced_text = re.sub(pattern, 
                                         f'<prosody rate="slow" pitch="+2st">{phrase}</prosody>', 
                                         enhanced_text, flags=re.IGNORECASE)
                    enhancements_used.append(f"prosody:{phrase}")
            
            # Add emotional emphasis
            positive_words = ["great", "wonderful", "excellent", "perfect", "fantastic"]
            for word in positive_words:
                if word in enhanced_text.lower():
                    pattern = rf'\b{re.escape(word)}\b'
                    enhanced_text = re.sub(pattern, 
                                         f'<emphasis level="strong">{word}</emphasis>', 
                                         enhanced_text, flags=re.IGNORECASE)
                    enhancements_used.append(f"positive_emphasis:{word}")
        
        # Multilingual SSML enhancements
        elif self.ssml_config_name == "multilingual_ssml":
            # Handle common acronyms
            acronyms = ["AI", "API", "URL", "FAQ", "CEO", "USA", "UK", "EU"]
            for acronym in acronyms:
                if acronym in enhanced_text:
                    enhanced_text = enhanced_text.replace(acronym, 
                                                        f'<say-as interpret-as="spell-out">{acronym}</say-as>')
                    enhancements_used.append(f"spell_out:{acronym}")
            
            # Add language-specific prosody
            enhanced_text = f'<prosody rate="{self.ssml_config["speaking_rate"]}" pitch="medium">{enhanced_text}</prosody>'
            enhancements_used.append("multilingual_prosody")
        
        # Basic SSML enhancements
        else:
            # Add basic pauses
            enhanced_text = enhanced_text.replace(".", '.<break time="0.3s"/>')
            enhanced_text = enhanced_text.replace("!", '!<break time="0.4s"/>')
            enhanced_text = enhanced_text.replace("?", '?<break time="0.3s"/>')
            enhancements_used.append("basic_pauses")
            
            # Basic emphasis on key words
            key_words = ["yes", "no", "please", "thank you", "sorry", "help"]
            for word in key_words:
                if word in enhanced_text.lower():
                    pattern = rf'\b{re.escape(word)}\b'
                    enhanced_text = re.sub(pattern, 
                                         f'<emphasis level="{self.ssml_config["emphasis_level"]}">{word}</emphasis>', 
                                         enhanced_text, flags=re.IGNORECASE)
                    enhancements_used.append(f"basic_emphasis:{word}")
        
        # Wrap in SSML speak tag
        enhanced_text = f'<speak>{enhanced_text}</speak>'
        
        # Store enhancements for metrics
        self.ssml_enhancements_used = enhancements_used
        
        logger.info(f"[SSML_TTS] Enhanced text with {len(enhancements_used)} SSML features")
        logger.debug(f"[SSML_TTS] Enhancements: {enhancements_used}")
        
        return enhanced_text
    
    async def tts_node(self, text, model_settings):
        """Enhanced TTS node with SSML processing and quality metrics"""
        tts_start_time = time.perf_counter()
        
        # Collect all text from the async iterator
        text_segments = []
        async for segment in text:
            text_segments.append(segment)
        
        full_text = "".join(text_segments)
        logger.info(f"[SSML_TTS] Processing text: '{full_text}'")
        
        # Apply SSML enhancements
        enhanced_text = self.enhance_text_with_ssml(full_text)
        
        # Prepare audio output file
        output_dir = "ssml_audio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audio_file = os.path.join(output_dir, f"ssml_{self.ssml_config_name}_{timestamp}.wav")
        
        # Generate TTS with SSML
        synthesis_start_time = time.perf_counter()
        
        # Create text generator for TTS
        async def ssml_text_generator():
            yield enhanced_text if self.ssml_config["enable_ssml"] else full_text
        
        # Process TTS
        audio_frames = []
        async for audio_frame in super().tts_node(ssml_text_generator(), model_settings):
            audio_frames.append(audio_frame)
            yield audio_frame
        
        synthesis_end_time = time.perf_counter()
        synthesis_duration_ms = (synthesis_end_time - synthesis_start_time) * 1000
        
        # Save audio file and calculate metrics
        audio_duration_ms = 0
        file_size_bytes = 0
        
        if audio_frames:
            combined_frame = rtc.combine_audio_frames(audio_frames)
            sample_rate = combined_frame.sample_rate
            num_channels = combined_frame.num_channels
            
            with wave.open(audio_file, 'wb') as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sample_rate)
                wf.writeframes(combined_frame.data.tobytes())
            
            audio_duration_ms = (len(combined_frame.data) / (sample_rate * num_channels)) * 1000
            file_size_bytes = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        
        # Calculate quality scores (simplified estimates)
        mos_score = self.estimate_mos_score(enhanced_text, audio_duration_ms)
        naturalness_score = self.estimate_naturalness_score(enhanced_text)
        clarity_score = self.estimate_clarity_score(enhanced_text)
        
        # Store metrics for this TTS generation
        self.current_ssml_metrics = {
            "enhanced_ssml_text": enhanced_text,
            "tts_synthesis_ms": synthesis_duration_ms,
            "audio_file_path": audio_file,
            "audio_duration_ms": audio_duration_ms,
            "file_size_bytes": file_size_bytes,
            "estimated_mos_score": mos_score,
            "naturalness_score": naturalness_score,
            "clarity_score": clarity_score,
            "original_text": full_text
        }
        
        logger.info(f"[SSML_TTS] TTS completed: {synthesis_duration_ms:.2f}ms, "
                   f"audio: {audio_duration_ms:.2f}ms, MOS: {mos_score:.1f}")
    
    def estimate_mos_score(self, enhanced_text: str, audio_duration_ms: float) -> float:
        """Estimate Mean Opinion Score based on SSML features and timing"""
        base_score = 3.0  # Neutral baseline
        
        # Bonus for SSML enhancements
        if self.ssml_config["enable_ssml"]:
            base_score += 0.5
            
            # Bonus for specific enhancements
            if "<emphasis" in enhanced_text:
                base_score += 0.3
            if "<prosody" in enhanced_text:
                base_score += 0.3
            if "<break" in enhanced_text:
                base_score += 0.2
            if "<say-as" in enhanced_text:
                base_score += 0.2
        
        # Penalty for very long or very short audio
        if audio_duration_ms > 0:
            if audio_duration_ms < 500:  # Too short
                base_score -= 0.5
            elif audio_duration_ms > 10000:  # Too long
                base_score -= 0.3
        
        return max(1.0, min(5.0, base_score))
    
    def estimate_naturalness_score(self, enhanced_text: str) -> float:
        """Estimate naturalness score based on SSML features"""
        base_score = 5.0
        
        # Bonus for natural pauses
        break_count = enhanced_text.count("<break")
        base_score += min(break_count * 0.2, 1.0)
        
        # Bonus for varied prosody
        if "<prosody" in enhanced_text:
            base_score += 0.5
        
        # Penalty for too many emphasis tags (unnatural)
        emphasis_count = enhanced_text.count("<emphasis")
        if emphasis_count > 5:
            base_score -= 0.3
        
        return max(1.0, min(10.0, base_score))
    
    def estimate_clarity_score(self, enhanced_text: str) -> float:
        """Estimate clarity score based on SSML features"""
        base_score = 5.0
        
        # Bonus for spell-out of acronyms
        if "<say-as" in enhanced_text:
            base_score += 0.5
        
        # Bonus for appropriate pauses
        if "<break" in enhanced_text:
            base_score += 0.3
        
        # Bonus for emphasis on important terms
        if "<emphasis" in enhanced_text:
            base_score += 0.2
        
        return max(1.0, min(10.0, base_score))
    
    def log_interaction_metrics(self, user_transcript: str, agent_response: str, 
                              end_to_end_latency: float):
        """Log SSML TTS metrics for a single interaction"""
        self.interaction_counter += 1
        
        # Get SSML metrics from the last TTS generation
        ssml_metrics = getattr(self, 'current_ssml_metrics', {})
        
        # Count SSML tag types
        enhanced_text = ssml_metrics.get('enhanced_ssml_text', '')
        emphasis_count = enhanced_text.count('<emphasis')
        prosody_count = enhanced_text.count('<prosody')
        break_count = enhanced_text.count('<break')
        
        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "ssml_config": self.ssml_config_name,
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "enhanced_ssml_text": enhanced_text,
            "end_to_end_latency_ms": f"{end_to_end_latency:.2f}",
            "tts_synthesis_ms": f"{ssml_metrics.get('tts_synthesis_ms', 0):.2f}",
            "audio_file_path": ssml_metrics.get('audio_file_path', ''),
            "ssml_enhancements_count": len(self.ssml_enhancements_used),
            "emphasis_tags": emphasis_count,
            "prosody_tags": prosody_count,
            "break_tags": break_count,
            "speaking_rate": self.ssml_config["speaking_rate"],
            "emphasis_level": self.ssml_config["emphasis_level"],
            "audio_duration_ms": f"{ssml_metrics.get('audio_duration_ms', 0):.2f}",
            "file_size_bytes": ssml_metrics.get('file_size_bytes', 0),
            "estimated_mos_score": f"{ssml_metrics.get('estimated_mos_score', 3.0):.1f}",
            "naturalness_score": f"{ssml_metrics.get('naturalness_score', 5.0):.1f}",
            "clarity_score": f"{ssml_metrics.get('clarity_score', 5.0):.1f}",
            "response_length_chars": len(agent_response),
            "response_length_words": len(agent_response.split()),
        }
        
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        logger.info(f"[SSML_TTS] Logged interaction {self.interaction_counter}: "
                   f"MOS: {ssml_metrics.get('estimated_mos_score', 3.0):.1f}, "
                   f"enhancements: {len(self.ssml_enhancements_used)}")


async def entrypoint(ctx: JobContext):
    """
    SSML TTS agent entrypoint.
    Tests SSML-enhanced TTS and measures speech quality improvements.
    """
    await ctx.connect()
    
    config = SSML_CONFIGS[CURRENT_SSML_CONFIG]
    
    logger.info(f"[SSML_TTS] Starting SSML TTS test: {config['name']}")
    logger.info(f"[SSML_TTS] Configuration: {config['description']}")
    logger.info(f"[SSML_TTS] Features:")
    logger.info(f"[SSML_TTS] - SSML enabled: {config['enable_ssml']}")
    logger.info(f"[SSML_TTS] - Speaking rate: {config['speaking_rate']}")
    logger.info(f"[SSML_TTS] - Emphasis level: {config['emphasis_level']}")
    
    # Create SSML TTS agent
    agent = SSMLTTSAgent(CURRENT_SSML_CONFIG)
    
    # Create session with standard configuration
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2", language="en-US", interim_results=False),
        tts=cartesia.TTS(model="sonic-2", voice="79a125e8-cd45-4c13-8a67-188112f4dd22", speed=1.0),
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
    
    logger.info(f"[SSML_TTS] SSML TTS agent started with {config['name']} configuration")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
