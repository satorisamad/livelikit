import logging
logging.basicConfig(level=logging.INFO)
import time
from datetime import datetime
import asyncio
import os
import wave
import csv
import json
import statistics
from typing import Dict, List, Optional
import difflib
import re

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.job import JobProcess
from livekit.protocol.agent import JobType
from livekit.agents.llm import ChatContext
from livekit.agents.voice import Agent, AgentSession, agent, events
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, openai, silero, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.cartesia import TTS as CartesiaTTS
from livekit.agents.stt import StreamAdapter
from livekit.agents.llm import LLMStream

# Note: Now using prompt.txt file directly instead of agent_prompts.py

logger = logging.getLogger("livekit.agents")
logger.setLevel(logging.INFO)

load_dotenv()

# Voice Agent Optimization Configuration
# This configuration supports multiple optimization modes for benchmarking
OPTIMIZATION_MODES = {
    "baseline": {
        # Basic configuration without optimizations
        "name": "Baseline (No Optimizations)",
        "description": "Basic VAD + non-streaming STT + simple LLM + basic TTS",

        # LLM Configuration
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.7,
        "llm_max_tokens": 150,

        # STT Configuration (Basic Deepgram)
        "stt_model": "nova-2",
        "stt_language": "en-US",
        "stt_interim_results": False,  # No streaming for baseline
        "stt_smart_format": True,
        "stt_punctuate": True,

        # TTS Configuration (Basic Cartesia)
        "tts_model": "sonic-2",
        "tts_sample_rate": 24000,
        "tts_encoding": "pcm_s16le",
        "tts_voice": "79a125e8-cd45-4c13-8a67-188112f4dd22",
        "tts_speed": 1.0,  # Normal speed

        # Turn Detection (Basic VAD)
        "turn_detection_mode": "vad",  # vad, stt, or multilingual_model
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
        "allow_interruptions": True,

        # Optimization Flags
        "enable_streaming_stt": False,
        "enable_partial_llm": False,
        "enable_ssml": False,
        "enable_turn_detector_plugin": False,

        # Timing (No artificial delays)
        "natural_response_delay": 0.0,
        "thinking_pause": 0.0,
        "tts_processing_delay": 0.0,
        "llm_processing_delay": 0.0,
        "final_pause": 0.0,
    },

    "optimized": {
        # Fully optimized configuration
        "name": "Optimized (All Features)",
        "description": "Streaming STT + partial LLM + turn detector + SSML TTS",

        # LLM Configuration
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.7,
        "llm_max_tokens": 150,

        # STT Configuration (Optimized Deepgram)
        "stt_model": "nova-2",
        "stt_language": "en-US",
        "stt_interim_results": True,  # Enable streaming
        "stt_smart_format": True,
        "stt_punctuate": True,

        # TTS Configuration (Enhanced Cartesia)
        "tts_model": "sonic-2",
        "tts_sample_rate": 24000,
        "tts_encoding": "pcm_s16le",
        "tts_voice": "79a125e8-cd45-4c13-8a67-188112f4dd22",
        "tts_speed": 1.0,

        # Turn Detection (Advanced)
        "turn_detection_mode": "multilingual_model",
        "min_endpointing_delay": 0.2,  # Faster response
        "max_endpointing_delay": 3.0,  # Shorter max wait
        "allow_interruptions": True,

        # Optimization Flags
        "enable_streaming_stt": True,
        "enable_partial_llm": True,
        "enable_ssml": True,
        "enable_turn_detector_plugin": True,

        # Timing (Minimal delays for speed)
        "natural_response_delay": 0.0,
        "thinking_pause": 0.0,
        "tts_processing_delay": 0.0,
        "llm_processing_delay": 0.0,
        "final_pause": 0.0,
    },

    "human_like": {
        # Current ultra-slow, human-like configuration
        "name": "Human-Like (Slow & Natural)",
        "description": "Ultra-slow timing for natural conversation feel",

        # LLM Configuration
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.7,
        "llm_max_tokens": 150,

        # STT Configuration
        "stt_model": "nova-2",
        "stt_language": "en-US",
        "stt_interim_results": False,  # Disabled for natural timing
        "stt_smart_format": True,
        "stt_punctuate": True,

        # TTS Configuration (Slow Speaking)
        "tts_model": "sonic-2",
        "tts_sample_rate": 24000,
        "tts_encoding": "pcm_s16le",
        "tts_voice": "79a125e8-cd45-4c13-8a67-188112f4dd22",
        "tts_speed": 0.6,  # Slow speech

        # Turn Detection (Very patient)
        "turn_detection_mode": "multilingual_model",
        "min_endpointing_delay": 2.0,  # Very patient
        "max_endpointing_delay": 6.0,
        "allow_interruptions": True,

        # Optimization Flags (Mostly disabled for natural feel)
        "enable_streaming_stt": False,
        "enable_partial_llm": False,
        "enable_ssml": False,
        "enable_turn_detector_plugin": True,

        # Timing (Ultra-slow for human-like feel)
        "natural_response_delay": 6.0,
        "thinking_pause": 3.0,
        "tts_processing_delay": 4.0,
        "llm_processing_delay": 3.0,
        "final_pause": 2.0,
        "contemplation_pause": 1.5,
    }
}

# Current active configuration - can be changed for testing
CURRENT_MODE = "baseline"  # Change to "optimized" or "human_like" for testing
AGENT_CONFIG = OPTIMIZATION_MODES[CURRENT_MODE]

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class VoiceOptimizationAgent(Agent):
    """
    Voice Agent with configurable optimization modes for benchmarking.
    Supports baseline, optimized, and human-like configurations.
    """

    def __init__(self, config_mode: str = "baseline") -> None:
        # Set the configuration for this agent instance
        self.config = OPTIMIZATION_MODES[config_mode]
        self.config_mode = config_mode

        logger.info(f"[CONFIG] VoiceOptimizationAgent.__init__ called with mode: {config_mode}")
        logger.info(f"[CONFIG] Mode description: {self.config['description']}")

        # Load instructions from prompt.txt file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                instructions = f.read().strip()
            logger.info(f"[CONFIG] Successfully loaded prompt from prompt.txt")
            logger.info(f"[CONFIG] Prompt length: {len(instructions)} characters")
        except FileNotFoundError:
            logger.error(f"[CONFIG] prompt.txt file not found")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
            logger.info(f"[CONFIG] Using fallback instructions")
        except Exception as e:
            logger.error(f"[CONFIG] Error loading prompt.txt: {e}")
            instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
            logger.info(f"[CONFIG] Using fallback instructions")

        super().__init__(
            # LLM configuration
            llm=openai.LLM(
                model=self.config["llm_model"],
                temperature=self.config["llm_temperature"],
            ),
            instructions=instructions,
            # TTS configuration (will be overridden by session)
            tts=cartesia.TTS(
                model=self.config["tts_model"],
                voice=self.config["tts_voice"],
                encoding=self.config["tts_encoding"],
                sample_rate=self.config["tts_sample_rate"],
                speed=self.config["tts_speed"],
            )
        )

        # Initialize comprehensive metrics logging
        self.metrics_file_path = os.path.join(os.path.dirname(__file__), f"voice_agent_metrics_{config_mode}.csv")
        self.csv_fieldnames = [
            "session_id", "timestamp", "interaction_id", "config_mode",
            # Transcript metrics
            "user_transcript", "transcript_confidence", "transcript_duration_ms",
            # Agent response metrics
            "agent_response", "response_length_chars", "response_length_words",
            # Latency metrics
            "mic_to_transcript_ms", "transcript_to_llm_ms", "llm_to_tts_start_ms",
            "tts_synthesis_ms", "end_to_end_latency_ms",
            # TTS quality metrics
            "tts_wav_path", "tts_sample_rate", "tts_duration_ms", "tts_file_size_bytes",
            # WER and quality metrics (to be filled manually or by evaluation scripts)
            "reference_transcript", "wer_score", "mos_score", "clarity_score", "expressiveness_score",
            # Additional context
            "room_name", "user_id", "error_occurred", "error_message"
        ]

        # Initialize session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_counter = 0
        self.reference_transcripts = {}  # For WER calculation

        if not os.path.exists(self.metrics_file_path):
            with open(self.metrics_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()

        # Initialize fallback logger
        self.logger = logger
        self._initialize_session_logging()

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate between reference and hypothesis transcripts"""
        if not reference or not hypothesis:
            return 0.0

        # Normalize text (lowercase, remove punctuation, split into words)
        ref_words = re.sub(r'[^\w\s]', '', reference.lower()).split()
        hyp_words = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()

        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0

        # Use difflib to calculate edit distance
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        operations = matcher.get_opcodes()

        substitutions = deletions = insertions = 0
        for op, i1, i2, j1, j2 in operations:
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1

        wer = (substitutions + deletions + insertions) / len(ref_words)
        return min(wer, 1.0)  # Cap at 1.0

    def _write_comprehensive_metrics(self, metrics_data: dict):
        """Write comprehensive metrics to CSV with all timing and quality data"""
        self.interaction_counter += 1

        # Calculate WER if reference transcript is available
        wer_score = None
        if metrics_data.get("reference_transcript") and metrics_data.get("user_transcript"):
            wer_score = self.calculate_wer(
                metrics_data["reference_transcript"],
                metrics_data["user_transcript"]
            )

        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"{self.session_id}_{self.interaction_counter:04d}",
            "config_mode": self.config_mode,

            # Transcript metrics
            "user_transcript": metrics_data.get("user_transcript", ""),
            "transcript_confidence": metrics_data.get("transcript_confidence", ""),
            "transcript_duration_ms": metrics_data.get("transcript_duration_ms", ""),

            # Agent response metrics
            "agent_response": metrics_data.get("agent_response", ""),
            "response_length_chars": len(metrics_data.get("agent_response", "")),
            "response_length_words": len(metrics_data.get("agent_response", "").split()),

            # Latency metrics
            "mic_to_transcript_ms": f"{metrics_data.get('mic_to_transcript_ms', ''):.2f}" if metrics_data.get('mic_to_transcript_ms') else "",
            "transcript_to_llm_ms": f"{metrics_data.get('transcript_to_llm_ms', ''):.2f}" if metrics_data.get('transcript_to_llm_ms') else "",
            "llm_to_tts_start_ms": f"{metrics_data.get('llm_to_tts_start_ms', ''):.2f}" if metrics_data.get('llm_to_tts_start_ms') else "",
            "tts_synthesis_ms": f"{metrics_data.get('tts_synthesis_ms', ''):.2f}" if metrics_data.get('tts_synthesis_ms') else "",
            "end_to_end_latency_ms": f"{metrics_data.get('end_to_end_latency_ms', ''):.2f}" if metrics_data.get('end_to_end_latency_ms') else "",

            # TTS quality metrics
            "tts_wav_path": metrics_data.get("tts_wav_path", ""),
            "tts_sample_rate": metrics_data.get("tts_sample_rate", ""),
            "tts_duration_ms": f"{metrics_data.get('tts_duration_ms', ''):.2f}" if metrics_data.get('tts_duration_ms') else "",
            "tts_file_size_bytes": metrics_data.get("tts_file_size_bytes", ""),

            # WER and quality metrics
            "reference_transcript": metrics_data.get("reference_transcript", ""),
            "wer_score": f"{wer_score:.4f}" if wer_score is not None else "",
            "mos_score": metrics_data.get("mos_score", ""),  # To be filled manually
            "clarity_score": metrics_data.get("clarity_score", ""),  # To be filled manually
            "expressiveness_score": metrics_data.get("expressiveness_score", ""),  # To be filled manually

            # Additional context
            "room_name": getattr(self, "room_name", ""),
            "user_id": metrics_data.get("user_id", ""),
            "error_occurred": metrics_data.get("error_occurred", False),
            "error_message": metrics_data.get("error_message", "")
        }

        with open(self.metrics_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)

        self.logger.info(f"[METRICS] Comprehensive metrics logged for interaction {self.interaction_counter}")
        self.logger.info(f"[METRICS] End-to-end latency: {metrics_data.get('end_to_end_latency_ms', 'N/A')} ms")
        if wer_score is not None:
            self.logger.info(f"[METRICS] WER Score: {wer_score:.4f}")

        # Legacy method for backward compatibility
        def _write_metrics_row(self, latency_ms, filename, text):
            """Legacy method - use _write_comprehensive_metrics instead"""
            metrics_data = {
                "agent_response": text,
                "end_to_end_latency_ms": latency_ms,
                "tts_wav_path": filename,
                "user_transcript": getattr(self, "current_transcript", "")
            }
            self._write_comprehensive_metrics(metrics_data)

        def _enhance_text_with_ssml(self, text: str) -> str:
            """Enhance text with SSML for better TTS quality"""
            if not AGENT_CONFIG["enable_ssml"]:
                return text

            # Healthcare-specific SSML enhancements
            enhanced_text = text

            # Add emphasis for important medical terms
            medical_terms = ["emergency", "urgent", "appointment", "doctor", "clinic", "insurance"]
            for term in medical_terms:
                enhanced_text = enhanced_text.replace(
                    term, f'<emphasis level="moderate">{term}</emphasis>'
                )

            # Add pauses for better clarity
            enhanced_text = enhanced_text.replace("!", '!<break time="0.3s"/>')
            enhanced_text = enhanced_text.replace("?", '?<break time="0.2s"/>')
            enhanced_text = enhanced_text.replace(",", ',<break time="0.1s"/>')

            # Adjust speaking rate for healthcare context
            enhanced_text = f'<prosody rate="{AGENT_CONFIG["tts_speed"]}">{enhanced_text}</prosody>'

            # Wrap in SSML speak tag
            enhanced_text = f'<speak>{enhanced_text}</speak>'

            self.logger.info(f"[SSML] Enhanced text length: {len(enhanced_text)} chars")
            return enhanced_text

        async def tts_node(self, text, model_settings):
            """Ultra slow TTS node for very human-like timing"""
            # Add ultra long delay before TTS processing
            if AGENT_CONFIG.get("tts_processing_delay", 0) > 0:
                delay = AGENT_CONFIG["tts_processing_delay"]
                self.logger.info(f"[ULTRA_SLOW] Adding {delay}s delay before TTS processing")
                await asyncio.sleep(delay)

            tts_start_time = time.perf_counter()
            self.logger.info(f"[ULTRA_SLOW] TTS node called with ultra slow timing")
            self.logger.info("[TRACE] TTS_START")

            # Prepare file paths for output WAV
            output_dir = "tts_outputs"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(output_dir, f"optimized_tts_{timestamp}.wav")

            # Collect all text from the async iterator with streaming optimization
            text_segments = []
            segment_count = 0
            async for segment in text:
                text_segments.append(segment)
                segment_count += 1
                self.logger.info(f"[STREAMING] Segment {segment_count}: '{segment}'")

                # Early TTS start for better latency (if enough content)
                if len("".join(text_segments)) > 20 and segment_count == 3:
                    self.logger.info(f"[OPTIMIZATION] Early TTS trigger at segment {segment_count}")

            full_text = "".join(text_segments)
            self.logger.info(f"[OPTIMIZED] Complete text ({len(full_text)} chars): '{full_text}'")

            # Apply SSML enhancement
            enhanced_text = self._enhance_text_with_ssml(full_text)

            # Initialize comprehensive metrics collection
            metrics_data = {
                "agent_response": full_text,
                "enhanced_text": enhanced_text if AGENT_CONFIG["enable_ssml"] else "",
                "tts_wav_path": filename,
                "tts_sample_rate": AGENT_CONFIG["tts_sample_rate"],
                "tts_model": AGENT_CONFIG["tts_model"],
                "ssml_enabled": AGENT_CONFIG["enable_ssml"],
                "streaming_enabled": AGENT_CONFIG["enable_streaming_stt"],
                "segment_count": segment_count,
                "error_occurred": False,
                "error_message": "",
                "room_name": getattr(self, "room_name", "")
            }

            # Merge in stored transcript metrics
            if hasattr(self, "current_transcript_metrics"):
                metrics_data.update(self.current_transcript_metrics)

            # Calculate various latency components
            if hasattr(self, "processing_start_time"):
                end_to_end_latency = (tts_start_time - self.processing_start_time) * 1000
                metrics_data["end_to_end_latency_ms"] = end_to_end_latency
                self.logger.info(f"[LATENCY] End-to-end latency: {end_to_end_latency:.2f} ms")

            if hasattr(self, "llm_response_time"):
                llm_to_tts_latency = (tts_start_time - self.llm_response_time) * 1000
                metrics_data["llm_to_tts_start_ms"] = llm_to_tts_latency
                self.logger.info(f"[LATENCY] LLM to TTS start: {llm_to_tts_latency:.2f} ms")

            # Optimized TTS synthesis with SSML
            synthesis_start_time = time.perf_counter()
            total_frames = 0

            try:
                # Create optimized text generator with SSML
                async def optimized_text_generator():
                    if AGENT_CONFIG["enable_ssml"]:
                        # Use enhanced text for SSML
                        yield enhanced_text
                    else:
                        # Use original segments for streaming
                        for segment in text_segments:
                            yield segment

                # Call the optimized TTS implementation
                audio_frames = []
                frame_count = 0
                async for audio_frame in super().tts_node(optimized_text_generator(), model_settings):
                    audio_frames.append(audio_frame)
                    total_frames += len(audio_frame.data)
                    frame_count += 1

                    # Log streaming progress
                    if frame_count % 10 == 0:
                        self.logger.info(f"[STREAMING] Processed {frame_count} audio frames")

                    yield audio_frame  # Yield to the pipeline

                synthesis_end_time = time.perf_counter()
                synthesis_duration_ms = (synthesis_end_time - synthesis_start_time) * 1000
                metrics_data["tts_synthesis_ms"] = synthesis_duration_ms

                # Save audio to WAV file for analysis
                if audio_frames:
                    combined_frame = rtc.combine_audio_frames(audio_frames)
                    sample_rate = combined_frame.sample_rate
                    num_channels = combined_frame.num_channels  # Correct attribute name
                    sample_width = 2  # 16-bit PCM

                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(num_channels)
                        wf.setsampwidth(sample_width)
                        wf.setframerate(sample_rate)
                        wf.writeframes(combined_frame.data.tobytes())

                    # Calculate audio duration and file size
                    audio_duration_ms = (len(combined_frame.data) / (sample_rate * num_channels)) * 1000
                    metrics_data["tts_duration_ms"] = audio_duration_ms
                    metrics_data["tts_sample_rate"] = sample_rate

                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        metrics_data["tts_file_size_bytes"] = file_size
                        self.logger.info(f"[TTS_QUALITY] Audio duration: {audio_duration_ms:.2f} ms, File size: {file_size} bytes")

                self.logger.info(f"[{datetime.now()}] TTS synthesis completed and saved to {filename}")
                self.logger.info(f"[LATENCY] TTS synthesis time: {synthesis_duration_ms:.2f} ms")
                # TRACE log for TTS end
                self.logger.info("[TRACE] TTS_END")

            except Exception as e:
                metrics_data["error_occurred"] = True
                metrics_data["error_message"] = str(e)
                self.logger.exception(f"TTS synthesis failed: {e}")
            finally:
                # Write comprehensive metrics
                self._write_comprehensive_metrics(metrics_data)

        async def on_reply(self, reply) -> None:
            self.llm_response_time = time.perf_counter()
            self.logger.info(f"[{datetime.now()}] LLM reply received. Reply type: {type(reply)}, Reply content: {reply}")

            # Calculate LLM processing time
            if hasattr(self, "llm_request_time"):
                llm_processing_ms = (self.llm_response_time - self.llm_request_time) * 1000
                self.logger.info(f"[LATENCY] LLM processing time: {llm_processing_ms:.2f} ms")

            # TRACE log for debugging pipeline
            self.logger.info(f"[TRACE] LLM_REPLY_TEXT: {reply.text}")
            # Note: TTS is now handled by the tts_node override

        async def on_connected(self, session: AgentSession):
            self.session = session # Store session for later use
            self.logger = session.logger # Store session logger for agent-specific logging
            self.room_name = session.room.name if hasattr(session, 'room') else "unknown"

            # Log session start
            self.logger.info(f"[SESSION] Agent connected to room: {self.room_name}")
            self.logger.info(f"[SESSION] Session ID: {self.session_id}")

            # Register event handlers
            session.on("user_input_transcribed", self._on_user_input_transcribed)

            # Initialize session-specific attributes
            self._initialize_session_logging()

        def _initialize_session_logging(self):
            """Initialize session-specific logging attributes"""
            if not hasattr(self, 'logger'):
                self.logger = logger  # Use module logger as fallback

            # Ensure metrics file is created
            if not os.path.exists(self.metrics_file_path):
                with open(self.metrics_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                    writer.writeheader()

            self.logger.info(f"[SESSION] Metrics logging initialized: {self.metrics_file_path}")

        def _on_user_input_transcribed(self, event: events.UserInputTranscribedEvent):
            current_time = time.perf_counter()

            # Optimized transcript processing with partial LLM prompting
            self.logger.info(f"[OPTIMIZED] Transcript event: is_final={event.is_final}, transcript='{event.transcript}'")

            if not event.is_final:
                # Streaming STT: Process interim results for partial LLM prompting
                if AGENT_CONFIG["enable_partial_llm"] and len(event.transcript) > 10:
                    self.logger.info(f"[PARTIAL_LLM] Interim transcript ({len(event.transcript)} chars): {event.transcript}")
                    # Store interim transcript for potential early processing
                    self.interim_transcript = event.transcript
                    self.interim_timestamp = current_time
                else:
                    self.logger.info(f"[STREAMING_STT] Interim: {event.transcript}")
                return

            # Optimized final transcript processing
            self.current_transcript = event.transcript
            self.transcript_final_time = current_time

            # Calculate optimized latency metrics
            mic_to_transcript_ms = None
            if hasattr(self, "audio_start_time"):
                mic_to_transcript_ms = (current_time - self.audio_start_time) * 1000
                self.logger.info(f"[OPTIMIZED_LATENCY] Mic to transcript: {mic_to_transcript_ms:.2f} ms")

            # Check if we had partial processing
            partial_processing_gain = 0
            if hasattr(self, "interim_timestamp") and AGENT_CONFIG["enable_partial_llm"]:
                partial_processing_gain = (current_time - self.interim_timestamp) * 1000
                self.logger.info(f"[PARTIAL_LLM] Processing time gained: {partial_processing_gain:.2f} ms")

            # Extract enhanced transcript metrics
            transcript_confidence = getattr(event, 'confidence', None)
            if transcript_confidence:
                self.logger.info(f"[STREAMING_STT] Confidence: {transcript_confidence}")

            # Log optimized transcript details
            self.logger.info(f"[OPTIMIZED] Final transcript received: {event.transcript}")
            self.logger.info(f"[STREAMING_STT] Length: {len(event.transcript)} chars, {len(event.transcript.split())} words")
            self.logger.info(f"[TRACE] TRANSCRIPT_FINAL: {event.transcript}")

            # Mark the start of processing for end-to-end latency measurement
            self.processing_start_time = current_time

            # Store transcript metrics for later use
            self.current_transcript_metrics = {
                "user_transcript": event.transcript,
                "transcript_confidence": transcript_confidence,
                "mic_to_transcript_ms": mic_to_transcript_ms,
                "transcript_duration_ms": len(event.transcript) * 50  # Rough estimate: 50ms per character
            }

            async def _generate_and_speak():
                # Add ultra natural response delay for very human-like timing
                if AGENT_CONFIG.get("natural_response_delay", 0) > 0:
                    delay = AGENT_CONFIG["natural_response_delay"]
                    self.logger.info(f"[ULTRA_SLOW] Adding {delay}s initial delay for ultra natural conversation flow")
                    await asyncio.sleep(delay)

                # Add thinking pause (like a human would pause to think)
                if AGENT_CONFIG.get("thinking_pause", 0) > 0:
                    thinking_delay = AGENT_CONFIG["thinking_pause"]
                    self.logger.info(f"[ULTRA_SLOW] Adding {thinking_delay}s thinking pause")
                    await asyncio.sleep(thinking_delay)

                # Add contemplation pause (additional deep thinking)
                if AGENT_CONFIG.get("contemplation_pause", 0) > 0:
                    contemplation_delay = AGENT_CONFIG["contemplation_pause"]
                    self.logger.info(f"[ULTRA_SLOW] Adding {contemplation_delay}s contemplation pause")
                    await asyncio.sleep(contemplation_delay)

                # Add LLM processing delay (simulate human processing time)
                if AGENT_CONFIG.get("llm_processing_delay", 0) > 0:
                    llm_delay = AGENT_CONFIG["llm_processing_delay"]
                    self.logger.info(f"[ULTRA_SLOW] Adding {llm_delay}s LLM processing delay")
                    await asyncio.sleep(llm_delay)

                # Mark LLM request start time
                self.llm_request_time = time.perf_counter()

                # Calculate transcript to LLM latency
                if hasattr(self, "transcript_final_time"):
                    transcript_to_llm_ms = (self.llm_request_time - self.transcript_final_time) * 1000
                    self.logger.info(f"[LATENCY] Transcript to LLM request: {transcript_to_llm_ms:.2f} ms")
                    self.current_transcript_metrics["transcript_to_llm_ms"] = transcript_to_llm_ms

                speech_handle = self._session_ref.generate_reply()

                # Add final pause before speaking
                if AGENT_CONFIG.get("final_pause", 0) > 0:
                    final_delay = AGENT_CONFIG["final_pause"]
                    self.logger.info(f"[SLOW_TIMING] Adding {final_delay}s final pause before speaking")
                    await asyncio.sleep(final_delay)

                await speech_handle # Wait for the speech to be played
                if speech_handle.chat_message:
                    await self.on_reply(speech_handle.chat_message)

            asyncio.create_task(_generate_and_speak())

        def generate_metrics_summary(self) -> dict:
            """Generate a summary report of collected metrics"""
            if not os.path.exists(self.metrics_file_path):
                return {"error": "No metrics file found"}

            try:
                with open(self.metrics_file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if not rows:
                    return {"error": "No metrics data found"}

                # Calculate summary statistics
                latencies = [float(row['end_to_end_latency_ms']) for row in rows if row['end_to_end_latency_ms']]
                wer_scores = [float(row['wer_score']) for row in rows if row['wer_score']]
                response_lengths = [int(row['response_length_words']) for row in rows if row['response_length_words']]

                summary = {
                    "session_id": self.session_id,
                    "total_interactions": len(rows),
                    "timestamp": datetime.now().isoformat(),

                    "latency_stats": {
                        "count": len(latencies),
                        "mean_ms": statistics.mean(latencies) if latencies else 0,
                        "median_ms": statistics.median(latencies) if latencies else 0,
                        "min_ms": min(latencies) if latencies else 0,
                        "max_ms": max(latencies) if latencies else 0,
                        "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
                    },

                    "wer_stats": {
                        "count": len(wer_scores),
                        "mean": statistics.mean(wer_scores) if wer_scores else 0,
                        "median": statistics.median(wer_scores) if wer_scores else 0,
                        "min": min(wer_scores) if wer_scores else 0,
                        "max": max(wer_scores) if wer_scores else 0
                    },

                    "response_stats": {
                        "mean_words": statistics.mean(response_lengths) if response_lengths else 0,
                        "median_words": statistics.median(response_lengths) if response_lengths else 0,
                        "total_words": sum(response_lengths)
                    },

                    "errors": len([row for row in rows if row['error_occurred'] == 'True'])
                }

                return summary

            except Exception as e:
                return {"error": f"Failed to generate summary: {str(e)}"}

        def log_metrics_summary(self):
            """Log a summary of metrics to the console"""
            summary = self.generate_metrics_summary()
            if "error" in summary:
                self.logger.error(f"[METRICS_SUMMARY] {summary['error']}")
                return

            self.logger.info("[METRICS_SUMMARY] ===== SESSION PERFORMANCE SUMMARY =====")
            self.logger.info(f"[METRICS_SUMMARY] Session ID: {summary['session_id']}")
            self.logger.info(f"[METRICS_SUMMARY] Total Interactions: {summary['total_interactions']}")
            self.logger.info(f"[METRICS_SUMMARY] Errors: {summary['errors']}")

            latency = summary['latency_stats']
            self.logger.info(f"[METRICS_SUMMARY] Latency - Mean: {latency['mean_ms']:.2f}ms, Median: {latency['median_ms']:.2f}ms, Range: {latency['min_ms']:.2f}-{latency['max_ms']:.2f}ms")

            wer = summary['wer_stats']
            if wer['count'] > 0:
                self.logger.info(f"[METRICS_SUMMARY] WER - Mean: {wer['mean']:.4f}, Median: {wer['median']:.4f}, Range: {wer['min']:.4f}-{wer['max']:.4f}")

            response = summary['response_stats']
            self.logger.info(f"[METRICS_SUMMARY] Responses - Mean: {response['mean_words']:.1f} words, Total: {response['total_words']} words")
            self.logger.info("[METRICS_SUMMARY] =======================================")

        def set_reference_transcript(self, interaction_id: str, reference: str):
            """Set a reference transcript for WER calculation"""
            self.reference_transcripts[interaction_id] = reference
            self.logger.info(f"[WER] Reference transcript set for interaction {interaction_id}: {reference}")

        async def on_disconnected(self):
            """Handle session cleanup and final metrics logging"""
            self.logger.info(f"[SESSION] Agent disconnected from room: {getattr(self, 'room_name', 'unknown')}")
            self.log_metrics_summary()

            # Save summary to JSON file
            summary = self.generate_metrics_summary()
            summary_file = os.path.join(os.path.dirname(__file__), f"session_summary_{self.session_id}.json")
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(f"[SESSION] Summary saved to {summary_file}")
            except Exception as e:
                self.logger.error(f"[SESSION] Failed to save summary: {e}")



async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the voice optimization agent.
    Creates agent with configurable optimization modes.
    """
    await ctx.connect()

    # Create agent with current configuration mode
    config = AGENT_CONFIG
    agent = VoiceOptimizationAgent(config_mode=CURRENT_MODE)

    logger.info(f"[CONFIG] Creating voice agent with mode: {CURRENT_MODE}")
    logger.info(f"[CONFIG] Mode description: {config['description']}")
    logger.info(f"[CONFIG] Streaming STT: {config.get('enable_streaming_stt', False)}")
    logger.info(f"[CONFIG] Partial LLM: {config.get('enable_partial_llm', False)}")
    logger.info(f"[CONFIG] SSML Enhancement: {config.get('enable_ssml', False)}")
    logger.info(f"[CONFIG] Turn Detection: {config.get('turn_detection_mode', 'vad')}")
    logger.info(f"[CONFIG] TTS Speed: {config.get('tts_speed', 1.0)}")

    # Calculate total artificial delays for human-like mode
    if CURRENT_MODE == "human_like":
        total_delay = (config.get('natural_response_delay', 0) +
                      config.get('thinking_pause', 0) +
                      config.get('contemplation_pause', 0) +
                      config.get('llm_processing_delay', 0) +
                      config.get('tts_processing_delay', 0) +
                      config.get('final_pause', 0))
        logger.info(f"[CONFIG] Total artificial delays: {total_delay}s")

    # Create session configuration based on optimization mode
    session_config = {}

    # Configure turn detection based on mode
    if config.get("turn_detection_mode") == "multilingual_model" and config.get("enable_turn_detector_plugin"):
        session_config["turn_detection"] = MultilingualModel()
    elif config.get("turn_detection_mode") == "stt":
        session_config["turn_detection"] = "stt"
    else:
        session_config["turn_detection"] = "vad"  # Default VAD mode

    # Configure VAD
    session_config["vad"] = silero.VAD.load()

    # Configure STT (Deepgram)
    session_config["stt"] = deepgram.STT(
        model=config["stt_model"],
        language=config["stt_language"],
        interim_results=config.get("stt_interim_results", False),
    )

    # Configure TTS (Cartesia)
    session_config["tts"] = cartesia.TTS(
        model=config["tts_model"],
        voice=config["tts_voice"],
        encoding=config["tts_encoding"],
        sample_rate=config["tts_sample_rate"],
        speed=config.get("tts_speed", 1.0),
    )

    # Configure endpointing delays
    session_config["min_endpointing_delay"] = config.get("min_endpointing_delay", 0.5)
    session_config["max_endpointing_delay"] = config.get("max_endpointing_delay", 6.0)
    session_config["allow_interruptions"] = config.get("allow_interruptions", True)

    # Create the session
    session = AgentSession(**session_config)

    # Store session reference in agent for metrics logging
    agent._session_ref = session
    agent.logger = logger  # Use module logger
    agent.room_name = ctx.room.name

    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )






if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))