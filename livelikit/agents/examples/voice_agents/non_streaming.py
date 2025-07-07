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

# Note: Now using prompt.txt file directly instead of agent_prompts.py

logger = logging.getLogger("livekit.agents")
logger.setLevel(logging.INFO)

load_dotenv()

# Configuration for the voice agent
AGENT_CONFIG = {
    "prompt_type": "default",  # Change this to use different prompts: "casual", "professional", "energetic", etc.
    "llm_model": "gpt-4o-mini",
    "tts_model": "sonic-2",
    "tts_sample_rate": 24000,
    "tts_encoding": "pcm_s16le"
}

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    class MyAgent(Agent):
        def __init__(self) -> None:
            logger.info(f"[CONFIG] MyAgent.__init__ called - loading prompt from prompt.txt")

            # Load instructions from prompt.txt file
            try:
                prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    instructions = f.read().strip()
                logger.info(f"[CONFIG] Successfully loaded prompt from prompt.txt")
                logger.info(f"[CONFIG] Prompt length: {len(instructions)} characters")
                logger.info(f"[CONFIG] Prompt preview: {instructions[:150]}...")
            except FileNotFoundError:
                logger.error(f"[CONFIG] prompt.txt file not found at {prompt_file_path}")
                instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
                logger.info(f"[CONFIG] Using fallback instructions")
            except Exception as e:
                logger.error(f"[CONFIG] Error loading prompt.txt: {e}")
                instructions = "You are Sarah, a helpful AI voice assistant for Harmony Health Clinic."
                logger.info(f"[CONFIG] Using fallback instructions")

            super().__init__(
                llm=openai.LLM(model=AGENT_CONFIG["llm_model"]),
                instructions=instructions,
                tts=cartesia.TTS(
                    model=AGENT_CONFIG["tts_model"],
                    encoding=AGENT_CONFIG["tts_encoding"],
                    sample_rate=AGENT_CONFIG["tts_sample_rate"]
                )
            )

            # Initialize comprehensive metrics logging
            self.metrics_file_path = os.path.join(os.path.dirname(__file__), "voice_agent_metrics.csv")
            self.csv_fieldnames = [
                "session_id", "timestamp", "interaction_id",
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

        async def tts_node(self, text, model_settings):
            """Override the TTS node to intercept TTS calls and collect metrics"""
            tts_start_time = time.perf_counter()
            self.logger.info(f"[DEBUG] tts_node called with text stream")
            self.logger.info("[TRACE] TTS_START")

            # Prepare file paths for output WAV
            output_dir = "tts_outputs"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(output_dir, f"tts_output_{timestamp}.wav")

            # Collect all text from the async iterator
            text_segments = []
            async for segment in text:
                text_segments.append(segment)
                self.logger.info(f"[DEBUG] TTS text segment: '{segment}'")

            full_text = "".join(text_segments)
            self.logger.info(f"[DEBUG] Full TTS text: '{full_text}'")

            # Initialize metrics collection with stored transcript metrics
            metrics_data = {
                "agent_response": full_text,
                "tts_wav_path": filename,
                "tts_sample_rate": 24000,
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

            # Call the default TTS node implementation
            synthesis_start_time = time.perf_counter()
            total_frames = 0

            try:
                # Create async generator from text segments
                async def text_generator():
                    for segment in text_segments:
                        yield segment

                # Call the default TTS implementation
                audio_frames = []
                async for audio_frame in super().tts_node(text_generator(), model_settings):
                    audio_frames.append(audio_frame)
                    total_frames += len(audio_frame.data)
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

            # Debug: Always log that this method is called
            self.logger.info(f"[DEBUG] _on_user_input_transcribed called: is_final={event.is_final}, transcript='{event.transcript}'")

            if not event.is_final:
                # Log interim transcripts for debugging
                self.logger.info(f"[TRANSCRIPT_INTERIM] {event.transcript}")
                return

            # Final transcript processing
            self.current_transcript = event.transcript
            self.transcript_final_time = current_time

            # Calculate mic-to-transcript latency if we have audio start time
            mic_to_transcript_ms = None
            if hasattr(self, "audio_start_time"):
                mic_to_transcript_ms = (current_time - self.audio_start_time) * 1000
                self.logger.info(f"[LATENCY] Mic to transcript: {mic_to_transcript_ms:.2f} ms")

            # Extract transcript confidence if available
            transcript_confidence = getattr(event, 'confidence', None)
            if transcript_confidence:
                self.logger.info(f"[TRANSCRIPT_QUALITY] Confidence: {transcript_confidence}")

            # Log transcript details
            self.logger.info(f"[{datetime.now()}] Final transcription received: {event.transcript}")
            self.logger.info(f"[TRANSCRIPT_QUALITY] Length: {len(event.transcript)} chars, {len(event.transcript.split())} words")
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
                # Mark LLM request start time
                self.llm_request_time = time.perf_counter()

                # Calculate transcript to LLM latency
                if hasattr(self, "transcript_final_time"):
                    transcript_to_llm_ms = (self.llm_request_time - self.transcript_final_time) * 1000
                    self.logger.info(f"[LATENCY] Transcript to LLM request: {transcript_to_llm_ms:.2f} ms")
                    self.current_transcript_metrics["transcript_to_llm_ms"] = transcript_to_llm_ms

                speech_handle = self._session_ref.generate_reply()
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



    # Create agent instance first
    logger.info(f"[CONFIG] Creating agent with prompt from prompt.txt file")
    agent = MyAgent()

    session = AgentSession(
        turn_detection=MultilingualModel(unlikely_threshold=0.05),
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
            interim_results=False,
        ),
        tts=cartesia.TTS(
            model="sonic-2",
            encoding="pcm_s16le",
            sample_rate=24000
        ),
    )

    # Store session reference in agent for metrics logging
    agent._session_ref = session
    agent.logger = logger  # Use module logger
    agent.room_name = ctx.room.name

    # Register event handlers on the session
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: events.UserInputTranscribedEvent):
        agent._on_user_input_transcribed(event)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    await ctx.connect()






if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))