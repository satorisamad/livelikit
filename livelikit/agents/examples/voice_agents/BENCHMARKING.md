# Voice Agent Benchmarking

This document outlines the procedures for benchmarking the LiveKit voice agent's performance in terms of end-to-end latency, transcript Word Error Rate (WER), and TTS clarity/expressiveness (subjective MOS).

## 1. End-to-End Latency Measurement

**Objective:** Measure the time from when mic input is received until the TTS output starts.

**Procedure:**
1.  **Run the Agent:** Start the `voice_agent.py` script in development mode:
    ```bash
    python voice_agent.py dev
    ```
2.  **Interact with the Agent:** Engage in a conversation with the agent. Speak clearly and allow the agent to respond fully.
3.  **Collect Logs:** The agent's console output will contain timestamps. Look for the following log entries:
    *   `[YYYY-MM-DD HH:MM:SS.microseconds] Transcription processing started for: <USER_TRANSCRIPT>`: This marks the beginning of the agent's processing for a given user input.
    *   `[YYYY-MM-DD HH:MM:SS.microseconds] Entering _tts method with text: <LLM_REPLY_TEXT>`: This marks when the TTS synthesis begins for the agent's reply.
4.  **Calculate Latency:** For each turn, subtract the timestamp of "Transcription processing started" from the timestamp of "Entering _tts method". This will give you the processing latency for that turn.
5.  **Analyze:** Collect data for multiple turns and calculate average, minimum, and maximum latencies.

## 2. Transcript Word Error Rate (WER) Benchmarking

**Objective:** Evaluate the accuracy of the agent's Speech-to-Text (STT) transcription.

**Procedure:**
1.  **Record User Input:** Simultaneously record your spoken input using an external recording device or software (e.g., Audacity, OBS Studio) while interacting with the agent. Ensure high-quality audio recording.
2.  **Generate Ground Truth Transcripts:** Manually transcribe your recorded audio inputs to create accurate "ground truth" text files. Each ground truth transcript should correspond to a specific user utterance.
3.  **Collect Agent Transcripts:** The agent's logs will contain the final transcriptions:
    *   `[YYYY-MM-DD HH:MM:SS.microseconds] Final transcription received: <AGENT_TRANSCRIPT>`
    Extract these agent-generated transcripts.
4.  **Compare and Calculate WER:** Use a WER calculation tool or script (e.g., `jiwer` Python library) to compare each ground truth transcript with its corresponding agent-generated transcript. The formula for WER is:
    `WER = (Substitutions + Insertions + Deletions) / Total_Words_in_Ground_Truth`
5.  **Analyze:** Calculate the average WER across multiple utterances to get an overall accuracy score.

## 3. TTS Clarity, Expressiveness (Subjective MOS) Evaluation

**Objective:** Subjectively assess the quality of the agent's Text-to-Speech (TTS) output.

**Procedure:**
1.  **Collect TTS Audio Files:** The `_tts` method in `voice_agent.py` is configured to save synthesized audio to WAV files in the `tts_outputs` directory. These files are named `tts_output_YYYYMMDD_HHMMSS_microseconds.wav`.
2.  **Prepare for MOS Evaluation:**
    *   Select a diverse set of TTS audio samples from the `tts_outputs` directory.
    *   Consider anonymizing the files or randomizing their order if multiple evaluators are involved.
3.  **Conduct Subjective Listening Tests (MOS):**
    *   Have human evaluators listen to each TTS audio sample.
    *   Ask them to rate the clarity, naturalness, and expressiveness of the speech on a scale (e.g., 1-5, where 1 is poor and 5 is excellent).
    *   A common scale is the Mean Opinion Score (MOS) scale:
        *   5: Excellent
        *   4: Good
        *   3: Fair
        *   2: Poor
        *   1: Bad
4.  **Calculate Average MOS:** Average the scores from all evaluators for each audio sample, and then average across all samples to get an overall MOS for the TTS system.
5.  **Analyze:** Use the MOS scores to understand the perceived quality of the TTS output and identify areas for improvement.
