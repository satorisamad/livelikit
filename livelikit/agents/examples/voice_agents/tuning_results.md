# Voice Agent Endpointing Tuning Results

This document summarizes the results of experiments conducted to tune the endpointing behavior of the LiveKit Voice Agent. The primary goal is to achieve a natural, responsive conversational flow by minimizing response latency while avoiding premature interruptions or false triggers.

## Experiment 1: STT-Based Endpointing (`turn_detection="stt"`)

This set of experiments focused on tuning Deepgram's native endpointing capabilities using the `endpointing_ms` parameter.

### Methodology
### Quantitative Results

- **Response Latency**: ~1000-1800ms
- **False Triggers**: High. Logs showed frequent sentence fragmentation (e.g., `"Nothing much. I'm just"` and `"what my wife has made."` were treated as separate inputs).

### Qualitative Observations

- **User Feedback**: The agent felt very responsive and "human-like" but was overly sensitive. It frequently interrupted if the user paused even for a moment, making it hard to complete a thought. The user had to speak continuously to avoid being cut off.

---

## Experiment 2: Increased VAD Delay

- **`min_endpointing_delay`**: `1.0s`
- **`max_endpointing_delay`**: `3.0s`

### Quantitative Results

- **Response Latency**: ~1300-1900ms
- **False Triggers**: Significantly reduced. Some minor fragmentation still occurred on longer pauses, but it was much less frequent than the previous experiment.

### Qualitative Observations

- **User Feedback**: A major improvement. The user reported that the conversation felt much more natural and "human." They could pause mid-sentence without being interrupted, and the agent waited to capture the full thought before responding. The flow was much smoother.

## STT-Based Endpointing (Deepgram)

### Experiment 1: `endpointing_ms = 500ms`

*   **Description**: Switched from VAD-based to STT-based endpointing (`turn_detection="stt"`) to leverage the speech-to-text provider's native end-of-utterance detection.
*   **Parameters**:
    *   `turn_detection`: `"stt"`
    *   `stt`: `deepgram.STT(endpointing_ms=500)`
*   **Hypothesis**: Using Deepgram's endpointing will be more accurate than generic VAD, reducing false triggers from short thinking pauses while maintaining low latency. The 500ms delay should provide a good balance between responsiveness and avoiding premature cut-offs.
*   **Quantitative Data**:
    *   Response Latency: Very low. The agent responded almost instantly after speech stopped.
    *   False Triggers: High. The agent frequently misinterpreted short pauses as the end of a turn, leading to interruptions.
    *   Missed Endpoints: Low. The agent did not miss the end of speech.
*   **Qualitative Observations**:
    *   The user experience was poor due to constant interruptions. The 500ms delay is too aggressive and does not allow for natural thinking pauses in conversation.
    *   The user frequently had to say "Stop" to halt the agent's premature responses, indicating a frustrating conversational flow.
*   **Conclusion**: While responsive, the 500ms endpointing delay is too short and creates an unnatural, interruptive conversational experience. A longer delay is needed.

### Experiment 2: `endpointing_ms = 1000ms`

*   **Description**: Increasing the STT endpointing delay to reduce interruptions.
*   **Parameters**:
    *   `turn_detection`: `"stt"`
    *   `stt`: `deepgram.STT(endpointing_ms=1000)`
*   **Hypothesis**: Increasing the delay to 1000ms will give users enough time for natural pauses, reducing false triggers and interruptions, leading to a smoother conversation.
*   **Quantitative Data**:
    *   Response Latency: Moderate. Noticeably higher than 500ms, but still acceptable.
    *   False Triggers: Medium. A significant reduction compared to 500ms, but interruptions still occurred when the user paused to think.
    *   Missed Endpoints: Low.
*   **Qualitative Observations**:
    *   The conversational flow was much improved. The user was able to speak more naturally without being cut off as frequently.
    *   However, the user still had to use "Stop" on several occasions, indicating that the 1000ms delay is not sufficient for longer, more complex sentences.
*   **Conclusion**: A clear improvement, but still not ideal. The delay needs to be increased further to accommodate more natural speaking patterns.

### Experiment 3: `endpointing_ms = 1500ms`

*   **Description**: Increasing the STT endpointing delay further to eliminate remaining interruptions.
*   **Parameters**:
    *   `turn_detection`: `"stt"`
    *   `stt`: `deepgram.STT(endpointing_ms=1500)`
*   **Hypothesis**: A 1500ms delay will strike the optimal balance, providing enough time for users to pause and formulate their thoughts without making the agent feel sluggish.
*   **Quantitative Data**:
    *   Response Latency: TBD
    *   False Triggers: TBD
    *   Missed Endpoints: TBD
*   **Qualitative Observations**:
    *   TBD

## VAD-Based Endpointing

### Experiment 1

*   **Parameters**:
    *   `min_endpointing_delay`: 0.2s
    *   `max_endpointing_delay`: 3.0s
*   **Quantitative Data**:
    *   *Response Latency*: (No data from logs)
    *   *False Triggers*: (No data from logs)
*   **Qualitative Observations**:
    *   User feedback: "its good". This suggests the responsiveness is acceptable and it doesn't seem to be interrupting too aggressively in this case.

### Experiment 2

*   **Parameters**:
    *   `min_endpointing_delay`: 0.5s
    *   `max_endpointing_delay`: 3.0s
*   **Quantitative Data**:
    *   *Response Latency*: (No data from logs)
    *   *False Triggers*: High. Transcripts show multiple incomplete sentences being sent to the LLM (e.g., "Hey. I'm good to", "Hi. Can you explain me about the").
*   **Qualitative Observations**:
    *   (Waiting for user feedback)

---
