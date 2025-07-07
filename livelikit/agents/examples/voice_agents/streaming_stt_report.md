# Report: Streaming STT & Partial Hypotheses Evaluation

This document evaluates the impact of using streaming STT and prompting the LLM with partial transcripts to reduce response latency.

## Test Session 1: Observations (2025-07-07)

### 1. Low-Latency Partial Transcript Prompting
- The implementation successfully prompts the LLM on the first partial transcript received after the user starts speaking, confirmed by `EVAL: Prompting LLM on partial transcript...` logs.
- This aggressive strategy leads to a highly responsive feel, with initial response latencies often between **1.0s - 1.5s**.

### 2. Accuracy vs. Responsiveness Trade-off
- The logs frequently show `EVAL: Final transcript differs from the partial that triggered the LLM.`, highlighting the core trade-off of this approach.
- **Notable Examples:**
  - Partial: `'an'` → Final: `'have an'`
  - Partial: `'So'` → Final: `'Book a flight to'`
  - Multilingual Correction: Partial `'तुम्हारे'` (Hindi) → Final `'to berlin'` (English).
- Prompting on these inaccurate early partials can cause the agent to generate irrelevant or nonsensical responses, requiring it to self-correct once the final transcript arrives.

### 3. High Latency Issue with Turn Detection
- A significant issue was observed where response latency spiked to **~7 seconds** on multiple occasions.
- **Analysis:** This delay is not from the LLM. It's a **~5-second gap** between the STT providing a final transcript and the `MultilingualModel` turn detector confirming the user's turn has ended.
- **Cause:** This occurs when the user speaks a short phrase (e.g., "Yes.") and pauses. The turn detector sees a low `eou_probability` and waits for a longer period of silence before firing.
- **Impact:** This delay is the primary bottleneck in the conversation flow and counteracts the benefits gained from partial transcript prompting.

### 4. Summary & Next Steps
- The aggressive partial-prompting strategy is functionally working but can lead to a slightly disjointed experience where the agent is fast but sometimes wrong.
- The main performance bottleneck is the turn-end detection logic, especially for short utterances.
- **Recommendation:** We should investigate tuning the `MultilingualModel` turn detector or adjusting our response strategy. We could, for example, wait for a slightly longer partial transcript before prompting the LLM to improve the initial accuracy without adding significant latency.

## Test Session 2: Observations (2025-07-07) - 3-Word Threshold

### 1. Improved Accuracy and Relevance
- The new strategy of waiting for a partial transcript of at least 3 words before prompting the LLM proved effective.
- While `Final transcript differs...` messages still occurred, the partial transcripts were much closer to the final meaning. This resulted in more relevant agent responses and a less disjointed conversational flow.
- **Example:** Partial `'the model is not responding'` vs. Final `'the model is not responding properly.'` The agent's response to the partial is still highly relevant.

### 2. Turn-End Latency Remains the Primary Bottleneck
- The issue of high latency after short user utterances persists, confirming that the `MultilingualModel` turn detector is the root cause.
- **Example:** After the user said, `"It's not good."`, the response latency was **6.59s**. The logs show the turn detector waited nearly 5 seconds before signaling the end of the user's turn.
- This confirms that tuning our prompting strategy alone is not enough. We must address the turn detector's behavior.

### 3. Summary & Next Steps
- The 3-word threshold is a good balance between responsiveness and accuracy. We should keep this change.
- The highest priority is now to **reduce the turn-end detection latency**. 
- **Recommendation:** Investigate the tunable parameters of the `MultilingualModel` to make it more sensitive and faster to react to short phrases and pauses.

## Test Session 3: Tuning Turn-End Detection

- **Objective**: Test the effect of setting `unlikely_threshold=0.1` in the `MultilingualModel` to reduce turn-end detection latency.
- **Observations**:
    - The new threshold successfully reduced latency for some short utterances (e.g., "Yes, babe."), with response times improving to ~2.3s.
    - However, for other phrases (e.g., "I think just for a peaceful ride," and "No. Just exploring things."), the calculated `eou_probability` remained below the `0.1` threshold.
    - This caused the agent to wait for a timeout before responding, resulting in high response latencies of ~6.7s.
- **Conclusion**: The `unlikely_threshold` of `0.1` is an improvement but is still not sensitive enough to eliminate the latency bottleneck in all conversational scenarios.

## Test Session 4: Final Tuning with `unlikely_threshold=0.05`

- **Objective**: Test the effect of setting `unlikely_threshold=0.05` to achieve consistent low-latency responses.
- **Observations**:
    - The new threshold worked very well for most short utterances, with response latencies consistently between **1.5s and 2.3s**.
    - The agent was responsive and the conversation felt natural and fluid for the majority of the test.
    - However, one utterance ("I think you should move on with next") resulted in a very low `eou_probability` of `0.005`, which was below our threshold. This caused a ~7s latency for that turn.
- **Conclusion**: The `unlikely_threshold` of `0.05` provides a good balance, but a single static threshold cannot perfectly handle all conversational nuances. Further improvements would likely require a more dynamic turn-detection strategy.

## Project Conclusion

We have successfully integrated and optimized streaming STT in the voice agent. By systematically testing, identifying the turn-end detection bottleneck, and tuning the `MultilingualModel`'s `unlikely_threshold`, we have dramatically reduced response latency. The agent is now significantly more responsive, providing a much more natural conversational experience. While minor edge cases remain, the primary objective of this project has been achieved.

## Test Configuration

- **Strategy**: Aggressive low-latency. The LLM is prompted with the *very first* partial transcript received from the user.
- **`interrupt_on_new_user_input`**: `True`

## Test Scenarios & Observations

Use the following scenarios to test the system. Fill in the table with your observations.

| Utterance | Partial Trigger | Final Transcript | Latency (s) | LLM Response Quality (1-5) | Notes (Corrections, Hallucinations) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Simple Question**<br>"What is the capital of France?" | | | | | |
| **Corrected Utterance**<br>"Book a flight to... I mean, a train to Berlin." | | | | | |
| **Long Sentence**<br>"Can you tell me about the history of the Eiffel Tower and how long it took to build?" | | | | | |
| **Ambiguous Start**<br>"Yeah, so about that thing we discussed..." | | | | | |

### Metrics Guide

- **Utterance**: The phrase you speak to the agent.
- **Partial Trigger**: The partial transcript that first triggered the LLM (from the logs).
- **Final Transcript**: The final, corrected transcript.
- **Latency (s)**: Time from when you *stop* speaking to when the agent *starts* speaking (from the logs).
- **LLM Response Quality (1-5)**: 
  - 5: Perfect, coherent, and complete.
  - 4: Good, but minor issues.
  - 3: Understandable, but has clear errors or hallucinations.
  - 2: Mostly incorrect or nonsensical.
  - 1: Completely wrong.
- **Notes**: Mention if the agent had to correct itself, if it hallucinated facts based on the partial, etc.
