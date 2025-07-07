# Voice Agent Metrics Collection and Analysis

This enhanced voice agent includes comprehensive metrics collection for evaluating performance across multiple dimensions:

## Metrics Collected

### 1. End-to-End Latency
- **Mic to Transcript**: Time from audio input to final transcript
- **Transcript to LLM**: Time from transcript to LLM request
- **LLM to TTS Start**: Time from LLM response to TTS synthesis start
- **TTS Synthesis**: Time to synthesize audio
- **End-to-End**: Total time from mic input to TTS output starts

### 2. Transcript Quality (WER)
- **Word Error Rate**: Calculated when reference transcripts are available
- **Transcript Confidence**: STT confidence scores
- **Transcript Length**: Character and word counts

### 3. TTS Quality Metrics
- **Audio Duration**: Length of synthesized audio
- **File Size**: Size of generated WAV files
- **Real-Time Factor**: Synthesis time / audio duration (lower is better)
- **Sample Rate**: Audio quality parameters

### 4. Subjective Quality (MOS)
- **Clarity Score**: 1-5 rating for audio clarity
- **Expressiveness Score**: 1-5 rating for naturalness
- **Overall MOS**: 1-5 overall quality rating

## Files Generated

### 1. `voice_agent_metrics.csv`
Main metrics file with all collected data:
- Session tracking with unique interaction IDs
- Timing measurements for all pipeline components
- TTS quality metrics
- Error tracking
- Placeholder columns for manual quality ratings

### 2. `session_summary_[SESSION_ID].json`
Session summary with statistical analysis:
- Latency statistics (mean, median, percentiles)
- WER analysis
- Error rates
- Performance trends

### 3. `tts_outputs/`
Directory containing all generated TTS audio files:
- Timestamped WAV files for each response
- Used for subjective quality evaluation

## Usage

### 1. Running the Voice Agent
```bash
python voice_agent.py
```

The agent automatically logs all metrics during operation.

### 2. Analyzing Metrics
```bash
# Basic analysis
python metrics_analyzer.py voice_agent_metrics.csv

# Save detailed report
python metrics_analyzer.py voice_agent_metrics.csv --output analysis_report.json
```

### 3. Subjective Quality Evaluation
```bash
# Evaluate all samples
python mos_evaluator.py voice_agent_metrics.csv

# Evaluate specific range
python mos_evaluator.py voice_agent_metrics.csv --start-from 10 --max-samples 20
```

The MOS evaluator will:
- Play each TTS audio file
- Prompt for clarity, expressiveness, and overall quality ratings
- Save progress after each evaluation
- Generate updated CSV with MOS scores

## Key Logging Features

### Trace Logs
Look for these trace markers in the logs:
- `[TRACE] TRANSCRIPT_FINAL`: Final user transcript
- `[TRACE] LLM_REPLY_TEXT`: Agent response text
- `[TRACE] TTS_START/TTS_END`: TTS synthesis boundaries

### Latency Logs
- `[LATENCY]` prefix for all timing measurements
- Component-wise breakdown of processing time
- End-to-end latency tracking

### Quality Logs
- `[TRANSCRIPT_QUALITY]` for STT metrics
- `[TTS_QUALITY]` for synthesis metrics
- `[METRICS]` for comprehensive data logging

### Session Logs
- `[SESSION]` for connection/disconnection events
- `[METRICS_SUMMARY]` for session performance summary

## Analyzing Results

### Latency Analysis
- **Target**: End-to-end latency < 1000ms for good user experience
- **Components**: Identify bottlenecks in the pipeline
- **Trends**: Monitor performance over time

### WER Analysis
- **Target**: WER < 0.1 (10%) for good transcript quality
- **Factors**: Analyze correlation with audio quality, background noise
- **Improvement**: Use reference transcripts for accurate WER calculation

### TTS Quality
- **Real-Time Factor**: Target < 0.5 for responsive synthesis
- **MOS Scores**: Target > 3.5 for acceptable quality
- **Consistency**: Monitor variation in quality metrics

## Setting Reference Transcripts

For accurate WER calculation, set reference transcripts:

```python
# In your agent code
agent.set_reference_transcript("interaction_id", "expected transcript text")
```

Or manually add to the CSV file in the `reference_transcript` column.

## Customization

### Adding New Metrics
1. Add column to `csv_fieldnames` in the agent constructor
2. Collect data in the appropriate event handler
3. Include in `metrics_data` dictionary before calling `_write_comprehensive_metrics`

### Custom Analysis
Extend `metrics_analyzer.py` with additional analysis functions:
- Correlation analysis between metrics
- Performance regression detection
- Custom visualization

## Dependencies

For the analysis tools:
```bash
pip install pandas matplotlib seaborn
```

## Best Practices

1. **Regular Analysis**: Run metrics analysis after each session
2. **Reference Data**: Maintain reference transcripts for WER calculation
3. **Quality Evaluation**: Regularly evaluate TTS quality with MOS scores
4. **Performance Monitoring**: Track latency trends over time
5. **Error Investigation**: Analyze error patterns and frequencies

## Troubleshooting

### Common Issues
- **Missing Audio Files**: Check `tts_outputs/` directory permissions
- **CSV Encoding**: Ensure UTF-8 encoding for international characters
- **Large Files**: Consider rotating metrics files for long-running sessions

### Performance Impact
- Metrics collection adds minimal overhead (~1-2ms per interaction)
- Audio file storage can consume disk space
- Consider cleanup policies for old audio files
