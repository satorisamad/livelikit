# 🎙️ Voice Agent Optimizer - Streamlit Dashboard

Interactive web interface for testing and comparing voice agent optimizations with real-time configuration, testing, and benchmarking capabilities.

## 🚀 Quick Start

### 1. Setup
```bash
# Run the setup script
python setup_streamlit.py

# OR manually install requirements
pip install -r requirements_streamlit.txt
```

### 2. Configure Environment
Create a `.env` file with your API keys:
```env
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key
OPENAI_API_KEY=your_openai_api_key
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

### 3. Launch
```bash
# Using the launch script
python launch_streamlit.py

# OR directly with Streamlit
streamlit run streamlit_voice_optimizer.py
```

### 4. Access
Open http://localhost:8501 in your browser

## 🎯 Features

### 📋 Configuration Builder
- **Interactive Parameter Tuning**: Adjust VAD, turn detection, streaming STT, and SSML settings
- **Real-time Preview**: See configuration changes instantly
- **Preset Management**: Save and load custom configurations
- **Validation**: Automatic configuration validation and error checking

### 🎤 Live Testing
- **Agent Control**: Start/stop voice agents with different configurations
- **Real-time Monitoring**: Live metrics and status updates
- **Multiple Configurations**: Test baseline, optimized, and custom setups
- **Process Management**: Safe agent lifecycle management

### 📊 Benchmarking
- **Multi-Configuration Comparison**: Compare multiple setups simultaneously
- **Automated Testing**: Run standardized test suites
- **Performance Metrics**: Latency, quality, and effectiveness scoring
- **Visual Analytics**: Interactive charts and graphs

### 📈 Results Analysis
- **Historical Data**: Analyze past test results
- **Detailed Breakdowns**: Component-wise performance analysis
- **Trend Analysis**: Track improvements over time
- **Export Capabilities**: Save results for further analysis

## 🔧 Interface Overview

### Sidebar Navigation
- **Mode Selection**: Choose between Configuration Builder, Live Testing, Benchmarking, or Results Analysis
- **Quick Settings**: Access common configuration options
- **Status Indicators**: Real-time system status

### Main Dashboard
- **Tabbed Interface**: Organized sections for different optimization categories
- **Interactive Controls**: Sliders, toggles, and dropdowns for parameter adjustment
- **Real-time Feedback**: Immediate visual feedback for changes
- **Contextual Help**: Tooltips and descriptions for all options

## ⚙️ Configuration Options

### 🎯 Basic Configuration
- **Agent Type**: Baseline, Optimized, or Custom
- **LLM Settings**: Model selection, temperature, max tokens
- **Core Parameters**: Essential voice agent settings

### 🔊 VAD/Endpointing
- **Delay Settings**: Min/max endpointing delays
- **Sensitivity**: Detection threshold adjustment
- **Turn Detection**: VAD vs STT-based detection
- **Custom Parameters**: Fine-tune for specific use cases

### 🔄 Turn Detection
- **Detection Methods**: Basic VAD, STT-based, MultilingualModel
- **Sensitivity Control**: Adjust interruption handling
- **Flow Optimization**: Natural conversation settings
- **Advanced Options**: Custom threshold configuration

### 📡 Streaming STT
- **Streaming Mode**: Enable/disable interim results
- **Partial Processing**: Early LLM triggering
- **Confidence Thresholds**: 60%, 70%, 80% options
- **Latency Optimization**: Balance speed vs accuracy

### 🎵 SSML TTS
- **Enhancement Levels**: None, Basic, Healthcare, Expressive
- **Speaking Rate**: Slow, Medium, Fast
- **Emphasis Control**: None, Moderate, Strong
- **Custom SSML**: Advanced markup options

## 📊 Metrics and Analytics

### Performance Metrics
- **Latency Measurements**: End-to-end response times
- **Component Breakdown**: STT, LLM, TTS timing
- **Quality Scores**: MOS, naturalness, clarity ratings
- **Optimization Effectiveness**: Quantified improvements

### Visual Analytics
- **Real-time Charts**: Live performance monitoring
- **Comparison Graphs**: Side-by-side configuration analysis
- **Trend Lines**: Historical performance tracking
- **Distribution Plots**: Latency and quality distributions

### Export Options
- **CSV Export**: Raw metrics data
- **JSON Reports**: Structured analysis results
- **Chart Images**: High-resolution visualizations
- **Summary Reports**: Executive-level summaries

## 🎮 Usage Scenarios

### 1. Quick Optimization Testing
```
1. Select "Live Testing" mode
2. Choose "optimized" configuration
3. Click "Start Agent"
4. Test with voice input
5. Monitor real-time metrics
```

### 2. Custom Configuration Development
```
1. Select "Configuration Builder" mode
2. Start with a preset (baseline/optimized)
3. Adjust parameters in each tab
4. Save custom configuration
5. Test in Live Testing mode
```

### 3. Performance Benchmarking
```
1. Select "Benchmarking" mode
2. Choose configurations to compare
3. Set number of test interactions
4. Run benchmark comparison
5. Analyze results and visualizations
```

### 4. Historical Analysis
```
1. Select "Results Analysis" mode
2. Choose CSV files from previous tests
3. Select analysis type
4. Generate detailed reports
5. Export findings
```

## 🔧 Technical Architecture

### Frontend (Streamlit)
- **Interactive UI**: Real-time parameter adjustment
- **Visualization**: Plotly charts and graphs
- **State Management**: Session-based configuration storage
- **Responsive Design**: Works on desktop and tablet

### Backend Integration
- **Agent Launcher**: Process management for voice agents
- **Configuration Manager**: Preset and validation system
- **Metrics Collector**: Real-time data aggregation
- **File Handler**: CSV/JSON import/export

### Data Flow
```
User Input → Configuration → Agent Launch → Metrics Collection → Analysis → Visualization
```

## 🚨 Troubleshooting

### Common Issues

#### "Agent launcher not available"
- **Cause**: Missing voice agent dependencies
- **Solution**: Install full requirements with `pip install -r requirements_streamlit.txt`

#### "Failed to start agent"
- **Cause**: Missing API keys or invalid configuration
- **Solution**: Check `.env` file and configuration validity

#### "No result files found"
- **Cause**: No previous test data available
- **Solution**: Run some tests first in Live Testing mode

#### Connection errors
- **Cause**: LiveKit server issues or network problems
- **Solution**: Verify LiveKit URL and credentials

### Performance Tips
- **Use Chrome/Firefox**: Better WebRTC support for voice testing
- **Close other tabs**: Reduce browser resource usage
- **Stable internet**: Ensure reliable connection for real-time features
- **Local testing**: Use localhost for best performance

## 🔮 Advanced Features

### Custom Presets
Create and save custom configurations for specific use cases:
- Healthcare optimized
- Low-latency gaming
- High-quality podcasting
- Multi-language support

### Batch Testing
Run multiple configurations automatically:
- Overnight test suites
- A/B testing scenarios
- Performance regression testing
- Quality assurance validation

### Integration Options
- **API endpoints**: Programmatic access to configurations
- **Webhook support**: Real-time notifications
- **Database integration**: Persistent metrics storage
- **CI/CD integration**: Automated testing pipelines

## 📚 Additional Resources

- **Voice Agent Documentation**: See `README_VOICE_OPTIMIZATION.md`
- **Optimization Summary**: See `OPTIMIZATION_SUMMARY.md`
- **Demo Script**: See `demo_script.md`
- **API Reference**: Individual agent module documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Test with the Streamlit interface
5. Submit a pull request

## 📄 License

This Streamlit dashboard is part of the LiveKit Voice Agent Optimization project and follows the same licensing terms.
