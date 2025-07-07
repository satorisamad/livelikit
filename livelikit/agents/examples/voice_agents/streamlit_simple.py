#!/usr/bin/env python3
"""
Simple Streamlit Voice Agent Optimizer

A simplified version that works without complex dependencies.
Provides configuration building and basic analysis capabilities.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import glob

# Page configuration
st.set_page_config(
    page_title="Voice Agent Optimizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration definitions
OPTIMIZATION_CONFIGS = {
    "baseline": {
        "name": "Baseline Configuration",
        "description": "Basic voice agent without optimizations",
        "features": {
            "streaming_stt": False,
            "partial_llm": False,
            "turn_detector_plugin": False,
            "ssml_enhancement": False
        },
        "parameters": {
            "min_endpointing_delay": 0.5,
            "max_endpointing_delay": 6.0,
            "stt_model": "nova-2",
            "tts_speed": 1.0,
            "llm_temperature": 0.7
        }
    },
    "optimized": {
        "name": "Optimized Configuration",
        "description": "Fully optimized with all features enabled",
        "features": {
            "streaming_stt": True,
            "partial_llm": True,
            "turn_detector_plugin": True,
            "ssml_enhancement": True
        },
        "parameters": {
            "min_endpointing_delay": 0.2,
            "max_endpointing_delay": 3.0,
            "stt_model": "nova-2",
            "tts_speed": 1.0,
            "llm_temperature": 0.7,
            "confidence_threshold": 0.7
        }
    }
}

def main():
    st.title("🎙️ Voice Agent Optimization Dashboard")
    st.markdown("Interactive configuration and analysis tool for voice agent optimizations")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🔧 Navigation")
        mode = st.selectbox(
            "Select Mode",
            ["Configuration Builder", "Results Analysis", "Documentation"],
            help="Choose the main operation mode"
        )
    
    if mode == "Configuration Builder":
        show_configuration_builder()
    elif mode == "Results Analysis":
        show_results_analysis()
    elif mode == "Documentation":
        show_documentation()

def show_configuration_builder():
    """Configuration builder interface"""
    st.header("⚙️ Configuration Builder")
    
    # Base configuration selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Base Configuration")
        base_config = st.selectbox(
            "Start with preset",
            list(OPTIMIZATION_CONFIGS.keys()),
            format_func=lambda x: OPTIMIZATION_CONFIGS[x]["name"]
        )
        
        if base_config:
            config_info = OPTIMIZATION_CONFIGS[base_config]
            st.write(f"**{config_info['name']}**")
            st.write(config_info['description'])
    
    with col2:
        st.subheader("Optimization Toggles")
        
        # Get base configuration
        base_features = OPTIMIZATION_CONFIGS[base_config]["features"]
        base_params = OPTIMIZATION_CONFIGS[base_config]["parameters"]
        
        # Feature toggles
        st.write("**Core Features:**")
        streaming_stt = st.checkbox(
            "🔄 Streaming STT", 
            value=base_features["streaming_stt"],
            help="Enable real-time speech-to-text processing"
        )
        
        partial_llm = st.checkbox(
            "⚡ Partial LLM Processing", 
            value=base_features["partial_llm"],
            help="Start LLM processing before complete transcript"
        )
        
        turn_detector = st.checkbox(
            "🎯 Advanced Turn Detection", 
            value=base_features["turn_detector_plugin"],
            help="Use MultilingualModel for better conversation flow"
        )
        
        ssml_enhancement = st.checkbox(
            "🎵 SSML Enhancement", 
            value=base_features["ssml_enhancement"],
            help="Enhanced speech synthesis with SSML"
        )
    
    # Parameter tuning
    st.subheader("🔧 Parameter Tuning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Endpointing**")
        min_delay = st.slider(
            "Min Delay (s)", 
            0.1, 2.0, 
            base_params["min_endpointing_delay"], 
            0.1
        )
        max_delay = st.slider(
            "Max Delay (s)", 
            1.0, 10.0, 
            base_params["max_endpointing_delay"], 
            0.5
        )
    
    with col2:
        st.write("**Speech Processing**")
        tts_speed = st.slider(
            "TTS Speed", 
            0.5, 2.0, 
            base_params["tts_speed"], 
            0.1
        )
        
        if partial_llm:
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.5, 1.0, 
                base_params.get("confidence_threshold", 0.7), 
                0.05
            )
        else:
            confidence_threshold = 0.0
    
    with col3:
        st.write("**LLM Settings**")
        llm_temperature = st.slider(
            "LLM Temperature", 
            0.0, 2.0, 
            base_params["llm_temperature"], 
            0.1
        )
        
        stt_model = st.selectbox(
            "STT Model",
            ["nova-2", "nova", "base"],
            index=0
        )
    
    # Build final configuration
    final_config = {
        "name": f"Custom Configuration ({datetime.now().strftime('%H:%M:%S')})",
        "base": base_config,
        "features": {
            "streaming_stt": streaming_stt,
            "partial_llm": partial_llm,
            "turn_detector_plugin": turn_detector,
            "ssml_enhancement": ssml_enhancement
        },
        "parameters": {
            "min_endpointing_delay": min_delay,
            "max_endpointing_delay": max_delay,
            "stt_model": stt_model,
            "tts_speed": tts_speed,
            "llm_temperature": llm_temperature,
            "confidence_threshold": confidence_threshold
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Display configuration summary
    st.subheader("📋 Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Enabled Features:**")
        for feature, enabled in final_config["features"].items():
            status = "✅" if enabled else "❌"
            st.write(f"{status} {feature.replace('_', ' ').title()}")
    
    with col2:
        st.write("**Parameters:**")
        for param, value in final_config["parameters"].items():
            if isinstance(value, float):
                st.write(f"• {param.replace('_', ' ').title()}: {value:.2f}")
            else:
                st.write(f"• {param.replace('_', ' ').title()}: {value}")
    
    # Performance estimation
    st.subheader("📊 Estimated Performance Impact")
    
    # Calculate estimated improvements
    baseline_latency = 2800  # ms
    estimated_latency = baseline_latency
    
    if streaming_stt:
        estimated_latency *= 0.7  # 30% improvement
    if partial_llm:
        estimated_latency *= 0.8  # 20% improvement
    if turn_detector:
        estimated_latency *= 0.9  # 10% improvement
    
    improvement = ((baseline_latency - estimated_latency) / baseline_latency) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Estimated Latency", 
            f"{estimated_latency:.0f}ms",
            delta=f"-{improvement:.1f}%"
        )
    
    with col2:
        quality_score = 3.0
        if ssml_enhancement:
            quality_score += 0.8
        if turn_detector:
            quality_score += 0.5
        
        st.metric(
            "Quality Score", 
            f"{quality_score:.1f}/5.0",
            delta=f"+{quality_score-3.0:.1f}"
        )
    
    with col3:
        complexity = sum(final_config["features"].values())
        st.metric(
            "Complexity Level", 
            f"{complexity}/4",
            delta=f"{complexity-2:+d}" if complexity != 2 else None
        )
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        config_file = f"custom_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_file, 'w') as f:
            json.dump(final_config, f, indent=2)
        st.success(f"Configuration saved to {config_file}")
        
        # Show next steps
        st.info("""
        **Next Steps:**
        1. Use this configuration with the voice agent scripts
        2. Run tests to measure actual performance
        3. Analyze results in the Results Analysis tab
        """)

def show_results_analysis():
    """Results analysis interface"""
    st.header("📈 Results Analysis")
    
    # Look for CSV files
    csv_files = glob.glob("*_metrics_*.csv")
    
    if not csv_files:
        st.warning("No metrics files found. Run some voice agent tests first!")
        st.info("""
        To generate metrics files, run the voice agent scripts:
        - `python baseline_voice_agent.py`
        - `python optimized_voice_agent.py`
        - etc.
        """)
        return
    
    # File selection
    selected_files = st.multiselect(
        "Select metrics files to analyze",
        csv_files,
        default=csv_files[:2] if len(csv_files) >= 2 else csv_files
    )
    
    if not selected_files:
        st.info("Select one or more metrics files to analyze.")
        return
    
    # Load and analyze data
    all_data = []
    for file in selected_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = file
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    if not all_data:
        st.error("No valid data found in selected files.")
        return
    
    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Analysis options
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Overview", "Latency Analysis", "Quality Comparison", "Configuration Impact"]
    )
    
    if analysis_type == "Overview":
        show_overview_analysis(combined_df)
    elif analysis_type == "Latency Analysis":
        show_latency_analysis(combined_df)
    elif analysis_type == "Quality Comparison":
        show_quality_analysis(combined_df)
    elif analysis_type == "Configuration Impact":
        show_configuration_analysis(combined_df)

def show_overview_analysis(df):
    """Show overview analysis"""
    st.subheader("📊 Overview Analysis")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", len(df))
    
    with col2:
        if 'end_to_end_latency_ms' in df.columns:
            avg_latency = pd.to_numeric(df['end_to_end_latency_ms'], errors='coerce').mean()
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    with col3:
        unique_configs = df['source_file'].nunique() if 'source_file' in df.columns else 0
        st.metric("Configurations", unique_configs)
    
    with col4:
        if 'config_mode' in df.columns:
            unique_modes = df['config_mode'].nunique()
            st.metric("Test Modes", unique_modes)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

def show_latency_analysis(df):
    """Show latency analysis"""
    st.subheader("⚡ Latency Analysis")
    
    if 'end_to_end_latency_ms' not in df.columns:
        st.error("No latency data found in the selected files.")
        return
    
    # Convert to numeric
    df['latency_numeric'] = pd.to_numeric(df['end_to_end_latency_ms'], errors='coerce')
    
    # Summary statistics
    st.write("**Latency Statistics:**")
    st.write(df['latency_numeric'].describe())
    
    # Simple histogram using Streamlit's built-in chart
    st.subheader("📊 Latency Distribution")
    st.histogram(df['latency_numeric'].dropna(), bins=20)

def show_quality_analysis(df):
    """Show quality analysis"""
    st.subheader("🎵 Quality Analysis")
    
    quality_columns = [col for col in df.columns if 'score' in col.lower()]
    
    if not quality_columns:
        st.warning("No quality metrics found in the data.")
        return
    
    st.write("**Available Quality Metrics:**")
    for col in quality_columns:
        if df[col].notna().any():
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            mean_val = numeric_data.mean()
            std_val = numeric_data.std()
            st.write(f"• {col}: {mean_val:.2f} ± {std_val:.2f}")

def show_configuration_analysis(df):
    """Show configuration impact analysis"""
    st.subheader("🔧 Configuration Impact Analysis")
    
    if 'config_mode' in df.columns:
        config_groups = df.groupby('config_mode')
        
        st.write("**Performance by Configuration:**")
        for config, group in config_groups:
            st.write(f"\n**{config.title()}:**")
            st.write(f"• Interactions: {len(group)}")
            
            if 'end_to_end_latency_ms' in group.columns:
                latency_data = pd.to_numeric(group['end_to_end_latency_ms'], errors='coerce')
                if not latency_data.empty:
                    st.write(f"• Average Latency: {latency_data.mean():.2f}ms")

def show_documentation():
    """Show documentation"""
    st.header("📚 Documentation")
    
    st.markdown("""
    ## 🎙️ Voice Agent Optimization Guide
    
    ### Configuration Options
    
    #### 🔄 Streaming STT
    - **Purpose**: Real-time speech-to-text processing
    - **Impact**: 20-40% latency reduction
    - **Trade-off**: Slightly higher complexity
    
    #### ⚡ Partial LLM Processing
    - **Purpose**: Start language processing before complete transcript
    - **Impact**: 200-500ms latency reduction
    - **Trade-off**: Requires confidence threshold tuning
    
    #### 🎯 Advanced Turn Detection
    - **Purpose**: Better conversation flow with MultilingualModel
    - **Impact**: 30% improvement in naturalness
    - **Trade-off**: Slightly higher resource usage
    
    #### 🎵 SSML Enhancement
    - **Purpose**: Enhanced speech synthesis quality
    - **Impact**: 25% improvement in speech quality
    - **Trade-off**: Minimal latency increase
    
    ### Parameter Guidelines
    
    #### Endpointing Delays
    - **Min Delay**: 0.2s (fast) to 2.0s (patient)
    - **Max Delay**: 3.0s (responsive) to 10.0s (very patient)
    - **Recommendation**: Start with 0.5s/6.0s for balanced performance
    
    #### TTS Speed
    - **Range**: 0.5x (very slow) to 2.0x (very fast)
    - **Healthcare**: 0.8-1.0x for clarity
    - **General**: 1.0-1.2x for efficiency
    
    #### Confidence Threshold
    - **60%**: Aggressive early processing
    - **70%**: Balanced (recommended)
    - **80%**: Conservative, higher accuracy
    
    ### Best Practices
    
    1. **Start Simple**: Begin with baseline configuration
    2. **Incremental Optimization**: Add one feature at a time
    3. **Measure Impact**: Test each change thoroughly
    4. **Consider Use Case**: Healthcare vs general conversation
    5. **Monitor Quality**: Balance speed with accuracy
    
    ### Troubleshooting
    
    - **High Latency**: Enable streaming STT and partial LLM
    - **Poor Quality**: Reduce TTS speed, enable SSML
    - **Interruptions**: Increase endpointing delays
    - **Missed Speech**: Decrease min endpointing delay
    """)

if __name__ == "__main__":
    main()
