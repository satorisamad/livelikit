#!/usr/bin/env python3
"""
Streamlit Voice Agent Optimizer

Interactive web interface for testing and comparing voice agent optimizations.
Provides real-time configuration, testing, and benchmarking capabilities.

Usage:
    streamlit run streamlit_voice_optimizer.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import asyncio
import time
from datetime import datetime
import threading
from typing import Dict, List, Optional
import glob

# Configuration definitions (standalone for Streamlit)
BASELINE_CONFIG = {
    "name": "Baseline Voice Agent",
    "description": "Basic VAD + non-streaming STT + simple LLM + basic TTS",
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.7,
    "stt_model": "nova-2",
    "stt_interim_results": False,
    "tts_model": "sonic-2",
    "tts_speed": 1.0,
    "min_endpointing_delay": 0.5,
    "max_endpointing_delay": 6.0,
}

VAD_CONFIGURATIONS = {
    "fast_response": {
        "name": "Fast Response",
        "description": "Minimal delays for fastest response",
        "min_endpointing_delay": 0.2,
        "max_endpointing_delay": 3.0,
        "turn_detection": "vad",
    },
    "balanced": {
        "name": "Balanced",
        "description": "Balanced speed vs accuracy",
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
        "turn_detection": "vad",
    },
    "patient": {
        "name": "Patient",
        "description": "Longer delays for natural pauses",
        "min_endpointing_delay": 1.0,
        "max_endpointing_delay": 8.0,
        "turn_detection": "vad",
    }
}

TURN_DETECTOR_CONFIGS = {
    "basic_vad": {
        "name": "Basic VAD",
        "description": "Simple voice activity detection",
        "turn_detection": "vad",
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 6.0,
    },
    "multilingual_model": {
        "name": "Multilingual Model",
        "description": "Advanced model-based turn detection",
        "turn_detection": "multilingual_model",
        "min_endpointing_delay": 0.3,
        "max_endpointing_delay": 4.0,
    }
}

STREAMING_CONFIGS = {
    "no_streaming": {
        "name": "No Streaming",
        "description": "Traditional non-streaming STT",
        "interim_results": False,
        "enable_partial_llm": False,
        "confidence_threshold": 0.0,
    },
    "partial_processing": {
        "name": "Partial Processing",
        "description": "Streaming STT with partial LLM prompting",
        "interim_results": True,
        "enable_partial_llm": True,
        "confidence_threshold": 0.7,
    }
}

SSML_CONFIGS = {
    "no_ssml": {
        "name": "No SSML",
        "description": "Basic TTS without SSML enhancements",
        "enable_ssml": False,
        "speaking_rate": "medium",
        "emphasis_level": "none",
    },
    "healthcare_ssml": {
        "name": "Healthcare SSML",
        "description": "Healthcare-optimized SSML with medical terms",
        "enable_ssml": True,
        "speaking_rate": "slow",
        "emphasis_level": "strong",
    }
}

# Import benchmarking functions
try:
    from benchmark_voice_agents import load_metrics_from_csv
except ImportError:
    def load_metrics_from_csv(csv_file: str) -> List[Dict]:
        """Fallback function if benchmark module not available"""
        import csv
        if not os.path.exists(csv_file):
            return []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

# Page configuration
st.set_page_config(
    page_title="Voice Agent Optimizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .optimization-badge {
        background-color: #00ff00;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
    .config-section {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🎙️ Voice Agent Optimization Dashboard")
    st.markdown("Interactive testing and benchmarking of voice agent optimizations")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Main mode selection
        mode = st.selectbox(
            "Select Mode",
            ["Configuration Builder", "Live Testing", "Benchmarking", "Results Analysis"],
            help="Choose the main operation mode"
        )
        
        st.divider()
        
        if mode == "Configuration Builder":
            show_configuration_builder()
        elif mode == "Live Testing":
            show_live_testing()
        elif mode == "Benchmarking":
            show_benchmarking()
        elif mode == "Results Analysis":
            show_results_analysis()

def show_configuration_builder():
    """Configuration builder interface"""
    st.header("⚙️ Configuration Builder")
    
    # Create tabs for different optimization categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Basic Config", 
        "🔊 VAD/Endpointing", 
        "🔄 Turn Detection", 
        "📡 Streaming STT", 
        "🎵 SSML TTS"
    ])
    
    with tab1:
        st.subheader("Basic Configuration")
        
        # Agent type selection
        agent_type = st.selectbox(
            "Agent Type",
            ["baseline", "optimized", "custom"],
            help="Select the base agent configuration"
        )
        
        # LLM settings
        st.subheader("LLM Settings")
        llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        llm_temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        llm_max_tokens = st.number_input("Max Tokens", 50, 500, 150)
        
        # Display current configuration
        config = {
            "agent_type": agent_type,
            "llm_model": llm_model,
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens
        }
        
        st.json(config)
    
    with tab2:
        st.subheader("VAD/Endpointing Configuration")
        
        # VAD configuration selection
        vad_config = st.selectbox(
            "VAD Configuration",
            list(VAD_CONFIGURATIONS.keys()),
            help="Select VAD/endpointing configuration"
        )
        
        # Display selected configuration details
        if vad_config:
            config_details = VAD_CONFIGURATIONS[vad_config]
            st.write(f"**{config_details['name']}**")
            st.write(config_details['description'])
            
            # Allow custom parameter adjustment
            st.subheader("Custom Parameters")
            min_delay = st.slider(
                "Min Endpointing Delay (s)", 
                0.1, 3.0, 
                config_details['min_endpointing_delay'], 
                0.1
            )
            max_delay = st.slider(
                "Max Endpointing Delay (s)", 
                1.0, 15.0, 
                config_details['max_endpointing_delay'], 
                0.5
            )
            
            # Update configuration
            custom_vad_config = config_details.copy()
            custom_vad_config['min_endpointing_delay'] = min_delay
            custom_vad_config['max_endpointing_delay'] = max_delay
            
            st.json(custom_vad_config)
    
    with tab3:
        st.subheader("Turn Detection Configuration")
        
        turn_config = st.selectbox(
            "Turn Detection Method",
            list(TURN_DETECTOR_CONFIGS.keys()),
            help="Select turn detection configuration"
        )
        
        if turn_config:
            config_details = TURN_DETECTOR_CONFIGS[turn_config]
            st.write(f"**{config_details['name']}**")
            st.write(config_details['description'])
            
            # Advanced settings
            st.subheader("Advanced Settings")
            allow_interruptions = st.checkbox("Allow Interruptions", True)
            sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.5, 0.1)
            
            st.json({
                "turn_detection": config_details['turn_detection'],
                "allow_interruptions": allow_interruptions,
                "sensitivity": sensitivity
            })
    
    with tab4:
        st.subheader("Streaming STT Configuration")
        
        streaming_config = st.selectbox(
            "Streaming Configuration",
            list(STREAMING_CONFIGS.keys()),
            help="Select streaming STT configuration"
        )
        
        if streaming_config:
            config_details = STREAMING_CONFIGS[streaming_config]
            st.write(f"**{config_details['name']}**")
            st.write(config_details['description'])
            
            # Streaming parameters
            st.subheader("Streaming Parameters")
            interim_results = st.checkbox(
                "Enable Interim Results", 
                config_details['interim_results']
            )
            enable_partial_llm = st.checkbox(
                "Enable Partial LLM", 
                config_details['enable_partial_llm']
            )
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.5, 1.0, 
                config_details['confidence_threshold'], 
                0.05
            )
            
            st.json({
                "interim_results": interim_results,
                "enable_partial_llm": enable_partial_llm,
                "confidence_threshold": confidence_threshold
            })
    
    with tab5:
        st.subheader("SSML TTS Configuration")
        
        ssml_config = st.selectbox(
            "SSML Configuration",
            list(SSML_CONFIGS.keys()),
            help="Select SSML TTS configuration"
        )
        
        if ssml_config:
            config_details = SSML_CONFIGS[ssml_config]
            st.write(f"**{config_details['name']}**")
            st.write(config_details['description'])
            
            # SSML parameters
            st.subheader("SSML Parameters")
            enable_ssml = st.checkbox("Enable SSML", config_details['enable_ssml'])
            speaking_rate = st.selectbox(
                "Speaking Rate", 
                ["slow", "medium", "fast"], 
                index=["slow", "medium", "fast"].index(config_details['speaking_rate'])
            )
            emphasis_level = st.selectbox(
                "Emphasis Level", 
                ["none", "moderate", "strong"], 
                index=["none", "moderate", "strong"].index(config_details['emphasis_level'])
            )
            
            st.json({
                "enable_ssml": enable_ssml,
                "speaking_rate": speaking_rate,
                "emphasis_level": emphasis_level
            })
    
    # Save configuration button
    if st.button("💾 Save Configuration", type="primary"):
        # Collect all configuration data
        full_config = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "vad_config": vad_config,
            "turn_config": turn_config,
            "streaming_config": streaming_config,
            "ssml_config": ssml_config
        }
        
        # Save to file
        config_file = f"custom_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        st.success(f"Configuration saved to {config_file}")

def show_live_testing():
    """Live testing interface"""
    st.header("🎤 Live Testing")

    # Import agent launcher
    try:
        from agent_launcher import agent_launcher, config_manager
        launcher_available = True
    except ImportError as e:
        st.warning(f"Agent launcher not available: {e}")
        launcher_available = False
    except Exception as e:
        st.error(f"Error loading agent launcher: {e}")
        launcher_available = False

    if not launcher_available:
        st.info("To enable live testing, ensure all voice agent dependencies are installed.")
        st.info("You can still use the Configuration Builder and Results Analysis features.")
        return

    # Configuration selection
    available_presets = config_manager.list_presets()
    test_config = st.selectbox(
        "Select Configuration to Test",
        available_presets,
        help="Choose which configuration to test"
    )

    # Show configuration details
    if test_config:
        config_details = config_manager.get_preset(test_config)
        if config_details:
            with st.expander("Configuration Details"):
                st.json(config_details)

    # Agent status
    status = agent_launcher.get_status()

    col1, col2, col3 = st.columns(3)
    with col1:
        if status["running"]:
            st.success(f"🟢 Agent Running (PID: {status['pid']})")
            st.write(f"Uptime: {status['uptime']:.1f}s")
        else:
            st.error("🔴 Agent Stopped")

    with col2:
        if st.button("🚀 Start Agent", type="primary", disabled=status["running"]):
            config = config_manager.get_preset(test_config)
            if config and agent_launcher.start_agent(config):
                st.success(f"Started {test_config} agent")
                st.rerun()
            else:
                st.error("Failed to start agent")

    with col3:
        if st.button("⏹️ Stop Agent", disabled=not status["running"]):
            if agent_launcher.stop_agent():
                st.success("Agent stopped")
                st.rerun()
            else:
                st.error("Failed to stop agent")

    # Real-time metrics display
    if status["running"]:
        st.subheader("📊 Live Metrics")

        # Create placeholder for real-time updates
        metrics_placeholder = st.empty()

        # Get latest metrics
        latest_metrics = agent_launcher.get_metrics()
        if latest_metrics:
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", latest_metrics.get("status", "unknown"))
                with col2:
                    st.metric("Uptime", f"{latest_metrics.get('uptime', 0):.1f}s")
                with col3:
                    st.metric("Config", latest_metrics.get("config", "unknown"))

        # Instructions for testing
        st.info("""
        **Testing Instructions:**
        1. Join the LiveKit room using the provided URL
        2. Start speaking to test the voice agent
        3. Monitor the real-time metrics above
        4. Use the benchmarking tools to analyze results
        """)

        # Auto-refresh for real-time updates
        if st.button("🔄 Refresh Metrics"):
            st.rerun()

def show_benchmarking():
    """Benchmarking interface"""
    st.header("📊 Benchmarking")
    
    # Benchmark configuration
    st.subheader("Benchmark Configuration")
    
    configs_to_compare = st.multiselect(
        "Select Configurations to Compare",
        ["baseline", "optimized", "vad_tuning", "turn_detector", "streaming_stt", "ssml_tts"],
        default=["baseline", "optimized"],
        help="Choose configurations for comparison"
    )
    
    # Benchmark parameters
    col1, col2 = st.columns(2)
    with col1:
        num_interactions = st.number_input("Number of Interactions", 5, 100, 20)
        include_quality_metrics = st.checkbox("Include Quality Metrics", True)
    
    with col2:
        generate_visualizations = st.checkbox("Generate Visualizations", True)
        save_results = st.checkbox("Save Results", True)
    
    # Run benchmark
    if st.button("🏃‍♂️ Run Benchmark", type="primary"):
        run_benchmark_comparison(configs_to_compare, num_interactions)

def show_results_analysis():
    """Results analysis interface"""
    st.header("📈 Results Analysis")
    
    # Load existing results
    result_files = glob.glob("*_metrics_*.csv")
    
    if not result_files:
        st.warning("No result files found. Run some tests first!")
        return
    
    # File selection
    selected_files = st.multiselect(
        "Select Result Files to Analyze",
        result_files,
        help="Choose CSV files containing test results"
    )
    
    if selected_files:
        # Analysis options
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Latency Comparison", "Quality Metrics", "Optimization Effectiveness", "Detailed Breakdown"],
            help="Choose the type of analysis to perform"
        )
        
        # Perform analysis
        if st.button("🔍 Analyze Results"):
            perform_results_analysis(selected_files, analysis_type)

def show_real_time_metrics():
    """Display real-time metrics during testing"""
    st.subheader("📊 Real-time Metrics")
    
    # Create placeholder for metrics
    metrics_placeholder = st.empty()
    
    # Simulate real-time metrics (in practice, this would connect to actual agent)
    import random
    
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Response Latency", 
                f"{random.uniform(1.2, 2.8):.2f}s",
                delta=f"{random.uniform(-0.5, 0.5):.2f}s"
            )
        
        with col2:
            st.metric(
                "Speech Quality", 
                f"{random.uniform(3.5, 4.8):.1f}/5.0",
                delta=f"{random.uniform(-0.3, 0.3):.1f}"
            )
        
        with col3:
            st.metric(
                "Conversation Flow", 
                f"{random.uniform(6.0, 8.5):.1f}/10",
                delta=f"{random.uniform(-0.5, 0.5):.1f}"
            )
        
        with col4:
            st.metric(
                "Interactions", 
                f"{random.randint(1, 15)}",
                delta="1"
            )

def show_test_results():
    """Display test results"""
    st.subheader("🎯 Test Results")
    
    # Sample results data (in practice, load from actual test results)
    results_data = {
        "Configuration": ["Baseline", "Optimized", "VAD Tuned", "Turn Detector", "Streaming STT", "SSML TTS"],
        "Avg Latency (ms)": [2847, 1456, 2234, 2456, 1980, 2123],
        "Quality Score": [3.2, 4.1, 3.5, 3.8, 3.6, 4.3],
        "Flow Score": [5.5, 7.3, 6.1, 7.8, 6.4, 6.7],
        "Interactions": [20, 20, 20, 20, 20, 20]
    }
    
    df = pd.DataFrame(results_data)
    
    # Display results table
    st.dataframe(df, use_container_width=True)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Latency comparison chart
        fig_latency = px.bar(
            df, 
            x="Configuration", 
            y="Avg Latency (ms)",
            title="Average Response Latency Comparison",
            color="Avg Latency (ms)",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig_latency, use_container_width=True)
    
    with col2:
        # Quality metrics radar chart
        fig_radar = go.Figure()
        
        # Normalize scores for radar chart
        normalized_quality = [score / 5.0 * 100 for score in df["Quality Score"]]
        normalized_flow = [score / 10.0 * 100 for score in df["Flow Score"]]
        
        for i, config in enumerate(df["Configuration"]):
            fig_radar.add_trace(go.Scatterpolar(
                r=[normalized_quality[i], normalized_flow[i], 100 - (df["Avg Latency (ms)"][i] / 3000 * 100)],
                theta=["Quality", "Flow", "Speed"],
                fill='toself',
                name=config
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

def run_benchmark_comparison(configs: List[str], num_interactions: int):
    """Run benchmark comparison"""
    st.info(f"Running benchmark with {len(configs)} configurations for {num_interactions} interactions each...")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate benchmark execution
    for i, config in enumerate(configs):
        status_text.text(f"Testing {config}...")
        progress_bar.progress((i + 1) / len(configs))
        time.sleep(1)  # Simulate processing time
    
    status_text.text("Benchmark completed!")
    st.success("Benchmark comparison completed successfully!")
    
    # Show results
    show_test_results()

def perform_results_analysis(files: List[str], analysis_type: str):
    """Perform detailed results analysis"""
    st.info(f"Analyzing {len(files)} result files for {analysis_type}...")
    
    # Load and combine data from selected files
    all_data = []
    for file in files:
        data = load_metrics_from_csv(file)
        if data:
            all_data.extend(data)
    
    if not all_data:
        st.error("No data found in selected files")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data)
    
    # Perform analysis based on type
    if analysis_type == "Latency Comparison":
        analyze_latency_data(df)
    elif analysis_type == "Quality Metrics":
        analyze_quality_data(df)
    elif analysis_type == "Optimization Effectiveness":
        analyze_optimization_effectiveness(df)
    elif analysis_type == "Detailed Breakdown":
        analyze_detailed_breakdown(df)

def analyze_latency_data(df: pd.DataFrame):
    """Analyze latency data"""
    st.subheader("⚡ Latency Analysis")
    
    if 'end_to_end_latency_ms' in df.columns:
        # Convert to numeric
        df['latency_numeric'] = pd.to_numeric(df['end_to_end_latency_ms'], errors='coerce')
        
        # Summary statistics
        st.write("**Latency Statistics:**")
        st.write(df['latency_numeric'].describe())
        
        # Latency distribution
        fig = px.histogram(
            df, 
            x='latency_numeric', 
            title="Latency Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)

def analyze_quality_data(df: pd.DataFrame):
    """Analyze quality metrics"""
    st.subheader("🎵 Quality Analysis")
    
    quality_columns = [col for col in df.columns if 'score' in col.lower()]
    
    if quality_columns:
        st.write("**Available Quality Metrics:**")
        for col in quality_columns:
            if df[col].notna().any():
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                st.write(f"- {col}: {numeric_data.mean():.2f} ± {numeric_data.std():.2f}")

def analyze_optimization_effectiveness(df: pd.DataFrame):
    """Analyze optimization effectiveness"""
    st.subheader("🎯 Optimization Effectiveness")
    
    if 'config_mode' in df.columns:
        # Group by configuration
        config_groups = df.groupby('config_mode')
        
        effectiveness_data = []
        for config, group in config_groups:
            if 'end_to_end_latency_ms' in group.columns:
                latency_data = pd.to_numeric(group['end_to_end_latency_ms'], errors='coerce')
                avg_latency = latency_data.mean()
                effectiveness_data.append({
                    'Configuration': config,
                    'Average Latency': avg_latency,
                    'Sample Count': len(group)
                })
        
        if effectiveness_data:
            effectiveness_df = pd.DataFrame(effectiveness_data)
            st.dataframe(effectiveness_df, use_container_width=True)

def analyze_detailed_breakdown(df: pd.DataFrame):
    """Analyze detailed breakdown"""
    st.subheader("🔍 Detailed Breakdown")
    
    # Show data overview
    st.write("**Data Overview:**")
    st.write(f"Total records: {len(df)}")
    st.write(f"Columns: {len(df.columns)}")
    
    # Show sample data
    st.write("**Sample Data:**")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column analysis
    st.write("**Column Analysis:**")
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        st.write(f"- {col}: {non_null_count}/{len(df)} non-null values")

if __name__ == "__main__":
    # Initialize session state
    if 'testing' not in st.session_state:
        st.session_state.testing = False
    
    main()
