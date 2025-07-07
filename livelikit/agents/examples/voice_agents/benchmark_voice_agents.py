#!/usr/bin/env python3
"""
Voice Agent Benchmarking Script

This script compares different voice agent configurations:
1. Baseline (no optimizations)
2. Optimized (all features enabled)
3. Human-like (slow, natural timing)

Usage:
    python benchmark_voice_agents.py --mode baseline
    python benchmark_voice_agents.py --mode optimized
    python benchmark_voice_agents.py --mode human_like
    python benchmark_voice_agents.py --compare  # Generate comparison report
"""

import argparse
import csv
import json
import os
import statistics
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

def load_metrics_from_csv(csv_file: str) -> List[Dict]:
    """Load metrics data from CSV file"""
    if not os.path.exists(csv_file):
        return []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def calculate_statistics(values: List[float]) -> Dict:
    """Calculate statistical measures for a list of values"""
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}
    
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0
    }

def analyze_metrics_file(csv_file: str, config_mode: str) -> Dict:
    """Analyze metrics from a single configuration"""
    data = load_metrics_from_csv(csv_file)
    
    if not data:
        return {"error": f"No data found in {csv_file}"}
    
    # Extract numeric values
    latencies = []
    response_lengths = []
    processing_times = {"mic_to_transcript": [], "llm_processing": [], "tts_synthesis": []}
    
    for row in data:
        try:
            if row.get("end_to_end_latency_ms"):
                latencies.append(float(row["end_to_end_latency_ms"]))
            if row.get("response_length_words"):
                response_lengths.append(int(row["response_length_words"]))
            
            # Processing time components
            for key in processing_times:
                if row.get(f"{key}_ms"):
                    processing_times[key].append(float(row[f"{key}_ms"]))
        except (ValueError, TypeError):
            continue
    
    analysis = {
        "config_mode": config_mode,
        "total_interactions": len(data),
        "timestamp": datetime.now().isoformat(),
        
        "latency_stats": calculate_statistics(latencies),
        "response_length_stats": calculate_statistics(response_lengths),
        
        "processing_breakdown": {
            key: calculate_statistics(values) 
            for key, values in processing_times.items()
        },
        
        "sample_interactions": data[:5] if len(data) >= 5 else data,
    }
    
    return analysis

def generate_comparison_report(baseline_file: str, optimized_file: str, output_dir: str = "."):
    """Generate a comprehensive comparison report"""
    
    # Analyze each configuration
    baseline_analysis = analyze_metrics_file(baseline_file, "baseline")
    optimized_analysis = analyze_metrics_file(optimized_file, "optimized")
    
    if "error" in baseline_analysis or "error" in optimized_analysis:
        print("Error: Could not load metrics files for comparison")
        return
    
    # Create comparison report
    report = {
        "comparison_timestamp": datetime.now().isoformat(),
        "configurations": {
            "baseline": baseline_analysis,
            "optimized": optimized_analysis
        },
        
        "performance_comparison": {
            "latency_improvement": {
                "baseline_mean_ms": baseline_analysis["latency_stats"]["mean"],
                "optimized_mean_ms": optimized_analysis["latency_stats"]["mean"],
                "improvement_ms": baseline_analysis["latency_stats"]["mean"] - optimized_analysis["latency_stats"]["mean"],
                "improvement_percent": ((baseline_analysis["latency_stats"]["mean"] - optimized_analysis["latency_stats"]["mean"]) / baseline_analysis["latency_stats"]["mean"]) * 100 if baseline_analysis["latency_stats"]["mean"] > 0 else 0
            },
            
            "response_quality": {
                "baseline_avg_words": baseline_analysis["response_length_stats"]["mean"],
                "optimized_avg_words": optimized_analysis["response_length_stats"]["mean"],
            }
        }
    }
    
    # Save detailed report
    report_file = os.path.join(output_dir, f"voice_agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary table
    print("\n" + "="*80)
    print("VOICE AGENT PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\nBaseline Configuration:")
    print(f"  Total Interactions: {baseline_analysis['total_interactions']}")
    print(f"  Average Latency: {baseline_analysis['latency_stats']['mean']:.2f} ms")
    print(f"  Latency Range: {baseline_analysis['latency_stats']['min']:.2f} - {baseline_analysis['latency_stats']['max']:.2f} ms")
    print(f"  Average Response Length: {baseline_analysis['response_length_stats']['mean']:.1f} words")
    
    print(f"\nOptimized Configuration:")
    print(f"  Total Interactions: {optimized_analysis['total_interactions']}")
    print(f"  Average Latency: {optimized_analysis['latency_stats']['mean']:.2f} ms")
    print(f"  Latency Range: {optimized_analysis['latency_stats']['min']:.2f} - {optimized_analysis['latency_stats']['max']:.2f} ms")
    print(f"  Average Response Length: {optimized_analysis['response_length_stats']['mean']:.1f} words")
    
    improvement = report["performance_comparison"]["latency_improvement"]
    print(f"\nPerformance Improvement:")
    print(f"  Latency Reduction: {improvement['improvement_ms']:.2f} ms ({improvement['improvement_percent']:.1f}%)")
    
    print(f"\nDetailed report saved to: {report_file}")
    print("="*80)

def create_visualization(baseline_file: str, optimized_file: str, output_dir: str = "."):
    """Create visualization comparing the configurations"""
    try:
        import matplotlib.pyplot as plt
        
        baseline_data = load_metrics_from_csv(baseline_file)
        optimized_data = load_metrics_from_csv(optimized_file)
        
        if not baseline_data or not optimized_data:
            print("Warning: Insufficient data for visualization")
            return
        
        # Extract latency data
        baseline_latencies = [float(row["end_to_end_latency_ms"]) for row in baseline_data if row.get("end_to_end_latency_ms")]
        optimized_latencies = [float(row["end_to_end_latency_ms"]) for row in optimized_data if row.get("end_to_end_latency_ms")]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Latency comparison
        ax1.boxplot([baseline_latencies, optimized_latencies], labels=['Baseline', 'Optimized'])
        ax1.set_title('End-to-End Latency Comparison')
        ax1.set_ylabel('Latency (ms)')
        
        # Latency over time
        ax2.plot(range(len(baseline_latencies)), baseline_latencies, 'b-', alpha=0.7, label='Baseline')
        ax2.plot(range(len(optimized_latencies)), optimized_latencies, 'r-', alpha=0.7, label='Optimized')
        ax2.set_title('Latency Over Time')
        ax2.set_xlabel('Interaction Number')
        ax2.set_ylabel('Latency (ms)')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, f"voice_agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_file}")
        
    except ImportError:
        print("Warning: matplotlib not available for visualization")

def main():
    parser = argparse.ArgumentParser(description="Voice Agent Benchmarking Tool")
    parser.add_argument("--mode", choices=["baseline", "optimized", "human_like"], 
                       help="Run specific agent mode")
    parser.add_argument("--compare", action="store_true", 
                       help="Generate comparison report from existing metrics")
    parser.add_argument("--baseline-file", default="baseline_metrics_*.csv",
                       help="Baseline metrics CSV file pattern")
    parser.add_argument("--optimized-file", default="optimized_metrics_*.csv",
                       help="Optimized metrics CSV file pattern")
    parser.add_argument("--output-dir", default=".", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    if args.compare:
        # Find the most recent metrics files
        import glob
        
        baseline_files = glob.glob(args.baseline_file)
        optimized_files = glob.glob(args.optimized_file)
        
        if not baseline_files or not optimized_files:
            print("Error: Could not find metrics files for comparison")
            print(f"Looking for baseline: {args.baseline_file}")
            print(f"Looking for optimized: {args.optimized_file}")
            return
        
        # Use the most recent files
        baseline_file = max(baseline_files, key=os.path.getctime)
        optimized_file = max(optimized_files, key=os.path.getctime)
        
        print(f"Comparing:")
        print(f"  Baseline: {baseline_file}")
        print(f"  Optimized: {optimized_file}")
        
        generate_comparison_report(baseline_file, optimized_file, args.output_dir)
        create_visualization(baseline_file, optimized_file, args.output_dir)
        
    elif args.mode:
        print(f"To run {args.mode} mode, use:")
        if args.mode == "baseline":
            print("python baseline_voice_agent.py")
        elif args.mode == "optimized":
            print("python optimized_voice_agent.py")
        elif args.mode == "human_like":
            print("python voice_agent.py  # (with CURRENT_MODE = 'human_like')")
    else:
        parser.print_help()

def analyze_vad_tuning_results(results_dir: str = "."):
    """Analyze VAD tuning results from multiple configurations"""
    import glob

    # Find all VAD tuning result files
    vad_files = glob.glob(os.path.join(results_dir, "vad_tuning_*.csv"))

    if not vad_files:
        print("No VAD tuning results found")
        return

    print("\n" + "="*80)
    print("VAD/ENDPOINTING PARAMETER TUNING ANALYSIS")
    print("="*80)

    results = {}

    for file_path in vad_files:
        # Extract config name from filename
        filename = os.path.basename(file_path)
        config_name = filename.split('_')[2]  # vad_tuning_CONFIG_timestamp.csv

        data = load_metrics_from_csv(file_path)
        if not data:
            continue

        # Calculate VAD-specific metrics
        total_interactions = len(data)
        false_triggers = sum(1 for row in data if row.get('false_trigger') == 'True')
        missed_endpoints = sum(1 for row in data if row.get('missed_endpoint') == 'True')
        natural_pauses = sum(1 for row in data if row.get('natural_pause_detected') == 'True')

        latencies = [float(row['end_to_end_latency_ms']) for row in data if row.get('end_to_end_latency_ms')]

        results[config_name] = {
            'total_interactions': total_interactions,
            'false_triggers': false_triggers,
            'missed_endpoints': missed_endpoints,
            'natural_pauses': natural_pauses,
            'false_trigger_rate': false_triggers / max(total_interactions, 1),
            'missed_endpoint_rate': missed_endpoints / max(total_interactions, 1),
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'min_delay': data[0].get('min_endpointing_delay', 'N/A') if data else 'N/A',
            'max_delay': data[0].get('max_endpointing_delay', 'N/A') if data else 'N/A',
        }

    # Display results
    for config_name, metrics in results.items():
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Endpointing Delays: {metrics['min_delay']}s - {metrics['max_delay']}s")
        print(f"  Total Interactions: {metrics['total_interactions']}")
        print(f"  Average Latency: {metrics['avg_latency']:.2f} ms")
        print(f"  False Triggers: {metrics['false_triggers']} ({metrics['false_trigger_rate']:.1%})")
        print(f"  Missed Endpoints: {metrics['missed_endpoints']} ({metrics['missed_endpoint_rate']:.1%})")
        print(f"  Natural Pauses Detected: {metrics['natural_pauses']}")

    # Find optimal configuration
    if results:
        # Score based on low false triggers, low missed endpoints, and reasonable latency
        scored_configs = []
        for config_name, metrics in results.items():
            # Lower is better for all these metrics
            score = (metrics['false_trigger_rate'] * 100 +
                    metrics['missed_endpoint_rate'] * 100 +
                    metrics['avg_latency'] / 100)
            scored_configs.append((config_name, score, metrics))

        scored_configs.sort(key=lambda x: x[1])  # Sort by score (lower is better)

        print(f"\n🏆 RECOMMENDED CONFIGURATION: {scored_configs[0][0].upper()}")
        best_metrics = scored_configs[0][2]
        print(f"   Balanced performance with {best_metrics['avg_latency']:.0f}ms latency")
        print(f"   {best_metrics['false_trigger_rate']:.1%} false triggers, {best_metrics['missed_endpoint_rate']:.1%} missed endpoints")

    print("="*80)

def analyze_turn_detector_results(results_dir: str = "."):
    """Analyze turn detector comparison results"""
    import glob

    # Find all turn detector result files
    turn_files = glob.glob(os.path.join(results_dir, "turn_detector_*.csv"))

    if not turn_files:
        print("No turn detector results found")
        return

    print("\n" + "="*80)
    print("TURN DETECTOR COMPARISON ANALYSIS")
    print("="*80)

    results = {}

    for file_path in turn_files:
        # Extract config name from filename
        filename = os.path.basename(file_path)
        config_name = filename.split('_')[2]  # turn_detector_CONFIG_timestamp.csv

        data = load_metrics_from_csv(file_path)
        if not data:
            continue

        # Calculate turn detector specific metrics
        total_interactions = len(data)
        interruptions = sum(1 for row in data if row.get('interruption_detected') == 'True')
        natural_pauses = sum(1 for row in data if row.get('natural_pause_respected') == 'True')
        premature_interruptions = sum(1 for row in data if row.get('premature_interruption') == 'True')

        # Calculate average conversation flow score
        flow_scores = [float(row['conversation_flow_score']) for row in data if row.get('conversation_flow_score')]
        avg_flow_score = statistics.mean(flow_scores) if flow_scores else 0

        latencies = [float(row['end_to_end_latency_ms']) for row in data if row.get('end_to_end_latency_ms')]

        results[config_name] = {
            'total_interactions': total_interactions,
            'interruptions': interruptions,
            'natural_pauses': natural_pauses,
            'premature_interruptions': premature_interruptions,
            'avg_flow_score': avg_flow_score,
            'interruption_rate': interruptions / max(total_interactions, 1),
            'natural_pause_rate': natural_pauses / max(total_interactions, 1),
            'premature_interruption_rate': premature_interruptions / max(total_interactions, 1),
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'turn_detection_method': data[0].get('turn_detection_method', 'N/A') if data else 'N/A',
        }

    # Display results
    for config_name, metrics in results.items():
        print(f"\n{config_name.upper().replace('_', ' ')} Configuration:")
        print(f"  Turn Detection Method: {metrics['turn_detection_method']}")
        print(f"  Total Interactions: {metrics['total_interactions']}")
        print(f"  Average Latency: {metrics['avg_latency']:.2f} ms")
        print(f"  Average Flow Score: {metrics['avg_flow_score']:.1f}/10")
        print(f"  Interruptions: {metrics['interruptions']} ({metrics['interruption_rate']:.1%})")
        print(f"  Natural Pauses Respected: {metrics['natural_pauses']} ({metrics['natural_pause_rate']:.1%})")
        print(f"  Premature Interruptions: {metrics['premature_interruptions']} ({metrics['premature_interruption_rate']:.1%})")

    # Find best configuration for conversation flow
    if results:
        # Score based on high flow score, low premature interruptions, reasonable latency
        scored_configs = []
        for config_name, metrics in results.items():
            # Higher flow score is better, lower premature interruptions is better
            score = (metrics['avg_flow_score'] * 10 -
                    metrics['premature_interruption_rate'] * 100 -
                    metrics['avg_latency'] / 100)
            scored_configs.append((config_name, score, metrics))

        scored_configs.sort(key=lambda x: x[1], reverse=True)  # Sort by score (higher is better)

        print(f"\n🏆 BEST CONVERSATION FLOW: {scored_configs[0][0].upper().replace('_', ' ')}")
        best_metrics = scored_configs[0][2]
        print(f"   Flow score: {best_metrics['avg_flow_score']:.1f}/10")
        print(f"   {best_metrics['premature_interruption_rate']:.1%} premature interruptions")
        print(f"   {best_metrics['natural_pause_rate']:.1%} natural pauses respected")

    print("="*80)

def analyze_streaming_stt_results(results_dir: str = "."):
    """Analyze streaming STT and partial hypotheses results"""
    import glob

    # Find all streaming STT result files
    streaming_files = glob.glob(os.path.join(results_dir, "streaming_stt_*.csv"))

    if not streaming_files:
        print("No streaming STT results found")
        return

    print("\n" + "="*80)
    print("STREAMING STT & PARTIAL HYPOTHESES ANALYSIS")
    print("="*80)

    results = {}

    for file_path in streaming_files:
        # Extract config name from filename
        filename = os.path.basename(file_path)
        config_name = filename.split('_')[2]  # streaming_stt_CONFIG_timestamp.csv

        data = load_metrics_from_csv(file_path)
        if not data:
            continue

        # Calculate streaming-specific metrics
        total_interactions = len(data)
        partial_llm_triggers = sum(1 for row in data if row.get('partial_llm_triggered') == 'True')

        # Calculate averages
        early_gains = [float(row['early_processing_gain_ms']) for row in data if row.get('early_processing_gain_ms')]
        accuracy_scores = [float(row['transcript_accuracy_score']) for row in data if row.get('transcript_accuracy_score')]
        partial_counts = [int(row['partial_transcripts_count']) for row in data if row.get('partial_transcripts_count')]
        latencies = [float(row['end_to_end_latency_ms']) for row in data if row.get('end_to_end_latency_ms')]

        results[config_name] = {
            'total_interactions': total_interactions,
            'partial_llm_triggers': partial_llm_triggers,
            'avg_early_gain_ms': statistics.mean(early_gains) if early_gains else 0,
            'avg_accuracy_score': statistics.mean(accuracy_scores) if accuracy_scores else 0,
            'avg_partial_count': statistics.mean(partial_counts) if partial_counts else 0,
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'partial_llm_trigger_rate': partial_llm_triggers / max(total_interactions, 1),
            'interim_results': data[0].get('interim_results_enabled', 'N/A') if data else 'N/A',
            'enable_partial_llm': data[0].get('enable_partial_llm', 'N/A') if data else 'N/A',
            'confidence_threshold': data[0].get('confidence_threshold', 'N/A') if data else 'N/A',
        }

    # Display results
    for config_name, metrics in results.items():
        print(f"\n{config_name.upper().replace('_', ' ')} Configuration:")
        print(f"  Interim Results: {metrics['interim_results']}")
        print(f"  Partial LLM: {metrics['enable_partial_llm']}")
        print(f"  Confidence Threshold: {metrics['confidence_threshold']}")
        print(f"  Total Interactions: {metrics['total_interactions']}")
        print(f"  Average Latency: {metrics['avg_latency']:.2f} ms")
        print(f"  Average Accuracy Score: {metrics['avg_accuracy_score']:.1f}/10")
        print(f"  Average Partial Transcripts: {metrics['avg_partial_count']:.1f}")
        print(f"  Partial LLM Triggers: {metrics['partial_llm_triggers']} ({metrics['partial_llm_trigger_rate']:.1%})")
        print(f"  Average Early Processing Gain: {metrics['avg_early_gain_ms']:.2f} ms")

    # Find best configuration for streaming performance
    if results:
        # Score based on latency reduction, accuracy, and early processing gains
        scored_configs = []
        for config_name, metrics in results.items():
            # Higher early gain and accuracy is better, lower latency is better
            score = (metrics['avg_early_gain_ms'] +
                    metrics['avg_accuracy_score'] * 10 -
                    metrics['avg_latency'] / 100)
            scored_configs.append((config_name, score, metrics))

        scored_configs.sort(key=lambda x: x[1], reverse=True)  # Sort by score (higher is better)

        print(f"\n🏆 BEST STREAMING PERFORMANCE: {scored_configs[0][0].upper().replace('_', ' ')}")
        best_metrics = scored_configs[0][2]
        print(f"   Average latency: {best_metrics['avg_latency']:.2f} ms")
        print(f"   Early processing gain: {best_metrics['avg_early_gain_ms']:.2f} ms")
        print(f"   Accuracy score: {best_metrics['avg_accuracy_score']:.1f}/10")
        print(f"   Partial LLM trigger rate: {best_metrics['partial_llm_trigger_rate']:.1%}")

    print("="*80)

def analyze_ssml_tts_results(results_dir: str = "."):
    """Analyze SSML TTS enhancement results"""
    import glob

    # Find all SSML TTS result files
    ssml_files = glob.glob(os.path.join(results_dir, "ssml_tts_*.csv"))

    if not ssml_files:
        print("No SSML TTS results found")
        return

    print("\n" + "="*80)
    print("SSML TTS ENHANCEMENT ANALYSIS")
    print("="*80)

    results = {}

    for file_path in ssml_files:
        # Extract config name from filename
        filename = os.path.basename(file_path)
        config_name = filename.split('_')[2]  # ssml_tts_CONFIG_timestamp.csv

        data = load_metrics_from_csv(file_path)
        if not data:
            continue

        # Calculate SSML-specific metrics
        total_interactions = len(data)

        # Calculate averages for quality scores
        mos_scores = [float(row['estimated_mos_score']) for row in data if row.get('estimated_mos_score')]
        naturalness_scores = [float(row['naturalness_score']) for row in data if row.get('naturalness_score')]
        clarity_scores = [float(row['clarity_score']) for row in data if row.get('clarity_score')]

        # Calculate SSML usage statistics
        enhancement_counts = [int(row['ssml_enhancements_count']) for row in data if row.get('ssml_enhancements_count')]
        emphasis_counts = [int(row['emphasis_tags']) for row in data if row.get('emphasis_tags')]
        prosody_counts = [int(row['prosody_tags']) for row in data if row.get('prosody_tags')]
        break_counts = [int(row['break_tags']) for row in data if row.get('break_tags')]

        # Audio quality metrics
        audio_durations = [float(row['audio_duration_ms']) for row in data if row.get('audio_duration_ms')]
        file_sizes = [int(row['file_size_bytes']) for row in data if row.get('file_size_bytes')]
        synthesis_times = [float(row['tts_synthesis_ms']) for row in data if row.get('tts_synthesis_ms')]
        latencies = [float(row['end_to_end_latency_ms']) for row in data if row.get('end_to_end_latency_ms')]

        results[config_name] = {
            'total_interactions': total_interactions,
            'avg_mos_score': statistics.mean(mos_scores) if mos_scores else 0,
            'avg_naturalness_score': statistics.mean(naturalness_scores) if naturalness_scores else 0,
            'avg_clarity_score': statistics.mean(clarity_scores) if clarity_scores else 0,
            'avg_enhancement_count': statistics.mean(enhancement_counts) if enhancement_counts else 0,
            'avg_emphasis_tags': statistics.mean(emphasis_counts) if emphasis_counts else 0,
            'avg_prosody_tags': statistics.mean(prosody_counts) if prosody_counts else 0,
            'avg_break_tags': statistics.mean(break_counts) if break_counts else 0,
            'avg_audio_duration_ms': statistics.mean(audio_durations) if audio_durations else 0,
            'avg_file_size_bytes': statistics.mean(file_sizes) if file_sizes else 0,
            'avg_synthesis_time_ms': statistics.mean(synthesis_times) if synthesis_times else 0,
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'speaking_rate': data[0].get('speaking_rate', 'N/A') if data else 'N/A',
            'emphasis_level': data[0].get('emphasis_level', 'N/A') if data else 'N/A',
        }

    # Display results
    for config_name, metrics in results.items():
        print(f"\n{config_name.upper().replace('_', ' ')} Configuration:")
        print(f"  Speaking Rate: {metrics['speaking_rate']}")
        print(f"  Emphasis Level: {metrics['emphasis_level']}")
        print(f"  Total Interactions: {metrics['total_interactions']}")
        print(f"  Average Latency: {metrics['avg_latency']:.2f} ms")
        print(f"  Average Synthesis Time: {metrics['avg_synthesis_time_ms']:.2f} ms")
        print(f"  Quality Scores:")
        print(f"    MOS Score: {metrics['avg_mos_score']:.1f}/5.0")
        print(f"    Naturalness: {metrics['avg_naturalness_score']:.1f}/10.0")
        print(f"    Clarity: {metrics['avg_clarity_score']:.1f}/10.0")
        print(f"  SSML Usage:")
        print(f"    Average Enhancements: {metrics['avg_enhancement_count']:.1f}")
        print(f"    Emphasis Tags: {metrics['avg_emphasis_tags']:.1f}")
        print(f"    Prosody Tags: {metrics['avg_prosody_tags']:.1f}")
        print(f"    Break Tags: {metrics['avg_break_tags']:.1f}")
        print(f"  Audio Metrics:")
        print(f"    Average Duration: {metrics['avg_audio_duration_ms']:.0f} ms")
        print(f"    Average File Size: {metrics['avg_file_size_bytes']:.0f} bytes")

    # Find best configuration for speech quality
    if results:
        # Score based on MOS, naturalness, clarity, and reasonable synthesis time
        scored_configs = []
        for config_name, metrics in results.items():
            # Higher quality scores are better, lower synthesis time is better
            quality_score = (metrics['avg_mos_score'] * 2 +
                           metrics['avg_naturalness_score'] +
                           metrics['avg_clarity_score'] -
                           metrics['avg_synthesis_time_ms'] / 1000)
            scored_configs.append((config_name, quality_score, metrics))

        scored_configs.sort(key=lambda x: x[1], reverse=True)  # Sort by score (higher is better)

        print(f"\n🏆 BEST SPEECH QUALITY: {scored_configs[0][0].upper().replace('_', ' ')}")
        best_metrics = scored_configs[0][2]
        print(f"   MOS Score: {best_metrics['avg_mos_score']:.1f}/5.0")
        print(f"   Naturalness: {best_metrics['avg_naturalness_score']:.1f}/10.0")
        print(f"   Clarity: {best_metrics['avg_clarity_score']:.1f}/10.0")
        print(f"   Average Synthesis Time: {best_metrics['avg_synthesis_time_ms']:.2f} ms")
        print(f"   SSML Enhancements: {best_metrics['avg_enhancement_count']:.1f}")

    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Agent Benchmarking Tool")
    parser.add_argument("--mode", choices=["baseline", "optimized", "human_like"],
                       help="Run specific agent mode")
    parser.add_argument("--compare", action="store_true",
                       help="Generate comparison report from existing metrics")
    parser.add_argument("--analyze-vad", action="store_true",
                       help="Analyze VAD tuning results")
    parser.add_argument("--analyze-turn-detector", action="store_true",
                       help="Analyze turn detector comparison results")
    parser.add_argument("--analyze-streaming", action="store_true",
                       help="Analyze streaming STT and partial hypotheses results")
    parser.add_argument("--analyze-ssml", action="store_true",
                       help="Analyze SSML TTS enhancement results")
    parser.add_argument("--baseline-file", default="baseline_metrics_*.csv",
                       help="Baseline metrics CSV file pattern")
    parser.add_argument("--optimized-file", default="optimized_metrics_*.csv",
                       help="Optimized metrics CSV file pattern")
    parser.add_argument("--output-dir", default=".",
                       help="Output directory for reports")

    args = parser.parse_args()

    if args.analyze_vad:
        analyze_vad_tuning_results(args.output_dir)
    elif args.analyze_turn_detector:
        analyze_turn_detector_results(args.output_dir)
    elif args.analyze_streaming:
        analyze_streaming_stt_results(args.output_dir)
    elif args.analyze_ssml:
        analyze_ssml_tts_results(args.output_dir)
    elif args.compare:
        # Find the most recent metrics files
        import glob

        baseline_files = glob.glob(args.baseline_file)
        optimized_files = glob.glob(args.optimized_file)

        if not baseline_files or not optimized_files:
            print("Error: Could not find metrics files for comparison")
            print(f"Looking for baseline: {args.baseline_file}")
            print(f"Looking for optimized: {args.optimized_file}")
            exit(1)

        # Use the most recent files
        baseline_file = max(baseline_files, key=os.path.getctime)
        optimized_file = max(optimized_files, key=os.path.getctime)

        print(f"Comparing:")
        print(f"  Baseline: {baseline_file}")
        print(f"  Optimized: {optimized_file}")

        generate_comparison_report(baseline_file, optimized_file, args.output_dir)
        create_visualization(baseline_file, optimized_file, args.output_dir)

    elif args.mode:
        print(f"To run {args.mode} mode, use:")
        if args.mode == "baseline":
            print("python baseline_voice_agent.py")
        elif args.mode == "optimized":
            print("python optimized_voice_agent.py")
        elif args.mode == "human_like":
            print("python voice_agent.py  # (with CURRENT_MODE = 'human_like')")
    else:
        parser.print_help()
