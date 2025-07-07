#!/usr/bin/env python3
"""
Comprehensive Voice Agent Benchmarking System

This script provides a complete benchmarking framework for comparing
all voice agent optimization configurations and generating detailed reports.

Usage:
    python comprehensive_benchmark.py --run-all-tests
    python comprehensive_benchmark.py --generate-report
    python comprehensive_benchmark.py --compare-configurations
"""

import argparse
import csv
import json
import os
import statistics
import glob
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess
import time

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

def analyze_configuration_metrics(file_pattern: str, config_type: str) -> Dict:
    """Analyze metrics for a specific configuration type"""
    files = glob.glob(file_pattern)
    if not files:
        return {"error": f"No files found for pattern: {file_pattern}"}
    
    # Use the most recent file
    latest_file = max(files, key=os.path.getctime)
    data = load_metrics_from_csv(latest_file)
    
    if not data:
        return {"error": f"No data in file: {latest_file}"}
    
    # Extract common metrics
    latencies = [float(row['end_to_end_latency_ms']) for row in data if row.get('end_to_end_latency_ms')]
    response_lengths = [int(row['response_length_words']) for row in data if row.get('response_length_words')]
    
    analysis = {
        "config_type": config_type,
        "file_path": latest_file,
        "total_interactions": len(data),
        "timestamp": datetime.now().isoformat(),
        "latency_stats": calculate_statistics(latencies),
        "response_length_stats": calculate_statistics(response_lengths),
    }
    
    # Add configuration-specific metrics
    if config_type == "baseline":
        analysis["optimization_level"] = "none"
        
    elif config_type == "optimized":
        # Optimized-specific metrics
        partial_triggers = sum(1 for row in data if row.get('partial_llm_triggered') == 'True')
        analysis["partial_llm_trigger_rate"] = partial_triggers / max(len(data), 1)
        analysis["optimization_level"] = "full"
        
    elif config_type.startswith("vad_tuning"):
        # VAD tuning metrics
        false_triggers = sum(1 for row in data if row.get('false_trigger') == 'True')
        missed_endpoints = sum(1 for row in data if row.get('missed_endpoint') == 'True')
        analysis["false_trigger_rate"] = false_triggers / max(len(data), 1)
        analysis["missed_endpoint_rate"] = missed_endpoints / max(len(data), 1)
        analysis["optimization_level"] = "vad_tuned"
        
    elif config_type.startswith("turn_detector"):
        # Turn detector metrics
        flow_scores = [float(row['conversation_flow_score']) for row in data if row.get('conversation_flow_score')]
        analysis["avg_conversation_flow"] = statistics.mean(flow_scores) if flow_scores else 0
        analysis["optimization_level"] = "turn_detector"
        
    elif config_type.startswith("streaming_stt"):
        # Streaming STT metrics
        early_gains = [float(row['early_processing_gain_ms']) for row in data if row.get('early_processing_gain_ms')]
        accuracy_scores = [float(row['transcript_accuracy_score']) for row in data if row.get('transcript_accuracy_score')]
        analysis["avg_early_processing_gain"] = statistics.mean(early_gains) if early_gains else 0
        analysis["avg_transcript_accuracy"] = statistics.mean(accuracy_scores) if accuracy_scores else 0
        analysis["optimization_level"] = "streaming_stt"
        
    elif config_type.startswith("ssml_tts"):
        # SSML TTS metrics
        mos_scores = [float(row['estimated_mos_score']) for row in data if row.get('estimated_mos_score')]
        naturalness_scores = [float(row['naturalness_score']) for row in data if row.get('naturalness_score')]
        clarity_scores = [float(row['clarity_score']) for row in data if row.get('clarity_score')]
        analysis["avg_mos_score"] = statistics.mean(mos_scores) if mos_scores else 0
        analysis["avg_naturalness_score"] = statistics.mean(naturalness_scores) if naturalness_scores else 0
        analysis["avg_clarity_score"] = statistics.mean(clarity_scores) if clarity_scores else 0
        analysis["optimization_level"] = "ssml_enhanced"
    
    return analysis

def generate_comprehensive_report(output_dir: str = ".") -> str:
    """Generate a comprehensive comparison report of all configurations"""
    
    print("🔍 Analyzing all voice agent configurations...")
    
    # Define configuration patterns to analyze
    config_patterns = {
        "baseline": "baseline_metrics_*.csv",
        "optimized": "optimized_metrics_*.csv",
        "vad_tuning_fast": "vad_tuning_fast_response_*.csv",
        "vad_tuning_balanced": "vad_tuning_balanced_*.csv",
        "vad_tuning_patient": "vad_tuning_patient_*.csv",
        "turn_detector_vad": "turn_detector_basic_vad_*.csv",
        "turn_detector_multilingual": "turn_detector_multilingual_model_*.csv",
        "streaming_stt_basic": "streaming_stt_basic_streaming_*.csv",
        "streaming_stt_partial": "streaming_stt_partial_processing_*.csv",
        "ssml_tts_basic": "ssml_tts_basic_ssml_*.csv",
        "ssml_tts_healthcare": "ssml_tts_healthcare_ssml_*.csv",
    }
    
    # Analyze each configuration
    analyses = {}
    for config_name, pattern in config_patterns.items():
        print(f"  Analyzing {config_name}...")
        analysis = analyze_configuration_metrics(pattern, config_name)
        if "error" not in analysis:
            analyses[config_name] = analysis
        else:
            print(f"    ⚠️  {analysis['error']}")
    
    if not analyses:
        print("❌ No configuration data found for analysis")
        return ""
    
    # Create comprehensive comparison
    report = {
        "report_timestamp": datetime.now().isoformat(),
        "configurations_analyzed": len(analyses),
        "configurations": analyses,
        
        "performance_comparison": {},
        "optimization_effectiveness": {},
        "recommendations": {}
    }
    
    # Calculate performance comparisons
    baseline_latency = analyses.get("baseline", {}).get("latency_stats", {}).get("mean", 0)
    
    for config_name, analysis in analyses.items():
        if config_name != "baseline" and baseline_latency > 0:
            config_latency = analysis.get("latency_stats", {}).get("mean", 0)
            if config_latency > 0:
                improvement = ((baseline_latency - config_latency) / baseline_latency) * 100
                report["performance_comparison"][config_name] = {
                    "latency_improvement_percent": improvement,
                    "latency_baseline_ms": baseline_latency,
                    "latency_optimized_ms": config_latency,
                    "latency_reduction_ms": baseline_latency - config_latency
                }
    
    # Generate optimization effectiveness scores
    for config_name, analysis in analyses.items():
        optimization_level = analysis.get("optimization_level", "unknown")
        latency = analysis.get("latency_stats", {}).get("mean", 0)
        
        # Calculate effectiveness score (0-100)
        effectiveness_score = 50  # Base score
        
        if optimization_level == "full":
            effectiveness_score += 20
        elif optimization_level in ["vad_tuned", "turn_detector", "streaming_stt", "ssml_enhanced"]:
            effectiveness_score += 10
        
        # Bonus for low latency
        if latency > 0 and baseline_latency > 0:
            latency_bonus = min(20, (baseline_latency - latency) / baseline_latency * 20)
            effectiveness_score += latency_bonus
        
        # Configuration-specific bonuses
        if config_name.startswith("turn_detector"):
            flow_score = analysis.get("avg_conversation_flow", 0)
            effectiveness_score += (flow_score - 5) * 2  # Bonus for good conversation flow
        
        if config_name.startswith("ssml_tts"):
            mos_score = analysis.get("avg_mos_score", 0)
            effectiveness_score += (mos_score - 3) * 10  # Bonus for good speech quality
        
        report["optimization_effectiveness"][config_name] = max(0, min(100, effectiveness_score))
    
    # Generate recommendations
    best_latency = min((a.get("latency_stats", {}).get("mean", float('inf')) 
                       for a in analyses.values() if a.get("latency_stats", {}).get("mean", 0) > 0), 
                      default=0)
    
    best_config = None
    for config_name, analysis in analyses.items():
        if analysis.get("latency_stats", {}).get("mean", 0) == best_latency:
            best_config = config_name
            break
    
    report["recommendations"] = {
        "fastest_configuration": best_config,
        "fastest_latency_ms": best_latency,
        "recommended_for_production": "optimized" if "optimized" in analyses else best_config,
        "recommended_for_quality": next((name for name in analyses.keys() if name.startswith("ssml_tts")), None),
        "recommended_for_naturalness": next((name for name in analyses.keys() if name.startswith("turn_detector")), None),
    }
    
    # Save detailed report
    report_file = os.path.join(output_dir, f"comprehensive_voice_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    print("\n" + "="*80)
    print("COMPREHENSIVE VOICE AGENT BENCHMARK REPORT")
    print("="*80)
    
    print(f"\n📊 Configurations Analyzed: {len(analyses)}")
    
    if baseline_latency > 0:
        print(f"\n⚡ Performance Improvements (vs Baseline {baseline_latency:.2f}ms):")
        for config_name, perf in report["performance_comparison"].items():
            improvement = perf["latency_improvement_percent"]
            print(f"  {config_name}: {improvement:+.1f}% ({perf['latency_reduction_ms']:+.2f}ms)")
    
    print(f"\n🎯 Optimization Effectiveness Scores:")
    for config_name, score in report["optimization_effectiveness"].items():
        print(f"  {config_name}: {score:.1f}/100")
    
    print(f"\n🏆 Recommendations:")
    recs = report["recommendations"]
    print(f"  Fastest: {recs['fastest_configuration']} ({recs['fastest_latency_ms']:.2f}ms)")
    print(f"  Production: {recs['recommended_for_production']}")
    print(f"  Quality: {recs['recommended_for_quality']}")
    print(f"  Naturalness: {recs['recommended_for_naturalness']}")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print("="*80)
    
    return report_file

def run_test_suite(test_duration_minutes: int = 5):
    """Run a comprehensive test suite across all configurations"""
    
    print(f"🚀 Starting comprehensive test suite (duration: {test_duration_minutes} minutes per config)")
    
    # List of test configurations to run
    test_configs = [
        ("baseline_voice_agent.py", "Baseline"),
        ("optimized_voice_agent.py", "Optimized"),
        ("vad_tuning_agent.py", "VAD Tuning"),
        ("turn_detector_agent.py", "Turn Detector"),
        ("streaming_stt_agent.py", "Streaming STT"),
        ("ssml_tts_agent.py", "SSML TTS"),
    ]
    
    print("⚠️  Note: This would run each agent for testing.")
    print("⚠️  In practice, you would run each agent manually or with a test framework.")
    print("⚠️  Each agent needs to be started in a LiveKit room for proper testing.")
    
    for script, name in test_configs:
        print(f"\n📋 Test configuration: {name}")
        print(f"   Script: {script}")
        print(f"   To run: python {script}")
        print(f"   Duration: {test_duration_minutes} minutes")
    
    print(f"\n✅ Test suite configuration complete")
    print(f"💡 Run each agent manually, then use --generate-report to analyze results")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Voice Agent Benchmarking System")
    parser.add_argument("--run-all-tests", action="store_true",
                       help="Run comprehensive test suite across all configurations")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate comprehensive comparison report")
    parser.add_argument("--test-duration", type=int, default=5,
                       help="Duration in minutes for each test configuration")
    parser.add_argument("--output-dir", default=".",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    if args.run_all_tests:
        run_test_suite(args.test_duration)
    elif args.generate_report:
        generate_comprehensive_report(args.output_dir)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("  1. Run individual agents to collect data")
        print("  2. Use --generate-report to analyze results")
        print("  3. Use benchmark_voice_agents.py for specific analysis")

if __name__ == "__main__":
    main()
