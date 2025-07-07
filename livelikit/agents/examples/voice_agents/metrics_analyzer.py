#!/usr/bin/env python3
"""
Voice Agent Metrics Analyzer

This script analyzes the metrics collected by the voice agent to provide
comprehensive reports on:
- End-to-end latency analysis
- Word Error Rate (WER) statistics  
- TTS quality metrics
- Performance trends over time

Usage:
    python metrics_analyzer.py [metrics_file.csv]
"""

import csv
import json
import statistics
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class VoiceAgentMetricsAnalyzer:
    def __init__(self, metrics_file: str):
        self.metrics_file = metrics_file
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load metrics data from CSV file"""
        if not os.path.exists(self.metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        
        print(f"Loaded {len(self.data)} metrics records")
    
    def analyze_latency(self) -> Dict:
        """Analyze end-to-end latency metrics"""
        latencies = []
        component_latencies = {
            'mic_to_transcript': [],
            'transcript_to_llm': [],
            'llm_to_tts_start': [],
            'tts_synthesis': []
        }
        
        for row in self.data:
            if row['end_to_end_latency_ms']:
                latencies.append(float(row['end_to_end_latency_ms']))
            
            for component, values in component_latencies.items():
                col_name = f"{component}_ms"
                if row.get(col_name):
                    values.append(float(row[col_name]))
        
        analysis = {
            'end_to_end': self._calculate_stats(latencies),
            'components': {k: self._calculate_stats(v) for k, v in component_latencies.items()}
        }
        
        return analysis
    
    def analyze_wer(self) -> Dict:
        """Analyze Word Error Rate statistics"""
        wer_scores = []
        transcript_lengths = []
        
        for row in self.data:
            if row['wer_score']:
                wer_scores.append(float(row['wer_score']))
            if row['user_transcript']:
                transcript_lengths.append(len(row['user_transcript'].split()))
        
        return {
            'wer_stats': self._calculate_stats(wer_scores),
            'transcript_length_stats': self._calculate_stats(transcript_lengths),
            'total_transcripts_with_wer': len(wer_scores)
        }
    
    def analyze_tts_quality(self) -> Dict:
        """Analyze TTS quality metrics"""
        tts_durations = []
        file_sizes = []
        synthesis_times = []
        
        for row in self.data:
            if row['tts_duration_ms']:
                tts_durations.append(float(row['tts_duration_ms']))
            if row['tts_file_size_bytes']:
                file_sizes.append(int(row['tts_file_size_bytes']))
            if row['tts_synthesis_ms']:
                synthesis_times.append(float(row['tts_synthesis_ms']))
        
        return {
            'audio_duration_stats': self._calculate_stats(tts_durations),
            'file_size_stats': self._calculate_stats(file_sizes),
            'synthesis_time_stats': self._calculate_stats(synthesis_times),
            'real_time_factor': self._calculate_rtf(synthesis_times, tts_durations)
        }
    
    def _calculate_rtf(self, synthesis_times: List[float], audio_durations: List[float]) -> Dict:
        """Calculate Real-Time Factor (synthesis_time / audio_duration)"""
        if not synthesis_times or not audio_durations:
            return {}
        
        rtf_values = []
        for i in range(min(len(synthesis_times), len(audio_durations))):
            if audio_durations[i] > 0:
                rtf = synthesis_times[i] / audio_durations[i]
                rtf_values.append(rtf)
        
        return self._calculate_stats(rtf_values)
    
    def _calculate_stats(self, values: List[float]) -> Dict:
        """Calculate statistical measures for a list of values"""
        if not values:
            return {'count': 0}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'percentile_95': sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0],
            'percentile_99': sorted(values)[int(0.99 * len(values))] if len(values) > 1 else values[0]
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'metrics_file': self.metrics_file,
                'total_records': len(self.data)
            },
            'latency_analysis': self.analyze_latency(),
            'wer_analysis': self.analyze_wer(),
            'tts_quality_analysis': self.analyze_tts_quality(),
            'error_analysis': self._analyze_errors()
        }
        
        return report
    
    def _analyze_errors(self) -> Dict:
        """Analyze error occurrences"""
        total_errors = 0
        error_messages = {}
        
        for row in self.data:
            if row['error_occurred'] == 'True':
                total_errors += 1
                msg = row['error_message']
                error_messages[msg] = error_messages.get(msg, 0) + 1
        
        return {
            'total_errors': total_errors,
            'error_rate': total_errors / len(self.data) if self.data else 0,
            'error_types': error_messages
        }
    
    def print_report(self):
        """Print formatted analysis report to console"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("VOICE AGENT PERFORMANCE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nAnalysis Date: {report['metadata']['analysis_timestamp']}")
        print(f"Total Records: {report['metadata']['total_records']}")
        
        # Latency Analysis
        print(f"\n{'LATENCY ANALYSIS':-^60}")
        latency = report['latency_analysis']['end_to_end']
        if latency['count'] > 0:
            print(f"End-to-End Latency:")
            print(f"  Mean: {latency['mean']:.2f}ms")
            print(f"  Median: {latency['median']:.2f}ms")
            print(f"  95th percentile: {latency['percentile_95']:.2f}ms")
            print(f"  Range: {latency['min']:.2f}ms - {latency['max']:.2f}ms")
        
        # Component breakdown
        components = report['latency_analysis']['components']
        for comp_name, comp_stats in components.items():
            if comp_stats['count'] > 0:
                print(f"\n{comp_name.replace('_', ' ').title()}:")
                print(f"  Mean: {comp_stats['mean']:.2f}ms")
                print(f"  Median: {comp_stats['median']:.2f}ms")
        
        # WER Analysis
        print(f"\n{'WER ANALYSIS':-^60}")
        wer = report['wer_analysis']['wer_stats']
        if wer['count'] > 0:
            print(f"Word Error Rate:")
            print(f"  Mean: {wer['mean']:.4f}")
            print(f"  Median: {wer['median']:.4f}")
            print(f"  Range: {wer['min']:.4f} - {wer['max']:.4f}")
            print(f"  Transcripts analyzed: {wer['count']}")
        
        # TTS Quality Analysis
        print(f"\n{'TTS QUALITY ANALYSIS':-^60}")
        tts = report['tts_quality_analysis']
        rtf = tts.get('real_time_factor', {})
        if rtf.get('count', 0) > 0:
            print(f"Real-Time Factor (lower is better):")
            print(f"  Mean: {rtf['mean']:.3f}")
            print(f"  Median: {rtf['median']:.3f}")
        
        synthesis = tts['synthesis_time_stats']
        if synthesis['count'] > 0:
            print(f"TTS Synthesis Time:")
            print(f"  Mean: {synthesis['mean']:.2f}ms")
            print(f"  Median: {synthesis['median']:.2f}ms")
        
        # Error Analysis
        print(f"\n{'ERROR ANALYSIS':-^60}")
        errors = report['error_analysis']
        print(f"Total Errors: {errors['total_errors']}")
        print(f"Error Rate: {errors['error_rate']:.2%}")
        
        if errors['error_types']:
            print("Error Types:")
            for error_type, count in errors['error_types'].items():
                print(f"  {error_type}: {count}")
        
        print("\n" + "="*60)
    
    def save_report(self, output_file: str):
        """Save analysis report to JSON file"""
        report = self.generate_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze voice agent metrics')
    parser.add_argument('metrics_file', nargs='?', 
                       default='voice_agent_metrics.csv',
                       help='Path to metrics CSV file')
    parser.add_argument('--output', '-o', 
                       help='Output file for JSON report')
    
    args = parser.parse_args()
    
    try:
        analyzer = VoiceAgentMetricsAnalyzer(args.metrics_file)
        analyzer.print_report()
        
        if args.output:
            analyzer.save_report(args.output)
        else:
            # Auto-generate output filename
            base_name = os.path.splitext(args.metrics_file)[0]
            output_file = f"{base_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            analyzer.save_report(output_file)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
