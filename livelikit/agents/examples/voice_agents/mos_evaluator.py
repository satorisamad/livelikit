#!/usr/bin/env python3
"""
MOS (Mean Opinion Score) Evaluator for Voice Agent TTS

This script helps evaluate TTS quality by playing audio files and collecting
subjective ratings for clarity, expressiveness, and overall quality.

Usage:
    python mos_evaluator.py [metrics_file.csv]
"""

import csv
import os
import sys
import subprocess
import platform
from typing import Dict, List
import argparse

class MOSEvaluator:
    def __init__(self, metrics_file: str):
        self.metrics_file = metrics_file
        self.output_file = metrics_file.replace('.csv', '_mos_evaluated.csv')
        self.audio_files = []
        self.load_audio_files()
    
    def load_audio_files(self):
        """Load audio file paths from metrics CSV"""
        if not os.path.exists(self.metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['tts_wav_path'] and os.path.exists(row['tts_wav_path']):
                    self.audio_files.append({
                        'interaction_id': row['interaction_id'],
                        'wav_path': row['tts_wav_path'],
                        'agent_response': row['agent_response'],
                        'user_transcript': row['user_transcript']
                    })
        
        print(f"Found {len(self.audio_files)} audio files to evaluate")
    
    def play_audio(self, file_path: str):
        """Play audio file using system default player"""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            else:  # Linux
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            print(f"Could not play audio file: {e}")
            print(f"Please manually play: {file_path}")
    
    def get_mos_rating(self, prompt: str, scale_info: str) -> int:
        """Get MOS rating from user input"""
        while True:
            try:
                print(f"\n{prompt}")
                print(scale_info)
                rating = int(input("Enter rating (1-5): "))
                if 1 <= rating <= 5:
                    return rating
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user")
                sys.exit(0)
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single audio sample"""
        print("\n" + "="*80)
        print(f"Evaluating: {sample['interaction_id']}")
        print(f"User said: '{sample['user_transcript']}'")
        print(f"Agent response: '{sample['agent_response']}'")
        print(f"Audio file: {sample['wav_path']}")
        print("="*80)
        
        # Play audio
        input("Press Enter to play audio...")
        self.play_audio(sample['wav_path'])
        
        # Collect ratings
        clarity_scale = """
1 = Very poor (unintelligible, lots of artifacts)
2 = Poor (difficult to understand, some artifacts)
3 = Fair (understandable with effort, minor artifacts)
4 = Good (clear and understandable)
5 = Excellent (very clear, no artifacts)"""
        
        expressiveness_scale = """
1 = Very poor (monotone, robotic)
2 = Poor (little variation, sounds artificial)
3 = Fair (some variation, somewhat natural)
4 = Good (good variation, mostly natural)
5 = Excellent (very expressive, human-like)"""
        
        overall_scale = """
1 = Very poor (would not use)
2 = Poor (barely acceptable)
3 = Fair (acceptable with reservations)
4 = Good (good quality)
5 = Excellent (excellent quality)"""
        
        clarity = self.get_mos_rating("Rate CLARITY:", clarity_scale)
        expressiveness = self.get_mos_rating("Rate EXPRESSIVENESS:", expressiveness_scale)
        overall = self.get_mos_rating("Rate OVERALL QUALITY:", overall_scale)
        
        # Optional comments
        comments = input("Optional comments: ").strip()
        
        return {
            'clarity_score': clarity,
            'expressiveness_score': expressiveness,
            'mos_score': overall,
            'comments': comments
        }
    
    def run_evaluation(self, start_from: int = 0, max_samples: int = None):
        """Run MOS evaluation on audio samples"""
        if not self.audio_files:
            print("No audio files found to evaluate")
            return
        
        # Load existing evaluations if file exists
        existing_evaluations = {}
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['mos_score']:  # Only count completed evaluations
                        existing_evaluations[row['interaction_id']] = row
        
        print(f"Found {len(existing_evaluations)} existing evaluations")
        
        # Filter out already evaluated samples
        samples_to_evaluate = [
            sample for sample in self.audio_files[start_from:]
            if sample['interaction_id'] not in existing_evaluations
        ]
        
        if max_samples:
            samples_to_evaluate = samples_to_evaluate[:max_samples]
        
        print(f"Will evaluate {len(samples_to_evaluate)} samples")
        
        if not samples_to_evaluate:
            print("All samples have already been evaluated!")
            return
        
        # Confirm before starting
        response = input(f"Start evaluation of {len(samples_to_evaluate)} samples? (y/n): ")
        if response.lower() != 'y':
            print("Evaluation cancelled")
            return
        
        # Evaluate samples
        evaluations = {}
        for i, sample in enumerate(samples_to_evaluate):
            print(f"\nProgress: {i+1}/{len(samples_to_evaluate)}")
            
            try:
                evaluation = self.evaluate_sample(sample)
                evaluations[sample['interaction_id']] = {
                    **sample,
                    **evaluation
                }
                
                # Save progress after each evaluation
                self.save_evaluations(evaluations, existing_evaluations)
                
            except KeyboardInterrupt:
                print(f"\nEvaluation stopped. Progress saved.")
                break
        
        print(f"\nEvaluation complete! Results saved to: {self.output_file}")
        self.print_summary(evaluations)
    
    def save_evaluations(self, new_evaluations: Dict, existing_evaluations: Dict):
        """Save evaluations to CSV file"""
        # Read original metrics file to get all columns
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            original_fieldnames = reader.fieldnames
            all_rows = list(reader)
        
        # Write updated file
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            # Add MOS fields to fieldnames if not present
            fieldnames = list(original_fieldnames)
            mos_fields = ['mos_score', 'clarity_score', 'expressiveness_score', 'comments']
            for field in mos_fields:
                if field not in fieldnames:
                    fieldnames.append(field)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in all_rows:
                interaction_id = row['interaction_id']
                
                # Update with existing evaluations
                if interaction_id in existing_evaluations:
                    row.update(existing_evaluations[interaction_id])
                
                # Update with new evaluations
                if interaction_id in new_evaluations:
                    eval_data = new_evaluations[interaction_id]
                    row.update({
                        'mos_score': eval_data['mos_score'],
                        'clarity_score': eval_data['clarity_score'],
                        'expressiveness_score': eval_data['expressiveness_score'],
                        'comments': eval_data.get('comments', '')
                    })
                
                writer.writerow(row)
    
    def print_summary(self, evaluations: Dict):
        """Print summary of evaluations"""
        if not evaluations:
            return
        
        clarity_scores = [e['clarity_score'] for e in evaluations.values()]
        expressiveness_scores = [e['expressiveness_score'] for e in evaluations.values()]
        overall_scores = [e['mos_score'] for e in evaluations.values()]
        
        print(f"\n{'EVALUATION SUMMARY':-^50}")
        print(f"Samples evaluated: {len(evaluations)}")
        print(f"Average Clarity: {sum(clarity_scores)/len(clarity_scores):.2f}")
        print(f"Average Expressiveness: {sum(expressiveness_scores)/len(expressiveness_scores):.2f}")
        print(f"Average Overall (MOS): {sum(overall_scores)/len(overall_scores):.2f}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate TTS quality with MOS ratings')
    parser.add_argument('metrics_file', nargs='?', 
                       default='voice_agent_metrics.csv',
                       help='Path to metrics CSV file')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start evaluation from this sample index')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    try:
        evaluator = MOSEvaluator(args.metrics_file)
        evaluator.run_evaluation(args.start_from, args.max_samples)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
