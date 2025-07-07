#!/usr/bin/env python3
"""
Voice Agent Launcher

Simplified launcher for voice agents that can be controlled from Streamlit.
Provides a bridge between the Streamlit interface and LiveKit agents.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Optional
import threading
import queue

class VoiceAgentLauncher:
    """
    Launcher for voice agents with configuration management.
    Provides a simple interface for starting/stopping agents with different configurations.
    """
    
    def __init__(self):
        self.current_process = None
        self.current_config = None
        self.metrics_queue = queue.Queue()
        self.is_running = False
        
    def create_config_file(self, config: Dict) -> str:
        """Create a temporary configuration file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = f"temp_config_{timestamp}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def get_agent_script(self, agent_type: str) -> str:
        """Get the appropriate agent script for the configuration"""
        agent_scripts = {
            "baseline": "baseline_voice_agent.py",
            "optimized": "optimized_voice_agent.py",
            "vad_tuning": "vad_tuning_agent.py",
            "turn_detector": "turn_detector_agent.py",
            "streaming_stt": "streaming_stt_agent.py",
            "ssml_tts": "ssml_tts_agent.py"
        }
        
        return agent_scripts.get(agent_type, "baseline_voice_agent.py")
    
    def start_agent(self, config: Dict) -> bool:
        """Start a voice agent with the given configuration"""
        try:
            # Stop any existing agent
            self.stop_agent()
            
            # Create configuration file
            config_file = self.create_config_file(config)
            
            # Get agent script
            agent_script = self.get_agent_script(config.get("agent_type", "baseline"))
            
            # Check if script exists
            if not os.path.exists(agent_script):
                print(f"Agent script not found: {agent_script}")
                return False
            
            # Start the agent process
            cmd = [sys.executable, agent_script]
            
            # Add environment variables for configuration
            env = os.environ.copy()
            env["VOICE_AGENT_CONFIG"] = config_file
            
            self.current_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.current_config = config
            self.is_running = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_agent)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            print(f"Started {config.get('agent_type', 'baseline')} agent with PID {self.current_process.pid}")
            return True
            
        except Exception as e:
            print(f"Error starting agent: {e}")
            return False
    
    def stop_agent(self) -> bool:
        """Stop the currently running agent"""
        try:
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.current_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if necessary
                    self.current_process.kill()
                    self.current_process.wait()
                
                print(f"Stopped agent with PID {self.current_process.pid}")
            
            self.current_process = None
            self.current_config = None
            self.is_running = False
            
            # Clean up temporary config files
            self._cleanup_temp_files()
            
            return True
            
        except Exception as e:
            print(f"Error stopping agent: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get the current status of the agent"""
        if not self.current_process:
            return {
                "running": False,
                "config": None,
                "pid": None,
                "uptime": 0
            }
        
        is_running = self.current_process.poll() is None
        
        return {
            "running": is_running,
            "config": self.current_config,
            "pid": self.current_process.pid if is_running else None,
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }
    
    def get_metrics(self) -> Optional[Dict]:
        """Get latest metrics from the agent"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _monitor_agent(self):
        """Monitor the agent process and collect metrics"""
        self.start_time = time.time()
        
        while self.is_running and self.current_process:
            if self.current_process.poll() is not None:
                # Process has terminated
                self.is_running = False
                break
            
            # Simulate metrics collection (in practice, this would read from agent logs)
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - self.start_time,
                "status": "running",
                "config": self.current_config.get("agent_type", "unknown") if self.current_config else "unknown"
            }
            
            try:
                self.metrics_queue.put_nowait(metrics)
            except queue.Full:
                # Remove old metrics if queue is full
                try:
                    self.metrics_queue.get_nowait()
                    self.metrics_queue.put_nowait(metrics)
                except queue.Empty:
                    pass
            
            time.sleep(1)  # Update every second
    
    def _cleanup_temp_files(self):
        """Clean up temporary configuration files"""
        import glob
        temp_files = glob.glob("temp_config_*.json")
        for file in temp_files:
            try:
                os.remove(file)
            except OSError:
                pass

class AgentConfigurationManager:
    """
    Manager for voice agent configurations.
    Provides presets and validation for different optimization scenarios.
    """
    
    def __init__(self):
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict:
        """Load predefined configuration presets"""
        return {
            "baseline": {
                "agent_type": "baseline",
                "name": "Baseline Configuration",
                "description": "Basic voice agent without optimizations",
                "llm_model": "gpt-4o-mini",
                "llm_temperature": 0.7,
                "stt_model": "nova-2",
                "stt_interim_results": False,
                "tts_model": "sonic-2",
                "tts_speed": 1.0,
                "min_endpointing_delay": 0.5,
                "max_endpointing_delay": 6.0,
                "enable_optimizations": False
            },
            
            "optimized": {
                "agent_type": "optimized",
                "name": "Optimized Configuration",
                "description": "Fully optimized voice agent with all features",
                "llm_model": "gpt-4o-mini",
                "llm_temperature": 0.7,
                "stt_model": "nova-2",
                "stt_interim_results": True,
                "tts_model": "sonic-2",
                "tts_speed": 1.0,
                "min_endpointing_delay": 0.2,
                "max_endpointing_delay": 3.0,
                "enable_streaming_stt": True,
                "enable_partial_llm": True,
                "enable_ssml": True,
                "enable_turn_detector_plugin": True,
                "enable_optimizations": True
            },
            
            "healthcare": {
                "agent_type": "ssml_tts",
                "name": "Healthcare Optimized",
                "description": "Healthcare-specific optimizations with SSML",
                "llm_model": "gpt-4o-mini",
                "llm_temperature": 0.7,
                "stt_model": "nova-2",
                "stt_interim_results": True,
                "tts_model": "sonic-2",
                "tts_speed": 0.9,
                "min_endpointing_delay": 0.3,
                "max_endpointing_delay": 4.0,
                "enable_ssml": True,
                "ssml_config": "healthcare_ssml",
                "speaking_rate": "slow",
                "emphasis_level": "strong",
                "enable_optimizations": True
            }
        }
    
    def get_preset(self, name: str) -> Optional[Dict]:
        """Get a configuration preset by name"""
        return self.presets.get(name)
    
    def list_presets(self):
        """List available configuration presets"""
        return list(self.presets.keys())

    def validate_config(self, config: Dict):
        """Validate a configuration"""
        required_fields = ["agent_type", "llm_model", "stt_model", "tts_model"]
        
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Validate specific values
        valid_agents = ["baseline", "optimized", "vad_tuning", "turn_detector", "streaming_stt", "ssml_tts"]
        if config["agent_type"] not in valid_agents:
            return False, f"Invalid agent_type: {config['agent_type']}"
        
        return True, "Configuration is valid"
    
    def create_custom_config(self, base_preset: str, overrides: Dict) -> Dict:
        """Create a custom configuration based on a preset with overrides"""
        base_config = self.get_preset(base_preset)
        if not base_config:
            raise ValueError(f"Unknown preset: {base_preset}")
        
        custom_config = base_config.copy()
        custom_config.update(overrides)
        
        return custom_config

# Global instances for use in Streamlit
agent_launcher = VoiceAgentLauncher()
config_manager = AgentConfigurationManager()

def main():
    """Command-line interface for the agent launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Agent Launcher")
    parser.add_argument("--config", help="Configuration preset name")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--stop", action="store_true", help="Stop running agent")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        for preset in config_manager.list_presets():
            config = config_manager.get_preset(preset)
            print(f"  {preset}: {config['description']}")
    
    elif args.stop:
        if agent_launcher.stop_agent():
            print("Agent stopped successfully")
        else:
            print("Failed to stop agent")
    
    elif args.status:
        status = agent_launcher.get_status()
        print(f"Agent Status: {status}")
    
    elif args.config:
        config = config_manager.get_preset(args.config)
        if config:
            if agent_launcher.start_agent(config):
                print(f"Started {args.config} agent")
                
                # Keep running and show status
                try:
                    while True:
                        time.sleep(5)
                        status = agent_launcher.get_status()
                        if not status["running"]:
                            print("Agent has stopped")
                            break
                        print(f"Agent running (uptime: {status['uptime']:.1f}s)")
                except KeyboardInterrupt:
                    print("\nStopping agent...")
                    agent_launcher.stop_agent()
            else:
                print(f"Failed to start {args.config} agent")
        else:
            print(f"Unknown preset: {args.config}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
