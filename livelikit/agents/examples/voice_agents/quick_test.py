#!/usr/bin/env python3
"""
Quick Test Script for Voice Agent Optimization

This script helps you quickly test different configurations and see results.
"""

import subprocess
import time
import os
import sys
from datetime import datetime

def run_test(agent_script, duration_minutes=2):
    """Run a voice agent test for a specified duration"""
    print(f"\n🚀 Starting {agent_script} test for {duration_minutes} minutes...")
    print(f"⏰ Test will run until {(datetime.now().timestamp() + duration_minutes * 60)}")
    
    try:
        # Start the agent
        process = subprocess.Popen([sys.executable, agent_script])
        
        print(f"✅ Agent started with PID {process.pid}")
        print("🎤 Join the LiveKit room and start speaking!")
        print("💡 Try these test phrases:")
        print("   - 'Hi, I need to schedule an appointment'")
        print("   - 'Do you accept Blue Cross insurance?'")
        print("   - 'I'm having chest pain and need urgent care'")
        print("   - 'Can you help me reschedule my appointment?'")
        print("   - 'What are your clinic hours?'")
        
        # Wait for the specified duration
        time.sleep(duration_minutes * 60)
        
        # Stop the agent
        process.terminate()
        process.wait(timeout=10)
        
        print(f"✅ Test completed for {agent_script}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Test stopped by user for {agent_script}")
        if process:
            process.terminate()
            process.wait(timeout=10)
    except Exception as e:
        print(f"❌ Error running {agent_script}: {e}")

def main():
    print("🎙️ Voice Agent Quick Test Suite")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("   Create a .env file with your API keys before testing.")
        print("   See .env.example for template.")
        return
    
    # Test configurations
    test_configs = [
        ("baseline_voice_agent.py", "Baseline (No Optimizations)"),
        ("optimized_voice_agent.py", "Optimized (All Features)"),
    ]
    
    print("📋 Available test configurations:")
    for i, (script, name) in enumerate(test_configs, 1):
        print(f"   {i}. {name}")
    
    print("\n🎯 Test Options:")
    print("   A. Run all tests (recommended)")
    print("   B. Run specific test")
    print("   C. Quick comparison test")
    
    choice = input("\nSelect option (A/B/C): ").upper()
    
    if choice == 'A':
        # Run all tests
        duration = input("Test duration per configuration (minutes, default 2): ")
        duration = int(duration) if duration.isdigit() else 2
        
        for script, name in test_configs:
            if os.path.exists(script):
                print(f"\n📊 Testing: {name}")
                run_test(script, duration)
            else:
                print(f"⚠️  Script not found: {script}")
        
        print("\n🎉 All tests completed!")
        print("📈 Check the Results Analysis in Streamlit to compare performance")
        
    elif choice == 'B':
        # Run specific test
        print("\nSelect configuration to test:")
        for i, (script, name) in enumerate(test_configs, 1):
            print(f"   {i}. {name}")
        
        selection = input("Enter number: ")
        if selection.isdigit() and 1 <= int(selection) <= len(test_configs):
            script, name = test_configs[int(selection) - 1]
            if os.path.exists(script):
                duration = input("Test duration (minutes, default 2): ")
                duration = int(duration) if duration.isdigit() else 2
                run_test(script, duration)
            else:
                print(f"⚠️  Script not found: {script}")
        else:
            print("❌ Invalid selection")
    
    elif choice == 'C':
        # Quick comparison
        print("\n⚡ Quick Comparison Test (1 minute each)")
        
        for script, name in test_configs:
            if os.path.exists(script):
                print(f"\n📊 Quick test: {name}")
                run_test(script, 1)
            else:
                print(f"⚠️  Script not found: {script}")
        
        print("\n🎉 Quick comparison completed!")
        
    else:
        print("❌ Invalid choice")
    
    # Show next steps
    print("\n📋 Next Steps:")
    print("1. Open Streamlit app: streamlit run streamlit_simple.py")
    print("2. Go to 'Results Analysis' tab")
    print("3. Select the generated CSV files")
    print("4. Compare performance metrics")
    print("5. Use 'Configuration Builder' to create custom setups")

if __name__ == "__main__":
    main()
