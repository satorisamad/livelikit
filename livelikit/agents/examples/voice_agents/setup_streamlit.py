#!/usr/bin/env python3
"""
Setup Script for Streamlit Voice Agent Optimizer

This script helps set up the environment and dependencies for the
Streamlit voice agent optimization dashboard.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    requirements_file = "requirements_streamlit.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_environment_variables():
    """Check if required environment variables are set"""
    required_vars = [
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY", 
        "OPENAI_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nCreate a .env file with these variables or set them in your environment.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def create_sample_env_file():
    """Create a sample .env file"""
    env_content = """# Voice Agent Optimization Environment Variables

# Deepgram STT API
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Cartesia TTS API  
CARTESIA_API_KEY=your_cartesia_api_key_here

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Optional: Additional Configuration
VOICE_AGENT_LOG_LEVEL=INFO
VOICE_AGENT_METRICS_DIR=./metrics
"""
    
    env_file = ".env.example"
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"📝 Created sample environment file: {env_file}")
    print("   Copy this to .env and fill in your API keys")

def check_streamlit_installation():
    """Check if Streamlit is properly installed"""
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} is installed")
        return True
    except ImportError:
        print("❌ Streamlit is not installed")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    try:
        # Try to import the main components
        sys.path.insert(0, os.getcwd())
        
        # Test basic imports
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        
        print("✅ Core dependencies are working")
        
        # Test if our app can be imported
        try:
            import streamlit_voice_optimizer
            print("✅ Streamlit voice optimizer app can be imported")
            return True
        except ImportError as e:
            print(f"⚠️  Voice optimizer app import warning: {e}")
            print("   This is normal if voice agent modules are not available")
            return True
            
    except ImportError as e:
        print(f"❌ Dependency import error: {e}")
        return False

def create_launch_script():
    """Create a launch script for the Streamlit app"""
    launch_script_content = """#!/usr/bin/env python3
\"\"\"
Launch script for Voice Agent Optimizer Streamlit app
\"\"\"

import subprocess
import sys
import os

def main():
    # Set environment variables if .env file exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env")
    
    # Launch Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_voice_optimizer.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]
    
    print("🚀 Launching Voice Agent Optimizer...")
    print("   Open http://localhost:8501 in your browser")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down Voice Agent Optimizer")

if __name__ == "__main__":
    main()
"""
    
    with open("launch_streamlit.py", 'w') as f:
        f.write(launch_script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("launch_streamlit.py", 0o755)
    
    print("✅ Created launch script: launch_streamlit.py")

def main():
    """Main setup function"""
    print("🎙️ Voice Agent Optimizer Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if pip is available
    if not check_command_exists("pip"):
        print("❌ pip is not available")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Check Streamlit installation
    if not check_streamlit_installation():
        print("❌ Streamlit installation failed")
        sys.exit(1)
    
    # Test the app
    if not test_streamlit_app():
        print("❌ App testing failed")
        sys.exit(1)
    
    # Create sample environment file
    if not os.path.exists(".env") and not os.path.exists(".env.example"):
        create_sample_env_file()
    
    # Check environment variables
    env_ok = check_environment_variables()
    
    # Create launch script
    create_launch_script()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    
    if not env_ok:
        print("1. Set up your API keys in a .env file")
        print("   (see .env.example for template)")
    
    print("2. Launch the app:")
    print("   python launch_streamlit.py")
    print("   OR")
    print("   streamlit run streamlit_voice_optimizer.py")
    
    print("\n3. Open http://localhost:8501 in your browser")
    
    print("\n💡 Tips:")
    print("- Use the Configuration Builder to create custom setups")
    print("- Test different optimizations in Live Testing mode")
    print("- Analyze results in the Benchmarking section")
    print("- Check the Results Analysis for detailed insights")

if __name__ == "__main__":
    main()
