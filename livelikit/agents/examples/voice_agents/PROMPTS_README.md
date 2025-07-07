# Voice Agent Prompts System

This enhanced voice agent now supports multiple personality prompts that can be easily configured without modifying the main code.

## 📁 Files Overview

- **`agent_prompts.py`** - Contains all available prompts and utility functions
- **`configure_agent.py`** - Interactive tool to configure the agent's prompt
- **`voice_agent.py`** - Main voice agent (now uses prompts from `agent_prompts.py`)

## 🎭 Available Prompts

### **Default Prompts:**

1. **`default`** - Friendly AI assistant named Shabana
2. **`professional`** - Professional business assistant
3. **`casual`** - Casual friend companion
4. **`educational`** - Patient educational tutor
5. **`technical`** - Technical expert assistant
6. **`energetic`** - Upbeat and enthusiastic assistant
7. **`calm`** - Soothing and relaxed assistant
8. **`health`** - Health and wellness focused (general advice only)
9. **`creative`** - Creative and artistic assistant

## 🚀 Quick Start

### Method 1: Interactive Configuration (Recommended)
```bash
python configure_agent.py
```

This will show you a menu to:
- View all available prompts
- Preview specific prompts
- Configure the agent with your chosen prompt

### Method 2: Command Line Configuration
```bash
# List all available prompts
python configure_agent.py list

# Preview a specific prompt
python configure_agent.py preview casual

# Set the agent to use a specific prompt
python configure_agent.py set energetic
```

### Method 3: Manual Configuration
Edit the `AGENT_CONFIG` in `voice_agent.py`:
```python
AGENT_CONFIG = {
    "prompt_type": "casual",  # Change this to any available prompt
    "llm_model": "gpt-4o-mini",
    "tts_model": "sonic-2",
    # ... other settings
}
```

## 🔄 Applying Changes

After changing the prompt configuration:
```bash
# Restart your voice agent
python voice_agent.py dev
```

The agent will log which prompt it's using:
```
[CONFIG] Using prompt type: 'casual'
```

## ➕ Adding Custom Prompts

### Option 1: Add to `agent_prompts.py`
```python
# Add your custom prompt to the PROMPTS dictionary
CUSTOM_PROMPT = """You are a specialized AI assistant for [your use case].

Guidelines:
- [Your specific guidelines]
- [Behavior instructions]
- [Tone and style]
"""

PROMPTS["custom"] = CUSTOM_PROMPT
```

### Option 2: Programmatically add prompts
```python
from agent_prompts import add_custom_prompt

add_custom_prompt("my_prompt", """
Your custom prompt text here...
""")
```

## 🎯 Prompt Examples

### **Casual Friend** (`casual`)
```
You are a friendly AI companion named Shabana. Your interface with users is voice-based.

Guidelines:
- Be warm, friendly, and conversational
- Use casual language and expressions
- Show interest in the user's day and activities
- Be supportive and encouraging
- Keep the conversation light and enjoyable
```

### **Professional** (`professional`)
```
You are a professional AI assistant. Your interface with users is voice-based.

Guidelines:
- Maintain a professional and courteous tone
- Provide clear, concise information
- Be helpful and efficient
- Ask clarifying questions when needed
- Keep responses appropriate for business contexts
```

### **Energetic** (`energetic`)
```
You are an energetic and enthusiastic AI assistant named Shabana. Your interface with users is voice-based.

Guidelines:
- Be upbeat and positive
- Use exclamation points and enthusiastic language
- Encourage and motivate the user
- Show excitement about topics
- Keep the energy high and engaging
```

## 🛠️ Advanced Configuration

### Modify Agent Settings
You can also configure other aspects in `voice_agent.py`:
```python
AGENT_CONFIG = {
    "prompt_type": "default",
    "llm_model": "gpt-4o-mini",        # Change LLM model
    "tts_model": "sonic-2",            # Change TTS model
    "tts_sample_rate": 24000,          # Audio quality
    "tts_encoding": "pcm_s16le"        # Audio encoding
}
```

### Runtime Prompt Selection
You can also pass the prompt type when creating the agent:
```python
# In your custom code
agent = MyAgent(prompt_type="energetic")
```

## 📊 Testing Different Prompts

1. **Configure the prompt**:
   ```bash
   python configure_agent.py set casual
   ```

2. **Start the agent**:
   ```bash
   python voice_agent.py dev
   ```

3. **Test the conversation** and observe the personality differences

4. **Analyze metrics** (if using the enhanced metrics system):
   ```bash
   python metrics_analyzer.py voice_agent_metrics.csv
   ```

## 💡 Tips for Creating Effective Prompts

1. **Be Specific**: Clearly define the agent's role and behavior
2. **Voice-Optimized**: Remember this is for voice interaction - keep responses conversational
3. **Consistent Tone**: Maintain the same personality throughout the prompt
4. **Clear Guidelines**: Provide specific instructions for different scenarios
5. **Test Thoroughly**: Try different conversation types to ensure the prompt works well

## 🔍 Troubleshooting

### Prompt Not Found Error
```
[CONFIG] Prompt 'xyz' not found. Available prompts: default, casual, professional...
[CONFIG] Falling back to default prompt
```
**Solution**: Use one of the available prompt names or add your custom prompt to `agent_prompts.py`

### Configuration Not Applied
**Solution**: Make sure to restart the voice agent after changing the configuration:
```bash
python voice_agent.py dev
```

### Want to Reset to Default
```bash
python configure_agent.py set default
```

## 🎉 Example Usage

```bash
# Try the energetic assistant
python configure_agent.py set energetic
python voice_agent.py dev

# Switch to professional mode
python configure_agent.py set professional  
python voice_agent.py dev

# Go back to casual friend
python configure_agent.py set casual
python voice_agent.py dev
```

Each prompt will give your voice agent a completely different personality and interaction style!
