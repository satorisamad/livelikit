#!/usr/bin/env python3
"""
Voice Agent Configuration Tool

This script allows you to easily configure the voice agent's prompt and settings.
"""

import os
import sys
from agent_prompts import get_prompt, list_available_prompts, PROMPTS

def display_prompts():
    """Display all available prompts with descriptions"""
    print("\n" + "="*60)
    print("AVAILABLE VOICE AGENT PROMPTS")
    print("="*60)
    
    for name, prompt in PROMPTS.items():
        print(f"\n📝 {name.upper()}")
        print("-" * 40)
        # Show first few lines of the prompt
        lines = prompt.strip().split('\n')
        for line in lines[:3]:
            if line.strip():
                print(f"   {line.strip()}")
        if len(lines) > 3:
            print("   ...")
        print()

def update_agent_config(prompt_type: str):
    """Update the voice_agent.py file with the selected prompt type"""
    voice_agent_file = "voice_agent.py"
    
    if not os.path.exists(voice_agent_file):
        print(f"❌ Error: {voice_agent_file} not found in current directory")
        return False
    
    # Read the current file
    with open(voice_agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the prompt_type in AGENT_CONFIG
    import re
    pattern = r'("prompt_type":\s*")[^"]*(")'
    replacement = f'\\g<1>{prompt_type}\\g<2>'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print(f"⚠️  Warning: Could not find prompt_type configuration in {voice_agent_file}")
        return False
    
    # Write the updated file
    with open(voice_agent_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Successfully updated {voice_agent_file} to use '{prompt_type}' prompt")
    return True

def preview_prompt(prompt_type: str):
    """Preview a specific prompt"""
    try:
        prompt = get_prompt(prompt_type)
        print(f"\n📋 PREVIEW: {prompt_type.upper()} PROMPT")
        print("="*60)
        print(prompt)
        print("="*60)
    except KeyError:
        print(f"❌ Error: Prompt '{prompt_type}' not found")

def interactive_mode():
    """Interactive prompt selection"""
    while True:
        print("\n🎤 VOICE AGENT CONFIGURATION")
        print("="*40)
        print("1. 📋 View all available prompts")
        print("2. 👀 Preview a specific prompt")
        print("3. ⚙️  Configure agent with a prompt")
        print("4. 🚪 Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            display_prompts()
            
        elif choice == "2":
            prompt_type = input("Enter prompt name to preview: ").strip().lower()
            preview_prompt(prompt_type)
            
        elif choice == "3":
            available = list_available_prompts()
            print(f"\nAvailable prompts: {', '.join(available)}")
            prompt_type = input("Enter prompt name to use: ").strip().lower()
            
            if prompt_type in available:
                if update_agent_config(prompt_type):
                    print(f"\n🎉 Agent configured with '{prompt_type}' prompt!")
                    print("💡 Restart your voice agent to apply changes:")
                    print("   python voice_agent.py dev")
            else:
                print(f"❌ Error: '{prompt_type}' is not a valid prompt name")
                
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == "list":
            display_prompts()
            
        elif command == "preview" and len(sys.argv) > 2:
            prompt_type = sys.argv[2].lower()
            preview_prompt(prompt_type)
            
        elif command == "set" and len(sys.argv) > 2:
            prompt_type = sys.argv[2].lower()
            available = list_available_prompts()
            
            if prompt_type in available:
                if update_agent_config(prompt_type):
                    print(f"\n🎉 Agent configured with '{prompt_type}' prompt!")
                    print("💡 Restart your voice agent to apply changes:")
                    print("   python voice_agent.py dev")
            else:
                print(f"❌ Error: '{prompt_type}' is not a valid prompt name")
                print(f"Available prompts: {', '.join(available)}")
                
        else:
            print("Usage:")
            print("  python configure_agent.py list                    # List all prompts")
            print("  python configure_agent.py preview <prompt_name>   # Preview a prompt")
            print("  python configure_agent.py set <prompt_name>       # Set agent prompt")
            print("  python configure_agent.py                         # Interactive mode")
            
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
