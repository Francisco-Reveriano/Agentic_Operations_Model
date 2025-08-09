#!/usr/bin/env python3
"""
Launch script for the Truist GenAI Operations Q&A Chatbot
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit chatbot"""
    try:
        # Ensure we're in the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        print("🚀 Starting Truist GenAI Operations Q&A Chatbot...")
        print("📂 Working directory:", script_dir)
        print("🌐 The chatbot will open in your default web browser")
        print("🛑 Press Ctrl+C to stop the chatbot")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "Streamlit_Demo.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Chatbot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
