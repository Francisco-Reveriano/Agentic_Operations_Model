import streamlit as st
import asyncio
import os
from datetime import datetime
import sys
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the q_a_agent from Master_Agent
from src.Agents.Master_Agent import q_a_agent

# Configure Streamlit page
st.set_page_config(
    page_title="Truist GenAI Operations Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

def clear_conversation():
    """Clear the conversation history"""
    st.session_state.messages = []
    st.rerun()

def format_agent_output(output_data: Dict[str, Any]) -> str:
    """Format the agent output into readable markdown"""
    markdown_content = []
    
    # Title
    markdown_content.append("# GenAI Impact Analysis Results\n")
    
    # Scenarios section
    markdown_content.append("## 📊 Scenario Analysis\n")
    
    # High Scenario
    markdown_content.append("### 🔴 High Impact Scenario")
    markdown_content.append(f"**Year-over-year changes:** {output_data.get('high_scenario', [])}")
    markdown_content.append(f"**Reasoning:** {output_data.get('high_scenario_reasoning', 'N/A')}\n")
    
    # Medium Scenario
    markdown_content.append("### 🟡 Medium Impact Scenario")
    markdown_content.append(f"**Year-over-year changes:** {output_data.get('medium_scenario', [])}")
    markdown_content.append(f"**Reasoning:** {output_data.get('medium_scenario_reasoning', 'N/A')}\n")
    
    # Low Scenario
    markdown_content.append("### 🟢 Low Impact Scenario")
    markdown_content.append(f"**Year-over-year changes:** {output_data.get('low_scenario', [])}")
    markdown_content.append(f"**Reasoning:** {output_data.get('low_scenario_reasoning', 'N/A')}\n")
    
    # Quality Assessment
    hallucination_score = output_data.get('hallucination_score', 'Unknown')
    emoji_map = {"Low": "✅", "Medium": "⚠️", "High": "❌"}
    emoji = emoji_map.get(hallucination_score, "❓")
    
    markdown_content.append("## 🎯 Quality Assessment")
    markdown_content.append(f"**Hallucination Risk:** {emoji} {hallucination_score}\n")
    
    return "\n".join(markdown_content)

async def process_user_query(user_input: str) -> Dict[str, Any]:
    """Process user query with the q_a_agent"""
    try:
        # Run the agent with the user input
        result = await q_a_agent.run(user_input)
        return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def display_message(message: Dict[str, str]):
    """Display a message in the chat interface"""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            st.write(message["content"])

# Sidebar
with st.sidebar:
    st.title("🤖 GenAI Q&A Assistant")
    st.markdown("---")
    
    st.markdown("### 💡 About")
    st.markdown("""
    This chatbot uses a specialized AI agent to analyze how Generative AI impacts banking operations and labor demand.
    
    **Features:**
    - Real-time analysis of GenAI impact scenarios
    - SEC filing research integration
    - Banking process automation assessment
    - Quality-checked responses
    """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Enter your query** about banking processes and GenAI impact
    2. **Include baseline metrics** (FTEs, costs, etc.) for better analysis
    3. **Specify the banking process** you want to analyze
    4. **Wait for analysis** - the agent will research and compute scenarios
    """)
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_conversation()
    
    st.markdown("---")
    
    # Display conversation count
    message_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
    st.metric("Questions Asked", message_count)

# Main interface
st.title("🏦 Truist GenAI Operations Analysis Chatbot")
st.markdown("Ask questions about how Generative AI will impact banking operations and labor demand.")

# Display conversation history
for message in st.session_state.messages:
    display_message(message)

# Chat input
if prompt := st.chat_input("Ask about GenAI impact on banking operations...", disabled=st.session_state.processing):
    # Add user message to conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process the query
    with st.chat_message("assistant"):
        # Create placeholders for dynamic updates
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        content_placeholder = st.empty()
        
        # Set processing state
        st.session_state.processing = True
        
        try:
            # Show initial status
            status_placeholder.info("🔍 Initializing analysis...")
            progress_bar = progress_placeholder.progress(0, text="Starting agent processing...")
            
            # Update progress during processing
            progress_bar.progress(20, text="🔎 Researching SEC filings...")
            
            # Run the async agent
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            progress_bar.progress(40, text="🧮 Running calculations...")
            
            try:
                result = loop.run_until_complete(process_user_query(prompt))
                progress_bar.progress(80, text="📝 Generating final analysis...")
                
                if result:
                    # Format and display the result
                    if hasattr(result, '__dict__'):
                        # Convert to dictionary if it's a Pydantic model
                        result_dict = result.__dict__ if hasattr(result, '__dict__') else result
                    else:
                        result_dict = result
                    
                    formatted_response = format_agent_output(result_dict)
                    
                    # Clear status and progress
                    status_placeholder.empty()
                    progress_placeholder.empty()
                    
                    # Display the formatted response
                    content_placeholder.markdown(formatted_response)
                    
                    # Add assistant response to conversation
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_response
                    })
                    
                    progress_bar.progress(100, text="✅ Analysis complete!")
                    
                else:
                    status_placeholder.error("❌ Failed to process query. Please try again.")
                    content_placeholder.markdown("Sorry, I encountered an error while processing your query. Please try rephrasing your question or check the system logs.")
                    
            finally:
                loop.close()
                
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            content_placeholder.markdown(f"**Error occurred:** {str(e)}\n\nPlease try again or contact support if the issue persists.")
            
        finally:
            # Reset processing state
            st.session_state.processing = False
            
            # Clear progress indicators after a short delay
            import time
            time.sleep(2)
            if 'progress_bar' in locals():
                progress_placeholder.empty()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 Powered by OpenAI Agents | Built for Truist Operations Analysis
</div>
""", unsafe_allow_html=True)

# Add some styling
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)
