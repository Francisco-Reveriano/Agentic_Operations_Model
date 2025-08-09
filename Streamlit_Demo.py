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

from agents import Runner, set_trace_processors, trace
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor
import weave

# Configure Streamlit page
st.set_page_config(
    page_title="GenAI Operations Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load selected environment variables from Streamlit secrets (overrides .env)
def _load_env_from_streamlit_secrets():
    try:
        # Copy all keys exactly as named from st.secrets to environment without renaming
        for key, value in st.secrets.items():
            if value is not None:
                os.environ[str(key)] = str(value)
    except Exception:
        # If st.secrets is not configured, just proceed
        pass

_load_env_from_streamlit_secrets()

# Initialize Weave tracing immediately and unconditionally
weave.init("Streamlit_Operations_Model")
set_trace_processors([WeaveTracingProcessor()])

# Import the q_a_agent from Master_Agent AFTER env is set
from src.Agents.Master_Agent import q_a_agent

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
    
    # Online coursework recommendations
    courses = output_data.get('online_coursework', []) or []
    markdown_content.append("## 📚 Recommended Online Coursework")
    if isinstance(courses, list) and len(courses) > 0:
        for course in courses:
            markdown_content.append(f"- {course}")
    else:
        markdown_content.append("_No coursework recommendations returned._")
    
    
    return "\n".join(markdown_content)

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
st.title("🏦 Client GenAI Operations Analysis Chatbot")
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
            
            # Indicate calculation phase
            progress_bar.progress(40, text="🧮 Running calculations...")
            
            # Use OpenAI Agent SDK Runner to execute the agent
            with st.spinner("Running agent with OpenAI Agent SDK..."):
                # Always trace the chat request; WeaveTracingProcessor captures model/tool events
                with trace("streamlit.chat_request"):
                    result = asyncio.run(Runner.run(q_a_agent, prompt, max_turns=100))

            progress_bar.progress(80, text="📝 Generating final analysis...")
            
            if result is not None:
                # Extract the agent's structured output
                output_obj = getattr(result, "final_output", result)

                if hasattr(output_obj, "dict"):
                    result_dict = output_obj.dict()
                elif hasattr(output_obj, "model_dump"):
                    result_dict = output_obj.model_dump()
                elif isinstance(output_obj, dict):
                    result_dict = output_obj
                else:
                    # Fallback: build a dict from known fields if available
                    candidate_fields = [
                        "high_scenario",
                        "medium_scenario",
                        "low_scenario",
                        "high_scenario_reasoning",
                        "medium_scenario_reasoning",
                        "low_scenario_reasoning",
                        "hallucination_score",
                    ]
                    result_dict = {
                        field: getattr(output_obj, field)
                        for field in candidate_fields
                        if hasattr(output_obj, field)
                    }
                
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
    🤖 Powered by OpenAI Agents | Built for Client Operations Analysis
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
