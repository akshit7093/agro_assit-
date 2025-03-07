"""Streamlit-based user interface for the Multi-Functional AI Assistant."""
import datetime
from typing import Coroutine
import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Multi-Functional AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
) 
import streamlit as st
import os
import sys
import asyncio
import time
import json
import tempfile
import io  # Add this import at the top level
from typing import Dict, Any, List
import cv2
from PIL import Image
import numpy as np
# Add PyPDF2 import
import PyPDF2
from loguru import logger  # Add logger import

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import AIAssistant
# Import cleanup utility
from utils.cleanup import clean_temp_directory

VISION_MODELS = ["gemini", "qwen"]

# Clean temporary files on startup
clean_temp_directory(temp_dir="data/temp")

# Add this import at the top of the file
from nlp.document_processor import process_document

# Then find the document upload section and update it:

# Document upload section
uploaded_document = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])
if uploaded_document is not None:
    # Save the uploaded document
    doc_path = os.path.join("data/temp", uploaded_document.name)
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "wb") as f:
        f.write(uploaded_document.getbuffer())
    
    # Process the document
    doc_result = process_document(doc_path)
    
    if "error" in doc_result:
        st.error(f"Error processing document: {doc_result['error']}")
    else:
        st.success(f"Document processed: {doc_result['file_name']}")
        
        # Store document content in session state for later use
        st.session_state.last_document_data = doc_result
        
        # Display document content preview (first 500 chars)
        st.subheader("Document Preview")
        preview = doc_result['content'][:500] + "..." if len(doc_result['content']) > 500 else doc_result['content']
        st.text_area("Content Preview", preview, height=200)
        
        # Add a button to ask about the document
        if st.button("Ask about this document", key="ask_doc_btn"):
            st.session_state.messages.append({
                "role": "user", 
                "content": f"I've uploaded a document named '{doc_result['file_name']}'. Please analyze it and tell me the key points or information contained in it."
            })
            st.rerun()

async def init_assistant():
    try:
        # Create a new event loop for async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize the AIAssistant synchronously
        assistant = AIAssistant()
        
        return assistant
    except Exception as e:
        st.error(f"Error initializing assistant: {str(e)}")
        return None

def init_session_state():
    # Initialize session_id first
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    
    # Initialize other basic state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "video_summary" not in st.session_state:
        st.session_state.video_summary = None
    
    # Initialize assistant last
    if "assistant" not in st.session_state:
        try:
            # Create and initialize the assistant synchronously
            assistant = AIAssistant()
            
            # Check if assistant was created successfully
            if assistant is None:
                st.error("Failed to initialize AI Assistant")
                return
            
            # Store in session state
            st.session_state.assistant = assistant
            
            # Set session_id in assistant
            if hasattr(st.session_state.assistant, "session_manager"):
                st.session_state.assistant.session_manager.current_session_id = st.session_state.session_id
                
        except Exception as e:
            st.error(f"Error initializing assistant: {str(e)}")
            return
async def process_task(text: str, task: str, model_id: str) -> Dict[str, Any]:
    try:
        # Add coroutine validation
        if not hasattr(st.session_state.assistant, "process_text") or \
           not asyncio.iscoroutinefunction(st.session_state.assistant.process_text):
            return {"error": "Text processing not initialized"}
            
        coro = st.session_state.assistant.process_text(text, task, {"model_name": model_id})
        if not asyncio.iscoroutine(coro):
            return {"error": "Invalid text processing coroutine"}
            
        return coro
    except Exception as e:
        st.error(f"Error processing task: {str(e)}")
        return {"error": str(e)}

def get_async_response(coro):
    try:
        if not asyncio.iscoroutine(coro):
            st.error("Invalid coroutine passed to async response handler")
            return None

        # Existing loop handling code...
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        st.error(f"Error in async execution: {str(e)}")
        return None



async def process_chat(prompt: str, model_id: str, image_path: str = None) -> Dict[str, Any]:
    try:
        # Ensure prompt is not None before any processing
        if prompt is None:
            prompt = "Hello, how can I help you today?"

        # Verify assistant is initialized
        if not hasattr(st.session_state, "assistant") or st.session_state.assistant is None:
            return {"error": "Assistant not properly initialized. Please refresh the page."}
            
        # Get current session ID
        current_session_id = st.session_state.session_id
        
        # Save conversation to document store
        if hasattr(st.session_state.assistant, "document_store"):
            # Create conversation document
            conversation_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in st.session_state.messages[-5:]  # Get last 5 messages for context
            ])
            
            conversation_id = f"conversation_{current_session_id}_{int(time.time())}"
            
            # Add to document store
            try:
                st.session_state.assistant.document_store.add_document(
                    conversation_id,
                    conversation_text,
                    metadata={
                        "type": "conversation",
                        "session_id": current_session_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model": model_id
                    }
                )
            except AttributeError:
                if hasattr(st.session_state.assistant.document_store, 'add_texts'):
                    st.session_state.assistant.document_store.add_texts(
                        [conversation_text],
                        metadatas=[{
                            "type": "conversation",
                            "session_id": current_session_id,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "model": model_id
                        }],
                        ids=[conversation_id]
                    )
        
        # Check if we have an image to process
        if image_path and os.path.exists(image_path) and model_id in VISION_MODELS:
            # Process image chat
            try:
                # Get the model from the model factory
                model = st.session_state.assistant.model_factory.get_model(model_id)
                
                # Check if model supports vision
                if not hasattr(model, 'generate_with_image'):
                    return {"error": f"Model {model_id} does not support image analysis"}
                
                # Use the generate_with_image method
                response = await model.generate_with_image(prompt, image_path)
                
                # Return the response text
                return response.text if hasattr(response, 'text') else str(response)
                
            except Exception as img_error:
                logger.error(f"Image chat error: {str(img_error)}")
                return {"error": f"Error processing image: {str(img_error)}"}
        
        # Text-only chat
        try:
            # Verify chat method exists
            if not hasattr(st.session_state.assistant, "chat") or \
               not asyncio.iscoroutinefunction(st.session_state.assistant.chat):
                st.error("Chat method not properly initialized")
                return {"error": "Chat functionality unavailable"}
            
            # Get the chat coroutine first
            chat_coroutine = st.session_state.assistant.chat(
                prompt,
                {
                    "model_name": model_id,
                    "temperature": st.session_state.temperature if hasattr(st.session_state, "temperature") else 0.7,
                    "max_tokens": st.session_state.max_tokens if hasattr(st.session_state, "max_tokens") else 1024,
                    "top_p": st.session_state.top_p if hasattr(st.session_state, "top_p") else 0.95,
                    "frequency_penalty": st.session_state.frequency_penalty if hasattr(st.session_state, "frequency_penalty") else 0.0,
                    "presence_penalty": st.session_state.presence_penalty if hasattr(st.session_state, "presence_penalty") else 0.0
                }
            )

            # Check if we actually got a coroutine
            if not asyncio.iscoroutine(chat_coroutine):
                st.error(f"Chat method returned invalid type: {type(chat_coroutine)}")
                return {"error": "Internal chat error"}

            # Now await the coroutine
            response = await chat_coroutine
            
            if response is None:
                return {"error": "No response generated from the model"}
                
            return response
            
        except Exception as chat_error:
            st.error(f"Chat error: {str(chat_error)}")
            return {"error": str(chat_error)}
            
    except Exception as e:
        st.error(f"Error in chat: {str(e)}")
        return {"error": str(e)}

async def process_image_chat(prompt: str, model_id: str, image_path: str, model_params: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        if model_params is None:
            model_params = {}
            
        model = st.session_state.assistant.model_factory.get_model(model_id)
        if not model or not hasattr(model, 'generate_with_image'):
            return {"error": f"Model {model_id} not initialized for images"}
            
        # Get the coroutine
        coro = model.generate_with_image(prompt, image_path)
        if not asyncio.iscoroutine(coro):
            return {"error": "Invalid image processing coroutine"}
            
        # Await the coroutine to get the actual response
        response = await coro
        
        # Return the response text
        return {
            "text": response.text if hasattr(response, 'text') else str(response),
            "model": model_id
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Image chat error: {str(e)}\n{error_details}")
        return {"error": f"Image chat error: {str(e)}"}

async def process_video_frames(video_path: str, model_id: str) -> Dict[str, Any]:
    """Process video by extracting frames and analyzing with vision model."""
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        # Calculate frame interval (process a frame every 0.5 seconds)
        frame_interval = max(1, int(fps * 0.5))
        
        # Initialize variables to store processed frames
        frames_data = []
        frame_descriptions = []
        frames_processed = 0
        
        # Create progress indicators
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Process frames at regular intervals
        for frame_idx in range(0, frame_count, frame_interval):
            # Set the video position
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            
            if not ret:
                continue
                
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Save frame as image
            temp_dir = os.path.join("data", "temp", "frames")
            os.makedirs(temp_dir, exist_ok=True)
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Analyze frame with vision model with retry logic
            max_retries = 3
            retry_delay = 1.0  # Start with 1 second delay
            
            for retry in range(max_retries):
                try:
                    description = await analyze_image_with_model(frame_path, model_id)
                    break  # If successful, break out of retry loop
                except Exception as e:
                    if "429" in str(e) or "Quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if retry < max_retries - 1:
                            logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            description = f"[Frame analysis limited due to API rate limits]"
                    else:
                        description = f"[Error analyzing frame: {str(e)}]"
                        break
            
            frame_descriptions.append(description)
            
            # Store frame data
            frames_data.append({
                "index": frame_idx,
                "timestamp": timestamp,
                "image_path": frame_path,
                "description": description
            })
            
            frames_processed += 1
            
            # Add a delay between API calls to avoid rate limits
            # More substantial delay for video processing
            await asyncio.sleep(1.0)  # 1 second delay between frames
        
        # Close the video file
        video.release()
        
        # Rest of the function remains the same...
        
        # Generate a summary of the video content
        # ... rest of the function remains the same
        
        # Rest of the function remains the same...
        
        # Update progress to completion
        if 'progress_text' in locals() and 'progress_bar' in locals():
            progress_text.empty()
            progress_bar.empty()
        
        # Generate a summary of the video content
        summary = ""
        if frame_descriptions:
            try:
                # Limit the number of descriptions to avoid token limits
                max_frames_for_summary = min(20, len(frame_descriptions))
                selected_descriptions = frame_descriptions[:max_frames_for_summary]
                
                # Join selected descriptions and ask the model to summarize
                all_descriptions = "\n".join([f"Frame {i+1}: {desc}" for i, desc in enumerate(selected_descriptions)])
                summary_prompt = f"The following are descriptions of key frames from a video. Please provide a concise summary of what's happening in the video based on these descriptions:\n\n{all_descriptions}"
                
                # Get summary from model
                summary = await process_chat(summary_prompt, model_id)
            except Exception as e:
                # Handle summary generation error gracefully
                logger.error(f"Error generating video summary: {str(e)}")
                summary = "Unable to generate video summary. You can ask about specific frames in the Video Frames tab."
        
        # Return the processed data
        return {
            "frames": frames_data,
            "summary": summary,
            "duration": duration,
            "total_frames": frame_count,
            "processed_frames": frames_processed,
            "fps": fps
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"error": f"Video processing failed: {str(e)}\nDetails: {error_details}"}

async def analyze_image_with_model(image_data, model_id):
    """Analyze image content using the selected vision model."""
    if image_data is None:
        logger.error("No image data provided")
        return "Image analysis failed: No image data provided"

    if not model_id:
        logger.error("No model ID provided")
        return "Image analysis failed: No model ID provided"

    temp_image_path = None
    try:
        # Save the image to a temporary file
        temp_dir = os.path.join("data", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        temp_image_path = os.path.join(temp_dir, f"temp_image_{timestamp}_{uuid.uuid4()}.jpg")
        
        # Convert BytesIO to file
        if isinstance(image_data, io.BytesIO):
            # Save BytesIO to file
            with open(temp_image_path, "wb") as f:
                f.write(image_data.getvalue())
        else:
            # If it's already a file path, just use it
            if isinstance(image_data, str) and os.path.exists(image_data):
                temp_image_path = image_data
                logger.info(f"Using existing image path: {temp_image_path}")
            else:
                # Handle PIL Image or other formats
                try:
                    from PIL import Image
                    if isinstance(image_data, Image.Image):
                        image_data.save(temp_image_path, format="JPEG")
                    else:
                        # Try to convert to PIL Image
                        Image.open(image_data).save(temp_image_path, format="JPEG")
                except Exception as e:
                    logger.error(f"Error saving image: {str(e)}")
                    return f"Image analysis failed: Error saving image: {str(e)}"
        
        # Verify the image file exists before proceeding
        if not os.path.exists(temp_image_path):
            return f"Image analysis failed: Image file not found at {temp_image_path}"
            
        # Get the model from the model factory
        model = st.session_state.assistant.model_factory.get_model(model_id)
        
        # Check if model supports vision
        if model_id not in VISION_MODELS:
            return f"Model {model_id} does not support image analysis. Please use one of: {', '.join(VISION_MODELS)}"
        
        # Analyze the image
        prompt = "Analyze this image in detail. Describe what you see, including objects, people, scenes, colors, and any text visible in the image."
        
        # Use the generate_with_image method
        response = await model.generate_with_image(prompt, temp_image_path)
        
        # Return the response text
        if hasattr(response, 'text'):
            return response.text
        else:
            return str(response)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error analyzing image with model: {str(e)}\n{error_details}")
        st.error(f"Error analyzing image: {str(e)}")
        return f"Image analysis failed: {str(e)}\nDetails: {error_details}"
    finally:
        # Clean up temporary files if needed
        # Only delete if we created a new temp file and not using an existing path
        if temp_image_path and temp_image_path != image_data and isinstance(image_data, str) and temp_image_path != image_data:
            try:
                # Uncomment to enable cleanup
                # os.remove(temp_image_path)
                pass
            except Exception:
                pass

def main():

    init_session_state()

    current_session_id=st.session_state.session_id
    st.title("Multi-Functional AI Assistant")
    st.markdown("""
    A powerful AI assistant that can help you with:
    - Text Summarization
    - Sentiment Analysis
    - Named Entity Recognition
    - Question Answering
    - Code Generation
    - Contextual Chat Interface
    """)

    with st.sidebar:
        st.header("Settings")
        
        # Create tabs for different settings categories
        settings_tabs = st.tabs(["Model", "Parameters", "Persona", "Documents", "Video Frames"])
        
        # Model Settings Tab
        with settings_tabs[0]:
            # Create model options
            all_models = [
                {"id": "gemini", "name": "Gemini Flash Lite", "has_vision": True},
                {"id": "deepseek", "name": "DeepSeek R1", "has_vision": False},
                {"id": "qwen", "name": "Qwen2.5 VL", "has_vision": True},
                {"id": "NVIDIA", "name": "NVIDIA Llama", "has_vision": False}
            ]
            
            # Select cloud model
            cloud_model_names = [m["name"] for m in all_models]
            selected_cloud_model = st.selectbox(
                "Select Model",
                cloud_model_names,
                index=cloud_model_names.index(st.session_state.get("selected_cloud_model", "Gemini Flash Lite")) 
                    if "selected_cloud_model" in st.session_state else 0
            )
            st.session_state.selected_cloud_model = selected_cloud_model
            
            # Find the selected model details
            selected_model = next((m for m in all_models if m["name"] == selected_cloud_model), None)
            if selected_model:
                model_id = selected_model["id"]
                st.session_state.selected_model_id = model_id
                st.session_state.is_vision_model = selected_model["has_vision"]  # Store vision capability
                
                if selected_model["has_vision"]:
                    st.info(f"📷 {selected_cloud_model} supports image/video analysis")

            # Now define is_vision based on the selected model's capabilities
            is_vision = st.session_state.get("is_vision_model", False)
            
        
        # Parameters Tab
        with settings_tabs[1]:
            # Add temperature slider
            if "temperature" not in st.session_state:
                st.session_state.temperature = 0.7
            
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
            
            # Add max tokens slider
            if "max_tokens" not in st.session_state:
                st.session_state.max_tokens = 1024
            
            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                min_value=256,
                max_value=4096,
                value=st.session_state.max_tokens,
                step=128,
                help="Maximum number of tokens in the response"
            )
            
            # Add top_p slider
            if "top_p" not in st.session_state:
                st.session_state.top_p = 0.95
            
            st.session_state.top_p = st.slider(
                "Top P",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.top_p,
                step=0.05,
                help="Controls diversity via nucleus sampling"
            )
            
            # Add frequency penalty
            if "frequency_penalty" not in st.session_state:
                st.session_state.frequency_penalty = 0.0
            
            st.session_state.frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.frequency_penalty,
                step=0.1,
                help="Penalizes frequent tokens. Positive values decrease repetition."
            )
            
            # Add presence penalty
            if "presence_penalty" not in st.session_state:
                st.session_state.presence_penalty = 0.0
            
            st.session_state.presence_penalty = st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.presence_penalty,
                step=0.1,
                help="Penalizes tokens already used. Positive values encourage new topics."
            )
            
            # Apply settings button
            if st.button("Apply Settings"):
                st.success("Settings applied!")
        
        # Persona Tab
        with settings_tabs[2]:
            # Predefined personas
            personas = {
                "Default Assistant": "I am a helpful AI assistant.",
                "Technical Expert": "I am a technical expert specializing in software development, data science, and AI. I provide detailed technical explanations and code examples.",
                "Creative Writer": "I am a creative writing assistant with a flair for storytelling, poetry, and creative content. I can help craft engaging narratives and artistic text.",
                "Academic Researcher": "I am an academic research assistant with expertise in scholarly writing, citation, and research methodologies across various disciplines.",
                "Business Consultant": "I am a business consultant with expertise in strategy, marketing, finance, and operations. I provide professional business advice and analysis.",
                "Custom": "Custom persona..."
            }
            
            # Persona selection
            if "selected_persona" not in st.session_state:
                st.session_state.selected_persona = "Default Assistant"
                st.session_state.persona_description = personas["Default Assistant"]
            
            selected_persona = st.selectbox(
                "Select Persona",
                list(personas.keys()),
                index=list(personas.keys()).index(st.session_state.selected_persona) if st.session_state.selected_persona in personas else 0
            )
            
            # Handle custom persona
            if selected_persona == "Custom":
                custom_persona = st.text_area(
                    "Define Custom Persona",
                    value=st.session_state.persona_description if st.session_state.selected_persona == "Custom" else "",
                    height=150,
                    help="Describe the persona you want the AI to adopt"
                )
                if custom_persona:
                    st.session_state.persona_description = custom_persona
            else:
                st.session_state.persona_description = personas[selected_persona]
            
            st.session_state.selected_persona = selected_persona
            
            # Display current persona
            st.info(f"**Current Persona:**\n\n{st.session_state.persona_description}")
            
            # Apply persona button
            if st.button("Apply Persona"):
                # Add system message to conversation history
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"[Persona changed to: {selected_persona}]"
                })
                
                # Add a notification message from the assistant about the persona change
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I'll now continue our conversation as {selected_persona}. How can I assist you?"
                })
                
                st.success(f"Persona changed to: {selected_persona}")
                st.rerun()  # Refresh to show the new message
        
        # Document Upload Tab
        with settings_tabs[3]:
            st.header("Document Upload (RAG)")
            
            # Make the direct file path option more prominent
            st.markdown("### 🚀 Unlimited File Size Option")
            st.info("**Use this method for files of ANY size:**")
            
            # Create buttons for common file types
            st.write("Select file type to upload:")
            col1, col2, col3, col4 = st.columns(4)
            
            file_path = None
            
            with col1:
                if st.button("📄 Document", key="doc_btn"):
                    import tkinter as tk
                    from tkinter import filedialog
                    
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(
                        title="Select Document",
                        filetypes=[("Documents", "*.pdf;*.docx;*.txt")]
                    )
                    root.destroy()
            
            with col2:
                if st.button("🎬 Video", key="video_btn"):
                    import tkinter as tk
                    from tkinter import filedialog
                    
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(
                        title="Select Video",
                        filetypes=[("Videos", "*.mp4;*.avi;*.mov;*.mkv")]
                    )
                    root.destroy()
            
            with col3:
                if st.button("📊 Data", key="data_btn"):
                    import tkinter as tk
                    from tkinter import filedialog
                    
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(
                        title="Select Data File",
                        filetypes=[("Data Files", "*.csv;*.json;*.xlsx")]
                    )
                    root.destroy()
            
            with col4:
                if st.button("📁 Any File", key="any_btn"):
                    import tkinter as tk
                    from tkinter import filedialog
                    
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(title="Select Any File")
                    root.destroy()
            
            # Store the selected file path in session state
            if file_path:
                st.session_state.selected_file_path = file_path
            
            # Display selected file info if available
            if hasattr(st.session_state, "selected_file_path") and st.session_state.selected_file_path:
                direct_file_path = st.session_state.selected_file_path
                
                if os.path.exists(direct_file_path):
                    file_name = os.path.basename(direct_file_path)
                    file_size_mb = os.path.getsize(direct_file_path) / (1024 * 1024)
                    st.success(f"Selected file: {file_name} ({file_size_mb:.2f} MB)")
                    
                    # Show file path with option to edit
                    edited_path = st.text_input("File path (edit if needed):", value=direct_file_path)
                    st.session_state.selected_file_path = edited_path
                    
                    if st.button("Process Selected File", key="process_selected"):
                        if os.path.exists(edited_path):
                            file_name = os.path.basename(edited_path)
                            with st.spinner(f"Processing {file_name}..."):
                                # Copy file to data directory
                                dest_path = os.path.join("data", "raw", file_name)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                
                                try:
                                    # For very large files, use shutil.copy instead of reading into memory
                                    import shutil
                                    shutil.copy2(edited_path, dest_path)
                                    
                                    # Process the file based on its type
                                    # Rest of your file processing code remains the same
                                    if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and is_vision:
                                        st.info(f"Processing video: {file_name}")
                                        # Process video frames
                                        video_result = get_async_response(process_video_frames(dest_path, model_id))
                                        st.session_state.last_video_data = video_result
                                        
                                        # Add video data to RAG system
                                        if "frames" in video_result and video_result["frames"]:
                                            # Create a text representation of the video content
                                            video_text = f"Video file: {file_name}\n"
                                            video_text += f"Duration: {video_result.get('duration', 0):.2f} seconds\n"
                                            video_text += f"Total frames: {video_result.get('total_frames', 0)}\n"
                                            video_text += f"Processed frames: {video_result.get('processed_frames', 0)}\n\n"
                                            
                                            # Add summary if available
                                            if "summary" in video_result and video_result["summary"]:
                                                video_text += f"Video Summary:\n{video_result['summary']}\n\n"
                                            
                                            # Add frame information
                                            video_text += "Frame Descriptions:\n"
                                            for i, frame in enumerate(video_result["frames"]):
                                                video_text += f"Frame {i+1} at {frame.get('timestamp', 0):.2f}s: {frame.get('description', 'No description')}\n"
                                            
                                            # Add to document store
                                            try:
                                                # Try add_document method first
                                                st.session_state.assistant.document_store.add_document(
                                                    file_name,
                                                    video_text,
                                                    metadata={"source": dest_path, "type": "video", "frames": len(video_result["frames"])}
                                                )
                                            except AttributeError:
                                                # If add_document doesn't exist, try alternative methods
                                                if hasattr(st.session_state.assistant.document_store, 'add'):
                                                    st.session_state.assistant.document_store.add(
                                                        file_name,
                                                        video_text,
                                                        metadata={"source": dest_path, "type": "video", "frames": len(video_result["frames"])}
                                                    )
                                                elif hasattr(st.session_state.assistant.document_store, 'add_texts'):
                                                    st.session_state.assistant.document_store.add_texts(
                                                        [video_text],
                                                        metadatas=[{"source": dest_path, "type": "video", "frames": len(video_result["frames"])}],
                                                        ids=[file_name]
                                                    )
                                                else:
                                                    raise AttributeError("Could not find a compatible method to add documents to the document store")
                                        
                                        st.success(f"Successfully processed video {file_name} and added to knowledge base")
                                    elif file_name.lower().endswith('.pdf'):
                                        st.info(f"Processing PDF: {file_name}")
                                        # Process PDF directly from file path
                                        try:
                                            # Create a PDF file reader
                                            pdf_reader = PyPDF2.PdfReader(dest_path)
                                            
                                            # Extract text from all pages
                                            pdf_text = ""
                                            for page_num in range(len(pdf_reader.pages)):
                                                page = pdf_reader.pages[page_num]
                                                pdf_text += page.extract_text() + "\n\n"
                                            
                                            # Add to document store if text was extracted
                                            if pdf_text.strip():
                                                # Use the correct method to add document to the store
                                                try:
                                                    # Try add_document method first (most common implementation)
                                                    st.session_state.assistant.document_store.add_document(
                                                        file_name,
                                                        pdf_text,
                                                        metadata={"source": dest_path, "type": "pdf", "pages": len(pdf_reader.pages)}
                                                    )
                                                except AttributeError:
                                                    # If add_document doesn't exist, try alternative methods
                                                    if hasattr(st.session_state.assistant.document_store, 'add'):
                                                        st.session_state.assistant.document_store.add(
                                                            file_name,
                                                            pdf_text,
                                                            metadata={"source": dest_path, "type": "pdf", "pages": len(pdf_reader.pages)}
                                                        )
                                                    elif hasattr(st.session_state.assistant.document_store, 'add_texts'):
                                                        st.session_state.assistant.document_store.add_texts(
                                                            [pdf_text],
                                                            metadatas=[{"source": dest_path, "type": "pdf", "pages": len(pdf_reader.pages)}],
                                                            ids=[file_name]
                                                        )
                                                    else:
                                                        raise AttributeError("Could not find a compatible method to add documents to the document store")
                                                
                                                st.success(f"Added PDF {file_name} ({len(pdf_reader.pages)} pages) to knowledge base")
                                            else:
                                                st.warning(f"Could not extract text from PDF {file_name}")
                                        except Exception as pdf_error:
                                            st.error(f"Error processing PDF {file_name}: {str(pdf_error)}")
                                    elif file_name.lower().endswith('.docx'):
                                        st.info(f"Processing DOCX: {file_name}")
                                        # Process DOCX directly from file path
                                        try:
                                            # Import docx library
                                            import docx
                                            
                                            # Read the document
                                            doc = docx.Document(dest_path)
                                            
                                            # Extract text from paragraphs
                                            docx_text = "\n".join([para.text for para in doc.paragraphs])
                                            
                                            # Add to document store
                                            # Add to document store
                                            try:
                                                # Try add_document method first
                                                st.session_state.assistant.document_store.add_document(
                                                    file_name,
                                                    docx_text,
                                                    metadata={"source": dest_path, "type": "docx"}
                                                )
                                            except AttributeError:
                                                # If add_document doesn't exist, try alternative methods
                                                if hasattr(st.session_state.assistant.document_store, 'add'):
                                                    st.session_state.assistant.document_store.add(
                                                        file_name,
                                                        docx_text,
                                                        metadata={"source": dest_path, "type": "docx"}
                                                    )
                                                elif hasattr(st.session_state.assistant.document_store, 'add_texts'):
                                                    st.session_state.assistant.document_store.add_texts(
                                                        [docx_text],
                                                        metadatas=[{"source": dest_path, "type": "docx"}],
                                                        ids=[file_name]
                                                    )
                                                else:
                                                    raise AttributeError("Could not find a compatible method to add documents to the document store")
                                            st.success(f"Added DOCX {file_name} to knowledge base")
                                        except Exception as docx_error:
                                            st.error(f"Error processing DOCX {file_name}: {str(docx_error)}")
                                    elif file_name.lower().endswith(('.txt', '.csv', '.json')):
                                        st.info(f"Processing text file: {file_name}")
                                        try:
                                            # Read text file
                                            with open(dest_path, 'r', encoding='utf-8') as f:
                                                text_content = f.read()
                                            
                                            # Add to document store
                                            st.session_state.assistant.document_store.add_document_sync(
                                                file_name,
                                                text_content,
                                                metadata={"source": dest_path, "type": "text"}
                                            )
                                            st.success(f"Added text file {file_name} to knowledge base")
                                        except Exception as text_error:
                                            st.error(f"Error processing text file {file_name}: {str(text_error)}")
                                    else:
                                        st.warning(f"File type for {file_name} requires specialized processing")
                                except Exception as e:
                                    st.error(f"Error processing file: {str(e)}")
                else:
                                            st.error(f"File not found: {direct_file_path}")
            
            # Standard file uploader with clear warning about limitations
            st.markdown("---")
            st.info("Standard file uploader (limited to 200MB by Streamlit):")
                    # In the file types section
            file_types = ["txt", "pdf", "csv", "json", "docx"] + (["jpg", "jpeg", "png", "gif", "mp4", "avi", "mov", "mkv"] if st.session_state.get("is_vision_model", False) else [])
            
            uploaded_files = st.file_uploader("Upload context documents", accept_multiple_files=True, type=file_types)
            
            # Process Documents button
            if uploaded_files and st.button("Process Documents"):
                with st.spinner("Processing..."):
                    # Your existing document processing code...
                    pass
                # Video Frames Tab - Updated to ensure it works correctly
        with settings_tabs[4]:
            st.header("Video Frame Analysis")
            
            if hasattr(st.session_state, "last_video_data") and st.session_state.last_video_data:
                video_data = st.session_state.last_video_data
                
                # Check if there was an error in video processing
                if isinstance(video_data, dict) and "error" in video_data:
                    st.error(f"Error in video processing: {video_data['error']}")
                else:
                    frames = video_data.get("frames", [])
                    
                    if frames:
                        st.success(f"Processed {len(frames)} frames at 0.5 second intervals")
                        
                        # Display video metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Frames", video_data.get("total_frames", 0))
                            st.metric("Duration", f"{video_data.get('duration', 0):.2f}s")
                        with col2:
                            st.metric("Processed Frames", video_data.get("processed_frames", 0))
                            st.metric("FPS", f"{video_data.get('fps', 0):.2f}")
                        
                        # Create a slider to navigate through frames
                        selected_frame_idx = st.slider("Select Frame", 0, len(frames)-1, 0)
                        
                        # Get the selected frame data
                        frame_data = frames[selected_frame_idx]
                        
                        # Display the frame image
                        if "image_path" in frame_data:
                            try:
                                # Check if file exists
                                if os.path.exists(frame_data["image_path"]):
                                    st.image(frame_data["image_path"], caption=f"Frame {selected_frame_idx+1}/{len(frames)}")
                                else:
                                    # Try to find the file with a different path structure
                                    base_filename = os.path.basename(frame_data["image_path"])
                                    alt_path = os.path.join("data", "temp", base_filename)
                                    if os.path.exists(alt_path):
                                        st.image(alt_path, caption=f"Frame {selected_frame_idx+1}/{len(frames)}")
                                    else:
                                        st.warning(f"Frame image not found. Tried paths: {frame_data['image_path']} and {alt_path}")
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                        
                        # Display the frame description
                        if "description" in frame_data:
                            st.subheader("Frame Description")
                            st.write(frame_data["description"])
                        
                        # Display frame timestamp
                        if "timestamp" in frame_data:
                            st.write(f"Timestamp: {frame_data['timestamp']:.2f} seconds")
                        
                        # Add a button to use this frame in a prompt
                        # For the "Ask about this frame" button (in the frame display section)
                        if st.button(f"Ask about this frame", key=f"ask_frame_{selected_frame_idx}"):
                            # Create the prompt
                            frame_timestamp = frame_data.get("timestamp", 0)
                            frame_prompt = f"Tell me more about what's happening at timestamp {frame_timestamp:.2f} seconds in the video."
                            
                            # Add the message to session state
                            st.session_state.messages.append({"role": "user", "content": frame_prompt})
                            
                            # Process the message and get a response
                            with st.spinner("Generating response..."):
                                response = get_async_response(process_chat(frame_prompt, model_id))
                                
                                # Add the response to session state
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Force a rerun to update the UI
                            st.rerun()
                    else:
                        st.warning("No frames available for this video.")
            else:
                st.info("Upload and process a video to see frame analysis here.")


    task = st.selectbox(
        "Select Task",
        ["Chat", "Summarize", "Sentiment", "Named Entities", "Question Answering", "Code Generation", "Comprehensive Analysis"]
    )

    # In the chat interface section
        # In the chat interface section
    if task == "Chat":
        st.header("Chat Interface")
        doc_count = len(st.session_state.assistant.document_store.get_all_documents()) if hasattr(st.session_state.assistant, "document_store") else 0
        st.info(f"RAG Context: {doc_count} documents indexed")
        
        # Handle document store operations
        if hasattr(st.session_state.assistant, "document_store"):
            try:
                # Create conversation document
                conversation_doc = {
                    "id": f"conversation_{current_session_id}_{int(time.time())}",
                    "content": "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]),
                    "metadata": {
                        "type": "conversation",
                        "session_id": current_session_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model": model_id
                    }
                }
                
                # Try different methods to add document
                try:
                    # Try add_document method first
                    st.session_state.assistant.document_store.add_document(
                        conversation_doc["id"],
                        conversation_doc["content"],
                        conversation_doc["metadata"]
                    )
                except AttributeError:
                    # If add_document doesn't exist, try alternative methods
                    if hasattr(st.session_state.assistant.document_store, 'add'):
                        st.session_state.assistant.document_store.add(
                            conversation_doc["id"],
                            conversation_doc["content"],
                            conversation_doc["metadata"]
                        )
                    elif hasattr(st.session_state.assistant.document_store, 'add_texts'):
                        st.session_state.assistant.document_store.add_texts(
                            [conversation_doc["content"]],
                            metadatas=[conversation_doc["metadata"]],
                            ids=[conversation_doc["id"]]
                        )
                    else:
                        raise AttributeError("Could not find a compatible method to add documents to the document store")
            except Exception as e:
                st.warning(f"Could not save conversation to RAG: {str(e)}")

        if hasattr(st.session_state, "last_video_data") and st.session_state.last_video_data:
            with st.expander("Video Data (Added to Knowledge Base)"):
                st.write(f"Video processed: {len(st.session_state.last_video_data.get('frames', []))} frames")
                
                # Display video summary if available
                if "summary" in st.session_state.last_video_data and st.session_state.last_video_data["summary"]:
                    st.subheader("Video Summary")
                    st.write(st.session_state.last_video_data["summary"])
                
                # Add notification about Video Frames tab
                st.info("👉 View detailed frame analysis in the 'Video Frames' tab in the sidebar")
                
                # Fix the button by using a unique key and st.rerun()
                if st.button("Ask about this video", key=f"ask_video_btn"):
                    # Create the prompt
                    video_prompt = "Tell me about the video I just uploaded. Provide a detailed description of what's happening in the video."
                    
                    # Add the message to session state
                    st.session_state.messages.append({"role": "user", "content": video_prompt})
                    
                    # Process the message and get a response
                    with st.spinner("Generating response..."):
                        response = get_async_response(process_chat(video_prompt, model_id))
                        
                        # Add the response to session state
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Force a rerun to update the UI
                    st.rerun()
        
        # Add image upload functionality for vision models
        if st.session_state.selected_model_id in VISION_MODELS:
            st.markdown("### 📷 Image Analysis")
            
            # Option 1: File uploader with clear size indication
            uploaded_image = st.file_uploader(
                "Upload an image to analyze", 
                type=["jpg", "jpeg", "png", "gif"],
                help="Upload images for vision model analysis"
            )
            
            # Option 2: Alternative upload method for larger files
            st.markdown("#### Or select from file system:")
            if st.button("Browse for image", key="browse_image_btn"):
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                image_path = filedialog.askopenfilename(
                    title="Select Image",
                    filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.gif")]
                )
                root.destroy()
                
                if image_path:
                    st.session_state.selected_image_path = image_path
            
            # Process uploaded image from file uploader
            if uploaded_image is not None:
                # Save the uploaded image
                image_path = os.path.join("data/temp", uploaded_image.name)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                # Display the image
                st.image(image_path, caption="Uploaded Image", width=400)
                st.session_state.current_image = image_path
            
            # Process selected image from file browser
            elif hasattr(st.session_state, "selected_image_path") and st.session_state.selected_image_path:
                image_path = st.session_state.selected_image_path
                
                if os.path.exists(image_path):
                    # Display the image
                    st.image(image_path, caption="Selected Image", width=400)
                    st.session_state.current_image = image_path
                    
                    # Show file info
                    file_name = os.path.basename(image_path)
                    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                    st.info(f"Image: {file_name} ({file_size_mb:.2f} MB)")
                else:
                    st.error(f"Image file not found: {image_path}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Display thinking process if available
                if message["role"] == "assistant" and "thinking" in message:
                    with st.expander("Show thinking process"):
                        st.markdown(message["thinking"])
        
        # Add option to show thinking process for deep thinking models - FIXED INDENTATION
        show_thinking = False
        if model_id in ["deepseek", "llama"]:  # Models that support detailed thinking
            show_thinking = st.checkbox("Show thinking process", value=False)
        
        # Chat input - FIXED INDENTATION
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
             # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                thinking_placeholder = st.empty()
                
                try:
                    # Ensure assistant is initialized
                    if "assistant" not in st.session_state:
                        init_session_state()
                    
                    # Process with or without image
                    if st.session_state.current_image and st.session_state.selected_model_id in VISION_MODELS:
                        response = get_async_response(process_chat(prompt, st.session_state.selected_model_id, st.session_state.current_image))
                    else:
                        response = get_async_response(process_chat(prompt, st.session_state.selected_model_id))
                    
                    # Display response
                    message_placeholder.markdown(response)
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Option to clear chat history and current image
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_image = None
            st.rerun()
    else:
        text_input = st.text_area("Input Text:", height=150)
        if st.button("Process"):
            if not text_input:
                st.warning("Please enter text")
                return
            
            with st.spinner("Analyzing..."):
                result = asyncio.run(process_task(text_input, {
                    "Summarize": "summarize",
                    "Sentiment": "sentiment",
                    "Named Entities": "ner",
                    "Question Answering": "qa",
                    "Code Generation": "code",
                    "Comprehensive Analysis": "analyze"
                }[task], model_id))

                if isinstance(result, Coroutine):
                    async def process_result():
                        return await result
                    
                    result = asyncio.run(process_result())
                    
                    if result is None:
                        result = {}
                    if "error" in result:
                        return
                
                if task == "Summarize":
                    st.subheader("Summary")
                    st.write(result)
                elif task == "Sentiment":
                    st.subheader("Sentiment Analysis")
                    st.json(result)
                elif task == "Named Entities":
                    st.subheader("Named Entities")
                    st.json(result)
                elif task == "Question Answering":
                    st.subheader("Answer")
                    st.write(result)
                elif task == "Code Generation":
                    language = st.selectbox("Language:", ["Python", "JavaScript", "Java", "C++", "Other"])
                    st.subheader("Generated Code")
                    st.code(result["code"], language=language.lower())
                elif task == "Comprehensive Analysis":
                    tabs = st.tabs(["Summary", "Details", "Entities", "Q&A", "Code", "Documents"])
                    with tabs[0]:
                        st.write(result.get("executive_summary", "No summary"))
                        st.write(result.get("summary", {}).get("summary", ""))
                        st.write(result.get("sentiment", {}).get("analysis", ""))
                    with tabs[1]:
                        st.json({
                            "processing_time": result.get("processing_time", 0),
                            "model_used": result.get("model_used"),
                            "text_length": len(text_input)
                        })
                    with tabs[2]:
                        st.write(result.get("entities", {}).get("entities", "None found"))
                    with tabs[3]:
                        for i, qa in enumerate(result.get("questions_and_answers", [])):
                            st.write(f"**Q{i+1}:** {qa.get('question')}")
                            st.write(qa.get('answer', {}).get('answer'))
                    with tabs[4]:
                        for i, code in enumerate(result.get("code_analysis", [])):
                            st.code(code.get("code"), language="python")
                            st.write(code.get("explanation"))
                    with tabs[5]:
                        for i, doc in enumerate(result.get("relevant_documents", [])):
                            st.write(f"**{doc.get('title')}** (Relevance: {doc.get('relevance', 0):.2f})")

    st.markdown("---\n<div style='text-align: center'><p>Built with ❤️ using Streamlit</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()