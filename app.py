import io
import json
import streamlit as st
import pandas as pd

from PIL import Image
from genai_kit.aws.claude import BedrockClaude
from genai_kit.aws.bedrock import BedrockModel


BASE_PROMPT = f"""이미지에 대해 설명하세요"""

def analyze_image(images, issue_list, model_id, custom_prompt=""):
    # Initialize Claude client for image analysis
    claude_client = BedrockClaude(
        region='us-west-2',
        modelId=model_id
    )
    
    # Convert images to bytes
    img_bytes_list = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes_list.append(img_byte_arr.getvalue())
    
    # Combine base prompt with custom prompt if provided
    prompt = custom_prompt if custom_prompt else BASE_PROMPT + f"""
    아래는 발생할 수 있는 이슈의 종류입니다:
    <issues>
    {issue_list.to_csv()}
    </issues>
    """

    # Get analysis from Claude using streaming
    try:
        for chunk in claude_client.converse_stream(
            text=prompt,
            images=img_bytes_list
        ):
            yield chunk
    except Exception as e:
        yield str(e)

def main():
    st.set_page_config(layout="wide")
    st.title("🧐 Android UI Validation with GenAI")

    issue_list = pd.read_csv('issues.csv')
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Prompt Input
    with col1:
        st.subheader("Custom Prompt")
        
        # Initialize session state if not exists
        if "custom_prompt" not in st.session_state:
            st.session_state.custom_prompt = BASE_PROMPT
       
        # Text area for custom prompt
        custom_prompt = st.text_area(
            "프롬프트를 입력하세요 (선택사항)",
            value=st.session_state.custom_prompt,
            height=500,
            key="custom_prompt",
        )

        # Handle clear button click
        if st.button("Clear", use_container_width=True, key="clear_button"):
            if "custom_prompt" in st.session_state:
                del st.session_state.custom_prompt
                st.session_state.custom_prompt = ""
            st.rerun()
        
    # Column 2: Image Upload and Preview
    with col2:
        st.subheader("Image Upload")
        uploaded_files = st.file_uploader(
            "이미지를 선택하세요 (복수 가능, 최대 20개)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        # Create a scrollable container for images
        image_container = st.container()
        with image_container:
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f'Image: {uploaded_file.name}', use_container_width=True)
    
    # Column 3: Analysis Results
    with col3:
        st.subheader("Analysis Results")
        
        # Add model selection dropdown
        claude_models = [
            (BedrockModel.SONNET_3_0_CR, "Claude 3.0 Sonnet"),
            # (BedrockModel.SONNET_3_5_CR, "Claude 3.5 Sonnet"),
            # (BedrockModel.SONNET_3_7_CR, "Claude 3.7 Sonnet"),
            (BedrockModel.NOVA_LITE_CR, "Nova Lite"),
            (BedrockModel.NOVA_PRO_CR, "Nova Pro"),                 
        ]

        selected_model = st.selectbox(
            "Select Bedrock Model",
            options=[model[0] for model in claude_models],
            format_func=lambda x: dict(claude_models)[x],
            index=0
        )
        
        if uploaded_files and st.button('Analyze Images', type="primary", use_container_width=True):
            # Create a placeholder for streaming output
            result_placeholder = st.empty()
            
            # Convert uploaded files to PIL Images
            images = [Image.open(file) for file in uploaded_files]
            
            # Stream the analysis results
            accumulated_text = ""
            for chunk in analyze_image(images, issue_list, selected_model, custom_prompt):
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('output', {}).get('message', {}).get('content')[0].get('text', '')
                else:
                    chunk_text = str(chunk)
                    
                accumulated_text += chunk_text
                try:
                    # Try to parse as JSON and display
                    json_data = json.loads(accumulated_text)
                    result_placeholder.json(json_data)
                except json.JSONDecodeError:
                    # If not valid JSON yet, show raw text
                    result_placeholder.text(accumulated_text)

if __name__ == "__main__":
    main()
