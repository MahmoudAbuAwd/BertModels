import streamlit as st
import torch
from transformers import pipeline
import time

# Set page config
st.set_page_config(
    page_title="BERT Fill-Mask Predictor",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for the model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.fill_mask = None

@st.cache_resource
def load_model():
    """Load the BERT fill-mask model with caching"""
    try:
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        fill_mask = pipeline(
            task="fill-mask",
            model="google-bert/bert-base-uncased",
            torch_dtype=torch_dtype,
            device=device
        )
        return fill_mask, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def main():
    st.title("🤖 BERT Fill-Mask Predictor")
    st.markdown("Enter a sentence with `[MASK]` to see what BERT thinks should fill in the blank!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses Google's BERT (base-uncased) model to predict masked words in sentences.
        
        **How to use:**
        1. Enter a sentence with `[MASK]` where you want predictions
        2. Choose how many predictions to show
        3. Click "Predict" to see results
        
        **Example sentences:**
        - "I love [MASK]."
        - "The weather is [MASK] today."
        - "Python is a [MASK] programming language."
        """)
        
        # Model info
        st.header("🔧 Model Info")
        if torch.cuda.is_available():
            st.success("✅ CUDA Available")
            st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ Using CPU")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading BERT model... This may take a moment."):
            fill_mask, device = load_model()
            if fill_mask:
                st.session_state.fill_mask = fill_mask
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check your setup.")
                return
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input text
        user_input = st.text_input(
            "Enter your sentence with [MASK]:",
            value="I love [MASK].",
            placeholder="Type your sentence here with [MASK]..."
        )
    
    with col2:
        # Number of predictions
        num_predictions = st.slider(
            "Number of predictions:",
            min_value=1,
            max_value=10,
            value=5
        )
    
    # Predict button
    if st.button("🔍 Predict", type="primary"):
        if not user_input.strip():
            st.error("Please enter a sentence.")
            return
        
        if "[MASK]" not in user_input:
            st.error("Please include [MASK] in your sentence.")
            return
        
        # Count number of masks
        mask_count = user_input.count("[MASK]")
        if mask_count > 1:
            st.warning("Multiple [MASK] tokens detected. BERT works best with single masks.")
        
        # Make prediction
        try:
            with st.spinner("Generating predictions..."):
                start_time = time.time()
                result = st.session_state.fill_mask(user_input)
                end_time = time.time()
            
            # Display results
            st.success(f"✅ Predictions generated in {end_time - start_time:.2f} seconds")
            
            # Show top predictions
            st.subheader("🎯 Top Predictions")
            
            # Create columns for better layout
            for i, prediction in enumerate(result[:num_predictions]):
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 1])
                    
                    with col1:
                        st.markdown(f"**#{i+1}**")
                    
                    with col2:
                        # Highlight the predicted word
                        sequence = prediction['sequence']
                        predicted_word = prediction['token_str']
                        highlighted = sequence.replace(predicted_word, f"**{predicted_word}**")
                        st.markdown(f"*{highlighted}*")
                    
                    with col3:
                        score = prediction['score']
                        st.markdown(f"`{score:.1%}`")
                    
                    # Add a subtle separator
                    if i < len(result[:num_predictions]) - 1:
                        st.markdown("---")
            
            # Show additional details in expander
            with st.expander("📊 Detailed Results"):
                st.json(result[:num_predictions])
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please try a different sentence or check your input format.")
    
    # Example sentences
    st.subheader("💡 Try These Examples")
    example_sentences = [
        "The cat is [MASK].",
        "I went to the [MASK] yesterday.",
        "Python is a [MASK] programming language.",
        "The movie was [MASK] and entertaining.",
        "She works as a [MASK] at the hospital."
    ]
    
    cols = st.columns(len(example_sentences))
    for i, (col, sentence) in enumerate(zip(cols, example_sentences)):
        with col:
            if st.button(f"Try: {sentence[:20]}...", key=f"example_{i}"):
                st.rerun()

if __name__ == "__main__":
    main()