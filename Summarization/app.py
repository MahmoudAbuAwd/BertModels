import streamlit as st
from transformers import pipeline
import time

# Set page config
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading to improve performance
@st.cache_resource
def load_summarizer():
    """Load the BART summarization model"""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def count_words(text):
    """Count words in text"""
    return len(text.split())

def estimate_reading_time(text):
    """Estimate reading time (average 200 words per minute)"""
    word_count = count_words(text)
    reading_time = word_count / 200
    return max(1, round(reading_time))

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform long texts into concise, meaningful summaries using advanced AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model... This may take a moment on first run."):
        summarizer = load_summarizer()
    
    if summarizer is None:
        st.error("Failed to load the summarization model. Please try again later.")
        return
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Summarization Settings")
    
    # Summary length settings
    st.sidebar.subheader("Summary Length")
    max_length = st.sidebar.slider(
        "Maximum length (words)", 
        min_value=20, 
        max_value=200, 
        value=60, 
        step=10,
        help="Maximum number of words in the summary"
    )
    
    min_length = st.sidebar.slider(
        "Minimum length (words)", 
        min_value=10, 
        max_value=min(max_length-5, 100), 
        value=25, 
        step=5,
        help="Minimum number of words in the summary"
    )
    
    # Advanced settings
    st.sidebar.subheader("Advanced Options")
    do_sample = st.sidebar.checkbox(
        "Enable sampling", 
        value=False,
        help="Enable sampling for more creative summaries (may be less consistent)"
    )
    
    # Sample texts
    st.sidebar.subheader("📋 Sample Texts")
    sample_texts = {
        "Technology": """
        Artificial intelligence is transforming industries by automating complex tasks, improving decision-making through data analysis, and enabling the development of innovative products and services across healthcare, finance, transportation, and more. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions that were previously impossible for humans to achieve. Deep learning, a subset of machine learning, uses neural networks with multiple layers to solve complex problems such as image recognition, natural language processing, and autonomous driving. Companies are investing billions of dollars in AI research and development to stay competitive in the rapidly evolving technological landscape.
        """,
        "Climate Change": """
        Climate change represents one of the most pressing challenges of our time, with global temperatures rising due to increased greenhouse gas emissions from human activities. The burning of fossil fuels, deforestation, and industrial processes have led to unprecedented levels of carbon dioxide in the atmosphere. Scientists warn that without immediate action to reduce emissions, we face severe consequences including rising sea levels, extreme weather events, biodiversity loss, and threats to food security. Governments, businesses, and individuals must work together to implement sustainable practices, invest in renewable energy, and develop innovative solutions to mitigate the effects of climate change and transition to a low-carbon economy.
        """,
        "Healthcare": """
        Modern healthcare is experiencing a revolutionary transformation through the integration of digital technologies, personalized medicine, and advanced diagnostic tools. Telemedicine has made healthcare more accessible, allowing patients to consult with doctors remotely and receive medical advice from the comfort of their homes. Artificial intelligence is being used to analyze medical images, predict disease outcomes, and assist in drug discovery processes. Genomic medicine enables doctors to tailor treatments based on individual genetic profiles, leading to more effective and targeted therapies. Electronic health records have improved patient care coordination and reduced medical errors, while wearable devices help monitor vital signs and health metrics in real-time.
        """
    }
    
    for topic, text in sample_texts.items():
        if st.sidebar.button(f"Load {topic} Sample"):
            st.session_state.sample_text = text.strip()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📄 Input Text")
        
        # Text input
        default_text = st.session_state.get('sample_text', '')
        input_text = st.text_area(
            "Enter or paste your text here:",
            value=default_text,
            height=300,
            placeholder="Paste your article, document, or any long text that you want to summarize...",
            help="The text should be at least 50 words for best results"
        )
        
        # Input text statistics
        if input_text:
            word_count = count_words(input_text)
            char_count = len(input_text)
            reading_time = estimate_reading_time(input_text)
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.markdown(
                    f'<div class="metric-card"><strong>{word_count}</strong><br>Words</div>',
                    unsafe_allow_html=True
                )
            with col_stats2:
                st.markdown(
                    f'<div class="metric-card"><strong>{char_count}</strong><br>Characters</div>',
                    unsafe_allow_html=True
                )
            with col_stats3:
                st.markdown(
                    f'<div class="metric-card"><strong>{reading_time} min</strong><br>Reading Time</div>',
                    unsafe_allow_html=True
                )
    
    with col2:
        st.subheader("✨ Summary")
        
        # Summarize button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text to summarize.")
            elif count_words(input_text) < 20:
                st.warning("Please enter at least 20 words for better summarization results.")
            else:
                try:
                    with st.spinner("Generating summary... Please wait."):
                        start_time = time.time()
                        
                        # Generate summary
                        summary_result = summarizer(
                            input_text,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=do_sample
                        )
                        
                        end_time = time.time()
                        processing_time = round(end_time - start_time, 2)
                        
                        # Display summary
                        summary_text = summary_result[0]['summary_text']
                        
                        st.markdown(
                            f'<div class="summary-box">{summary_text}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Summary statistics
                        summary_word_count = count_words(summary_text)
                        compression_ratio = round((1 - summary_word_count / count_words(input_text)) * 100, 1)
                        
                        st.success(f"✅ Summary generated in {processing_time}s")
                        
                        col_sum1, col_sum2 = st.columns(2)
                        with col_sum1:
                            st.metric("Summary Length", f"{summary_word_count} words")
                        with col_sum2:
                            st.metric("Compression", f"{compression_ratio}%")
                        
                        # Copy button (using st.code for easy copying)
                        st.subheader("📋 Copy Summary")
                        st.code(summary_text, language=None)
                        
                except Exception as e:
                    st.error(f"An error occurred while generating the summary: {str(e)}")
        
        # Instructions
        st.markdown("""
        ### 💡 Tips for Better Summaries:
        - **Text Length**: Use texts with at least 50+ words
        - **Quality**: Well-structured texts produce better summaries
        - **Length Settings**: Adjust min/max length based on your needs
        - **Sampling**: Enable for more creative but less consistent results
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Powered by <strong>BART-Large-CNN</strong> from Hugging Face Transformers</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# This code is part of a Streamlit app for text summarization using a pre-trained BART model.

