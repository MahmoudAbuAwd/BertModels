import streamlit as st
import torch
import time
import json
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from functools import lru_cache
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import os

# Page configuration
st.set_page_config(
    page_title="🤖 AI Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-label {
        color: #495057 !important;
        font-weight: 600 !important;
        font-size: 0.9em !important;
        margin-bottom: 0.5rem !important;
    }
    
    .metric-value {
        color: #212529 !important;
        font-size: 1.2em !important;
        font-weight: bold !important;
    }
    
    .answer-box {
        background: #f8fff8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .answer-text {
        color: #1e4d1e !important;
        font-size: 1.1em !important;
        line-height: 1.6 !important;
        margin: 0.5rem 0 !important;
        font-weight: 500 !important;
    }
    
    .answer-header {
        color: #155724 !important;
        font-weight: bold !important;
        margin-bottom: 1rem !important;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitQAPipeline:
    def __init__(self, model_name: str = "deepset/deberta-v3-base-squad2"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = None
        self.is_loaded = False
        
        # Initialize session state for history
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        if 'stats' not in st.session_state:
            st.session_state.stats = {
                'total_questions': 0,
                'avg_inference_time': 0,
                'total_inference_time': 0,
                'response_times': [],
                'confidence_scores': []
            }
    
    @st.cache_resource
    def load_model(_self, model_name: str):
        """Load and cache the QA model"""
        try:
            with st.spinner("🔄 Loading AI model..."):
                # Load model components
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                
                # GPU optimization
                device = 0 if torch.cuda.is_available() else -1
                if device == 0:
                    model = model.cuda()
                    if hasattr(model, 'half'):
                        model = model.half()
                
                # Create optimized pipeline
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_seq_len=384,
                    stride=128,
                    max_question_len=64,
                    max_answer_len=100,
                    handle_impossible_answer=True,
                    top_k=1
                )
                
                # Warm up model
                qa_pipeline(
                    question="What is this?",
                    context="This is a warmup context for the model."
                )
                
                return qa_pipeline, device
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, -1
    
    def initialize(self):
        """Initialize the QA pipeline"""
        if not self.is_loaded:
            self.qa_pipeline, self.device = self.load_model(self.model_name)
            self.is_loaded = self.qa_pipeline is not None
        return self.is_loaded
    
    @lru_cache(maxsize=100)
    def get_cached_answer(self, question: str, context_hash: int) -> Tuple[str, float, float]:
        """Get cached answer for repeated questions"""
        return self._process_question(question, context_hash)
    
    def _process_question(self, question: str, context_hash: int) -> Tuple[str, float, float]:
        """Internal method to process questions"""
        context = st.session_state.get('current_context', '')
        
        if not self.is_loaded or not self.qa_pipeline:
            return "❌ Model not loaded properly. Please refresh the page.", 0.0, 0.0
        
        start_time = time.time()
        
        try:
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=150
            )
            
            inference_time = time.time() - start_time
            return result['answer'], result['score'], inference_time
            
        except Exception as e:
            return f"❌ Error processing question: {str(e)}", 0.0, 0.0
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Main method to answer questions"""
        if not question.strip():
            return {"error": "Please enter a question."}
        
        if not context.strip():
            return {"error": "Please provide context for the question."}
        
        # Store current context
        st.session_state.current_context = context.strip()
        context_hash = hash(context.strip())
        
        # Get answer
        answer, confidence, inference_time = self.get_cached_answer(
            question.strip(), context_hash
        )
        
        # Update statistics
        self._update_stats(inference_time, confidence)
        
        # Add to history
        self._add_to_history(question, answer, confidence, inference_time)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'inference_time': inference_time,
            'device': 'GPU' if self.device == 0 else 'CPU'
        }
    
    def _update_stats(self, inference_time: float, confidence: float):
        """Update performance statistics"""
        st.session_state.stats['total_questions'] += 1
        st.session_state.stats['total_inference_time'] += inference_time
        st.session_state.stats['avg_inference_time'] = (
            st.session_state.stats['total_inference_time'] / 
            st.session_state.stats['total_questions']
        )
        st.session_state.stats['response_times'].append(inference_time)
        st.session_state.stats['confidence_scores'].append(confidence)
        
        # Keep only last 100 records for charts
        if len(st.session_state.stats['response_times']) > 100:
            st.session_state.stats['response_times'] = st.session_state.stats['response_times'][-100:]
            st.session_state.stats['confidence_scores'] = st.session_state.stats['confidence_scores'][-100:]
    
    def _add_to_history(self, question: str, answer: str, confidence: float, inference_time: float):
        """Add interaction to history"""
        st.session_state.history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'inference_time': inference_time
        })
        
        # Keep only last 50 interactions
        if len(st.session_state.history) > 50:
            st.session_state.history = st.session_state.history[-50:]

# Initialize QA system
@st.cache_resource
def get_qa_system():
    return StreamlitQAPipeline()

# Sample contexts
SAMPLE_CONTEXTS = {
    "🤖 AI Engineer Profile": """
    Mahmoud Abuawd is an AI Engineer who builds systems for computer vision and NLP.
    He has experience with TensorFlow, PyTorch, and Hugging Face models.
    He specializes in transformer architectures and has worked on projects involving
    image classification, object detection, sentiment analysis, and question answering systems.
    He holds a degree in Computer Science and has published research on neural networks.
    His recent work focuses on deploying large language models in production environments.
    """,
    
    "🏢 Company Information": """
    TechCorp is a leading AI company founded in 2020. The company specializes in 
    developing machine learning solutions for healthcare, finance, and retail industries.
    They have offices in New York, London, and Tokyo, with over 500 employees worldwide.
    The CEO is Sarah Johnson, and the company's revenue reached $50 million in 2023.
    Their flagship product is an AI-powered diagnostic tool used in over 200 hospitals.
    """,
    
    "⌚ Product Description": """
    The SmartWatch Pro is a cutting-edge wearable device featuring a 1.4-inch OLED display,
    GPS tracking, heart rate monitoring, and 7-day battery life. It's water-resistant up to 
    50 meters and supports over 100 workout modes. The device runs on WearOS and is compatible 
    with both Android and iOS smartphones. It retails for $299 and comes in three colors:
    midnight black, silver, and rose gold. The watch includes sleep tracking, stress monitoring,
    and can make contactless payments.
    """,
    
    "🧬 Scientific Research": """
    Recent studies in machine learning have shown that transformer architectures can achieve
    state-of-the-art performance on various natural language processing tasks. The attention
    mechanism allows models to focus on relevant parts of the input sequence, leading to
    better understanding of context and relationships. BERT, GPT, and T5 are examples of
    successful transformer-based models that have revolutionized the field of AI.
    """
}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Question Answering System</h1>
        <p>Ask questions about any context using advanced AI models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize QA system
    qa_system = get_qa_system()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Model status
        if qa_system.initialize():
            st.success("✅ Model loaded successfully")
            device_info = "🚀 GPU" if qa_system.device == 0 else "💻 CPU"
            st.info(f"Running on: {device_info}")
        else:
            st.error("❌ Model failed to load")
            return
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        stats = st.session_state.stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Questions", stats['total_questions'])
        with col2:
            avg_time = stats['avg_inference_time']
            st.metric("Avg Response", f"{avg_time:.3f}s")
        
        if stats['confidence_scores']:
            avg_conf = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.stats = {
                'total_questions': 0,
                'avg_inference_time': 0,
                'total_inference_time': 0,
                'response_times': [],
                'confidence_scores': []
            }
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        # Context input section
        st.subheader("📄 Context")
        
        # Sample context selector
        selected_sample = st.selectbox(
            "Choose a sample context or paste your own:",
            [""] + list(SAMPLE_CONTEXTS.keys()),
            key="sample_selector"
        )
        
        # Context text area
        if selected_sample:
            default_context = SAMPLE_CONTEXTS[selected_sample]
        else:
            default_context = ""
        
        context = st.text_area(
            "Context",
            value=default_context,
            height=200,
            placeholder="Paste your context here...",
            label_visibility="collapsed"
        )
        
        # Question input
        st.subheader("❓ Your Question")
        question = st.text_input(
            "Question",
            placeholder="What would you like to know about the context?",
            label_visibility="collapsed"
        )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_clicked = st.button("🚀 Get Answer", use_container_width=True)
        
        # Process question
        if submit_clicked or (question and context):
            if question and context:
                with st.spinner("🤔 Thinking..."):
                    result = qa_system.answer_question(question, context)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    # Display answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <h4 class="answer-header">✅ Answer:</h4>
                        <p class="answer-text">{result['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        confidence = result['confidence'] * 100
                        if confidence >= 80:
                            conf_class = "confidence-high"
                        elif confidence >= 50:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">🎯 Confidence</div>
                            <div class="metric-value {conf_class}">{confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">⏱️ Response Time</div>
                            <div class="metric-value">{result['inference_time']:.3f}s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">🖥️ Device</div>
                            <div class="metric-value">{result['device']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">🤖 Model</div>
                            <div class="metric-value">DeBERTa-v3</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please provide both context and question.")
    
    with tab2:
        st.subheader("📊 Performance Analytics")
        
        if st.session_state.history:
            # Recent history table
            st.subheader("📝 Recent Questions")
            history_df = pd.DataFrame(st.session_state.history[-10:])
            history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            history_df['inference_time'] = history_df['inference_time'].apply(lambda x: f"{x:.3f}s")
            
            st.dataframe(
                history_df[['timestamp', 'question', 'answer', 'confidence', 'inference_time']],
                column_config={
                    'timestamp': 'Time',
                    'question': 'Question',
                    'answer': 'Answer',
                    'confidence': 'Confidence',
                    'inference_time': 'Response Time'
                },
                use_container_width=True
            )
            
            # Charts
            if len(st.session_state.stats['response_times']) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⏱️ Response Time Trend")
                    times_df = pd.DataFrame({
                        'Question': range(1, len(st.session_state.stats['response_times']) + 1),
                        'Response Time (s)': st.session_state.stats['response_times']
                    })
                    fig_time = px.line(times_df, x='Question', y='Response Time (s)', 
                                     title="Response Time Over Time")
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Confidence Distribution")
                    conf_df = pd.DataFrame({
                        'Confidence': [score * 100 for score in st.session_state.stats['confidence_scores']]
                    })
                    fig_conf = px.histogram(conf_df, x='Confidence', nbins=20,
                                          title="Confidence Score Distribution")
                    st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("📈 No data yet. Ask some questions to see analytics!")
    
    with tab3:
        st.subheader("ℹ️ About This System")
        
        st.markdown("""
        ## 🎯 Overview
        This Question Answering system uses a fine-tuned **DeBERTa-v3-base** model trained on SQuAD 2.0 dataset
        to provide accurate answers to questions based on provided context.
        
        ## ✨ Features
        - **🎯 High Accuracy**: Fine-tuned on SQuAD 2.0 for reliable answers
        - **⚡ Fast Inference**: Optimized for quick response times
        - **📊 Confidence Scores**: Know how confident the model is
        - **💾 Smart Caching**: Repeated questions are answered instantly
        - **📈 Real-time Analytics**: Track performance and usage
        - **🎨 Modern Interface**: Clean, responsive Streamlit design
        
        ## 🚀 How to Use
        1. **📄 Provide Context**: Paste relevant text or choose a sample
        2. **❓ Ask Questions**: Type your question about the context
        3. **✅ Get Answers**: Receive answers with confidence scores
        4. **📊 Review Analytics**: Check performance metrics and history
        
        ## 🔧 Technical Details
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Model Information:**
            - Model: deepset/deberta-v3-base-squad2
            - Architecture: DeBERTa v3
            - Training Data: SQuAD 2.0
            - Parameters: ~184M
            """)
        
        with col2:
            st.markdown("""
            **Performance Optimizations:**
            - GPU acceleration when available
            - FP16 precision for faster inference
            - Smart caching for repeated queries
            - Optimized tokenization parameters
            """)
        
        st.markdown("""
        ## 🛠️ Deployment
        
        **Local Development:**
        ```bash
        pip install streamlit transformers torch pandas plotly
        streamlit run app.py
        ```
        
        **Production Deployment:**
        - Streamlit Cloud (Free)
        - Docker containers
        - AWS/GCP/Azure cloud platforms
        
        ## 📞 Support
        For issues or questions, please check the model documentation on Hugging Face.
        """)

if __name__ == "__main__":
    main()