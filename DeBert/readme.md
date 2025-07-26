# 🤖 AI Question Answering System

An optimized Question Answering system built with Transformers and Streamlit, featuring high-performance inference, smart caching, and a modern web interface.

## ✨ Features

- **🎯 High Accuracy**: Fine-tuned DeBERTa-v3-base model trained on SQuAD 2.0
- **⚡ Fast Inference**: GPU acceleration with FP16 precision and smart optimizations
- **💾 Intelligent Caching**: LRU cache for repeated queries
- **📊 Real-time Analytics**: Performance metrics and confidence tracking
- **🎨 Modern Web UI**: Interactive Streamlit interface with responsive design
- **📈 Batch Processing**: Efficient handling of multiple questions
- **🔄 Auto-warmup**: Model preloading for consistent performance

## 🖥️ Usage

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run DeBert/app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Interactive Q&A interface
- Sample contexts to get started
- Real-time performance analytics
- Question history tracking
- Confidence visualization

### Option 2: Command Line Interface

```bash
python DeBert/main.py
```

**Commands:**
- Type your question and press Enter
- Type `batch` to test batch processing
- Type `quit` to exit


## 🔧 Configuration

### Model Selection

Change the model in both files:

```python
# Default: deepset/deberta-v3-base-squad2
qa_system = OptimizedQAPipeline(model_name="your-model-name")
```

**Alternative models:**
- `deepset/roberta-base-squad2`
- `distilbert-base-cased-distilled-squad`
- `bert-large-uncased-whole-word-masking-finetuned-squad`

### Performance Tuning

```python
# In OptimizedQAPipeline.__init__()
max_seq_len=384,        # Reduce for faster processing
max_answer_len=30,      # Limit answer length
stride=128,             # Overlap for long contexts
```

## 📊 Performance Metrics

The system tracks several performance indicators:

- **Response Time**: Inference time per question
- **Confidence Score**: Model's certainty in answers (0-100%)
- **Throughput**: Questions processed per second
- **Cache Hit Rate**: Percentage of cached responses

## 🎯 Sample Usage

### Web Interface

1. Choose a sample context or paste your own
2. Ask questions about the context
3. View answers with confidence scores
4. Check analytics for performance insights

### Command Line

```python
# Example interaction
Ask a question:
> What is Mahmoud's profession?

Answer: AI Engineer
Confidence: 0.956
Inference time: 0.234 seconds
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

