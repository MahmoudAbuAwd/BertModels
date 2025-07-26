# AI Text Summarizer

A user-friendly web application for text summarization using Facebook's BART-Large-CNN model, built with Streamlit.

## Overview

This project provides an intuitive interface for generating high-quality text summaries using state-of-the-art natural language processing. The application leverages the pre-trained BART (Bidirectional and Auto-Regressive Transformers) model to transform long articles, documents, and texts into concise, meaningful summaries.

## Features

### Core Functionality
- **AI-Powered Summarization**: Uses Facebook's BART-Large-CNN model for high-quality text summarization
- **Customizable Summary Length**: Adjustable minimum and maximum word count settings
- **Real-Time Text Analysis**: Live word count, character count, and reading time estimation
- **Sample Text Library**: Pre-loaded examples across different domains (Technology, Climate, Healthcare)

### User Interface
- **Clean, Modern Design**: Professional styling with responsive layout
- **Interactive Controls**: Intuitive sliders and buttons for parameter adjustment
- **Progress Indicators**: Loading states and processing time display
- **Copy-Friendly Output**: Easy-to-copy summary results

### Advanced Options
- **Sampling Mode**: Toggle between deterministic and creative summary generation
- **Input Validation**: Minimum text length requirements for optimal results
- **Performance Metrics**: Compression ratio and processing time analytics
- **Error Handling**: Robust error management and user feedback

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Facebook BART-Large-CNN (Hugging Face Transformers)
- **Backend**: Python
- **Model Loading**: Cached pipeline for improved performance


## Usage


1. Run the application:
   ```bash
   streamlit run Summarization/app.py
   ```

1. **Launch the application** using the command above
2. **Enter your text** in the input area (minimum 20 words recommended)
3. **Adjust settings** in the sidebar:
   - Set maximum summary length (20-200 words)
   - Set minimum summary length (10-100 words)
   - Enable/disable sampling for creative variation
4. **Click "Generate Summary"** to process your text
5. **View results** including summary, statistics, and performance metrics
6. **Copy the summary** using the provided code block

## Model Information

- **Model**: `facebook/bart-large-cnn`
- **Architecture**: BART (Bidirectional and Auto-Regressive Transformers)
- **Training**: Fine-tuned on CNN/DailyMail dataset
- **Specialization**: Abstractive text summarization
- **Performance**: Optimized for news articles and general text

## Sample Use Cases

- **Academic Research**: Summarize lengthy research papers and articles
- **News Analysis**: Condense news articles for quick consumption
- **Business Documents**: Create executive summaries of reports
- **Content Creation**: Generate brief descriptions for long-form content
- **Study Materials**: Summarize textbooks and educational content

## Performance Considerations

- **First Load**: Model download may take 1-2 minutes initially
- **Subsequent Uses**: Cached model loads instantly
- **Processing Time**: Typically 2-5 seconds per summary
- **Text Length**: Optimal performance with 50+ word inputs
- **Memory Usage**: Requires ~2GB RAM for model loading

## Technical Details

### Model Pipeline
- Utilizes Hugging Face Transformers pipeline
- Implements caching with `@st.cache_resource`
- Supports both deterministic and sampling-based generation

### Input Processing
- Text validation and word count analysis
- Reading time estimation (200 words/minute average)
- Character count and compression ratio calculation

### Output Features
- Formatted summary display with custom styling
- Performance metrics and processing statistics
- Copy-friendly text output for easy sharing

## License

Open source project using pre-trained models from Hugging Face Transformers library.