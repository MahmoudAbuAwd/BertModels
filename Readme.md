# NLP Project Suite with BERT

This project provides a suite of NLP tools built around BERT architecture, including DeBERT implementations, prediction services, and text summarization capabilities.

## Project Structure

```
.
├── DeBert/
│   ├── app.py                # Flask application for BERT services
│   ├── main.py               # Main DEBert implementation script
│   ├── readme.md             # BERT-specific documentation
│   └── sample_data.txt       # Sample data for testing
│
├── Prediction_Bert/
│   ├── app.py                # Prediction service application
│   ├── main.py               # Prediction implementation
│   └── readme.md             # Prediction module docs
│
├── Summarization/
│   ├── app.py                # Summarization web service
│   ├── main.py               # Summarization implementation
│   └── readme.md             # Summarization docs
│
├── python-version            # Specifies Python version compatibility
└── requirements.txt          # Project dependencies
└── Licence                   # Project Licence


```

## Getting Started

### Prerequisites
- Python 3.7+
- pip package manager
- Git (for cloning)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MahmoudAbuAwd/BertModels.git
cd BertModels
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Applications

### 1. BERT Module
```bash
cd DeBert
python app.py  # Runs the web service (default port 5000)
# or
python main.py  # Runs the standalone implementation
```

### 2. Prediction Module
```bash
cd Prediction_Bert
python app.py  # Starts prediction API service
# or
python main.py --input "your text here"  # Command-line prediction
```

### 3. Summarization Module
```bash
cd Summarization
python app.py  # Launches summarization web interface
# or
python main.py --text "your long text to summarize"  # CLI summarization
```

## Configuration

Each module has its own configuration options. Refer to the individual `readme.md` files in each directory for module-specific settings.

## API Endpoints

When running the `app.py` files, the following endpoints are typically available:

- **BERT Service**: `http://localhost:5000/Debert`
- **Prediction Service**: `http://localhost:5001/predict`
- **Summarization Service**: `http://localhost:5002/summarize`

Note: Ports may vary depending on configuration.

## License

This project is licensed under the MIT License - see the LICENSE file (to be added) for details.
