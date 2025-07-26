# BERT Fill-Mask Predictor

A Streamlit web application that uses Google's BERT model to predict masked words in sentences.

## Features

- Interactive interface for entering sentences with `[MASK]` tokens
- Displays top predictions with confidence scores
- Supports multiple predictions (1-10 results)
- Shows processing time for each prediction
- Example sentences for quick testing
- Works on both CPU and GPU (automatically detects CUDA availability)

## Requirements

- Python 3.7+
- Streamlit
- PyTorch
- Transformers library
- CUDA toolkit (optional, for GPU acceleration)


## Usage

1. Run the application:
   ```bash
   streamlit run Prediction_Bert/app.py
   ```

2. Enter a sentence containing `[MASK]` in the input box
3. Select the number of predictions to show (1-10)
4. Click "Predict" to see the results

## Examples

Try these sample inputs:
- "I love [MASK]."
- "The weather is [MASK] today."
- "Python is a [MASK] programming language."

## Model Information

- Uses `google-bert/bert-base-uncased` from Hugging Face
- Automatically uses GPU if available (CUDA)
- First run will download the model (~440MB)

## License

This project is open source and available under the MIT License.
```
