from transformers import pipeline

# Load summarization pipeline with a pre-trained BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text (can be long)
text = """
Artificial intelligence is transforming industries by automating complex tasks, improving decision-making through data analysis, and enabling the development of innovative products and services across healthcare, finance, transportation, and more"""

# Generate summary
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)

# Print summary
print("Summary:", summary[0]['summary_text'])
