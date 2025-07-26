from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import time
from functools import lru_cache

class OptimizedQAPipeline:
    def __init__(self, model_name="deepset/deberta-v3-base-squad2", use_gpu=True):
        """
        Initialize optimized QA pipeline with various performance improvements
        """
        self.device = 0 if torch.cuda.is_available() and use_gpu else -1
        
        # Load tokenizer and model separately for more control
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move model to GPU if available
        if self.device == 0:
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")
        
        # Create pipeline with optimizations
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            # Performance optimizations
            return_overflowing_tokens=False,
            max_seq_len=384,  # Reduce from default 512 for faster processing
            stride=128,
            max_question_len=64,
            max_answer_len=30,
            handle_impossible_answer=True,
            top_k=1  # Only return best answer
        )
        
        # Enable model optimizations
        if hasattr(self.model, 'half') and self.device == 0:
            self.model.half()  # Use FP16 for faster inference on GPU
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with a dummy question to optimize first inference"""
        dummy_context = "This is a warmup context."
        dummy_question = "What is this?"
        self.qa_pipeline(question=dummy_question, context=dummy_context)
        print("Model warmed up successfully")
    
    @lru_cache(maxsize=128)
    def cached_answer(self, question, context_hash):
        """Cache answers for repeated questions with same context"""
        return self._get_answer_internal(question, context_hash)
    
    def _get_answer_internal(self, question, context_hash):
        """Internal method for getting answers"""
        # Retrieve original context from hash (simplified - in practice you'd store this properly)
        context = self._get_context_from_hash(context_hash)
        
        start_time = time.time()
        result = self.qa_pipeline(
            question=question, 
            context=context,
            max_answer_len=50,  # Limit answer length for faster processing
            top_k=1
        )
        inference_time = time.time() - start_time
        
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'inference_time': inference_time
        }
    
    def _get_context_from_hash(self, context_hash):
        """Simple context retrieval - in practice, implement proper hash-to-context mapping"""
        # This is a simplified version - you'd want a proper mapping system
        return self.current_context
    
    def answer_question(self, question, context):
        """Main method to get answers with optimizations"""
        self.current_context = context
        
        # Preprocess question and context
        question = question.strip()
        context = self._preprocess_context(context)
        
        # Create a simple hash for caching
        context_hash = hash(context)
        
        # Get cached or new answer
        result = self.cached_answer(question, context_hash)
        
        return result
    
    def _preprocess_context(self, context):
        """Preprocess context for better performance"""
        # Remove extra whitespace and normalize
        context = ' '.join(context.split())
        
        # Truncate if too long (keeping most relevant parts)
        max_context_length = 300  # Adjust based on your needs
        if len(context) > max_context_length:
            # Simple truncation - could be improved with sliding window
            context = context[:max_context_length] + "..."
        
        return context
    
    def batch_answer(self, questions, context):
        """Process multiple questions efficiently"""
        results = []
        preprocessed_context = self._preprocess_context(context)
        
        for question in questions:
            result = self.answer_question(question, preprocessed_context)
            results.append(result)
        
        return results

# Usage example with performance monitoring
def main():
    # Initialize optimized pipeline
    qa_system = OptimizedQAPipeline()
    
    # Enhanced context with more information
    context = """
    Mahmoud Abuawd is an AI Engineer who builds systems for computer vision and NLP.
    He has experience with TensorFlow, PyTorch, and Hugging Face models.
    He specializes in transformer architectures and has worked on projects involving
    image classification, object detection, sentiment analysis, and question answering systems.
    He holds a degree in Computer Science and has published research on neural networks.
    """
    
    # Interactive loop with performance metrics
    print("Optimized QA System Ready!")
    print("Type 'quit' to exit, 'batch' to test batch processing")
    print("-" * 50)
    
    while True:
        question = input("\nAsk a question:\n> ")
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'batch':
            # Test batch processing
            test_questions = [
                "What is Mahmoud's profession?",
                "What frameworks does he use?",
                "What is his educational background?"
            ]
            
            print("\nBatch processing results:")
            start_time = time.time()
            results = qa_system.batch_answer(test_questions, context)
            total_time = time.time() - start_time
            
            for i, (q, result) in enumerate(zip(test_questions, results)):
                print(f"\nQ{i+1}: {q}")
                print(f"A{i+1}: {result['answer']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Time: {result['inference_time']:.3f}s")
            
            print(f"\nTotal batch time: {total_time:.3f}s")
            continue
        
        # Get answer with performance metrics
        result = qa_system.answer_question(question, context)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Inference time: {result['inference_time']:.3f} seconds")

if __name__ == "__main__":
    main()