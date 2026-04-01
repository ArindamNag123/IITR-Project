"""
Language Translator Module
Translates queries between English, Hindi, and Thai for chatbot applications.
Supports bidirectional translation: English <-> Hindi <-> Thai
"""

from transformers import MarianMTModel, MarianTokenizer
import torch

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class MultilingualTranslator:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Single model for all languages
        model_name = "facebook/m2m100_418M"  # or "facebook/m2m100_1.2B" for better quality
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Language codes
        self.lang_codes = {
            'en': 'en',  # English
            'hi': 'hi',  # Hindi  
            'th': 'th'   # Thai
        }
    
    def translate(self, text, source_lang, target_lang):
        # Set source language
        self.tokenizer.src_lang = self.lang_codes[source_lang]
        
        # Tokenize
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate translation with target language token
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(self.lang_codes[target_lang])
        )
        
        # Decode
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

class LanguageTranslator:
    """
    Handles translation between English, Hindi languages
    Using Meta's M2M100 model or Helsinki-NLP Models for optimal translation
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the translator with necessary models
        
        Args:
            use_gpu (bool): Whether to use GPU for faster translation (if available)
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        
        # Define translation model pairs
        # Using Helsinki-NLP models for better quality
        self.model_configs = {
            'en_to_hi': 'Helsinki-NLP/opus-mt-en-hi',
            'hi_to_en': 'Helsinki-NLP/opus-mt-hi-en',
            # 'en_to_th': 'Helsinki-NLP/opus-mt-en-th',
            # 'th_to_en': 'Helsinki-NLP/opus-mt-th-en',
        }
        
        print(f"Initializing translator on {self.device}...")

        
    def load_model(self, model_name):
        """Load a specific model if not already loaded"""
        if model_name not in self.models:
            print(f"Loading model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(self.model_configs[model_name])
            model = MarianMTModel.from_pretrained(self.model_configs[model_name])
            model.to(self.device)
            
            self.tokenizers[model_name] = tokenizer
            self.models[model_name] = model
    
    def translate(self, text, source_lang, target_lang):
        """
        Translate text from source language to target language
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code ('en', 'hi', 'th')
            target_lang (str): Target language code ('en', 'hi', 'th')
        
        Returns:
            str: Translated text
        """
        if source_lang == target_lang:
            return text
        
        # Determine model name
        model_key = f'{source_lang}_to_{target_lang}'
        
        if model_key not in self.model_configs:
            raise ValueError(f"Translation pair {source_lang} -> {target_lang} not supported")
        
        # Load model if necessary
        self.load_model(model_key)
        
        # Tokenize
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            translated = model.generate(**inputs)
        
        # Decode result
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return result[0] if result else text
    
    def translate_english_to_hindi(self, text):
        """Translate English to Hindi"""
        return self.translate(text, 'en', 'hi')
    
    # def translate_english_to_thai(self, text):
    #     """Translate English to Thai"""
    #     return self.translate(text, 'en', 'th')
    
    def translate_hindi_to_english(self, text):
        """Translate Hindi to English"""
        return self.translate(text, 'hi', 'en')
    
    # def translate_thai_to_english(self, text):
    #     """Translate Thai to English"""
    #     return self.translate(text, 'th', 'en')


class ChatbotTranslator:
    """
    Specialized translator for chatbot queries
    Detects language and translates accordingly
    """
    
    def __init__(self, use_gpu=False):
        """Initialize the chatbot translator"""
        self.translator = LanguageTranslator(use_gpu=use_gpu)
    
    def detect_language(self, text):
        """
        Simple language detection based on character ranges
        
        Args:
            text (str): Text to detect language from
        
        Returns:
            str: Detected language code ('en', 'hi', 'th')
        """
        # Devanagari script range (Hindi)
        if any('\u0900' <= c <= '\u097F' for c in text):
            return 'hi'
        
        # Thai script range
        if any('\u0E00' <= c <= '\u0E7F' for c in text):
            return 'th'
        
        # Default to English
        return 'en'
    
    def translate_to_english(self, query):
        """
        Translate chatbot query to English
        Auto-detects source language
        
        Args:
            query (str): User query in any supported language
        
        Returns:
            dict: Contains original, detected_language, and english_translation
        """
        detected_lang = self.detect_language(query)
        
        if detected_lang == 'en':
            english_query = query
        else:
            english_query = self.translator.translate(query, detected_lang, 'en')
        
        return {
            'original': query,
            'detected_language': detected_lang,
            'english_translation': english_query
        }
    
    def translate_from_english(self, response, target_language):
        """
        Translate chatbot response from English to target language
        
        Args:
            response (str): Response in English
            target_language (str): Target language code
        
        Returns:
            dict: Contains original and translation
        """
        if target_language not in ['hi', 'th', 'en']:
            raise ValueError("Target language must be 'hi', 'th', or 'en'")
        
        if target_language == 'en':
            translated = response
        else:
            translated = self.translator.translate(response, 'en', target_language)
        
        return {
            'original': response,
            'target_language': target_language,
            'translation': translated
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Language Translator Demo")
    print("=" * 60)
    
    # Create translator
    print("\nInitializing translators...")
    chatbot_translator = ChatbotTranslator(use_gpu=False)
    fb_translator = MultilingualTranslator(use_gpu=False)
    
    # Example queries
    test_queries = [
        "Hello, how are you?",  # English
        "नमस्ते, आप कैसे हैं?",  # Hindi: "Hello, how are you?"
        # "สวัสดี คุณเป็นอย่างไรบ้าง",  # Thai: "Hello, how are you?"
        "Hi, my name is Akilesh and I am learning NLP. I am from Chennai. Tell me what is the prize of this item?"  # English
    ]
    
    print("\n" + "=" * 60)
    print("TRANSLATING TO ENGLISH")
    print("=" * 60)
    
    for query in test_queries:
        result = chatbot_translator.translate_to_english(query)
        result2 = fb_translator.translate(query, result['detected_language'], 'en')
        print(f"\nOriginal: {result['original']}")
        print(f"Language: {result['detected_language']}")
        print(f"English: {result['english_translation']}")
        print(f"English Translation from FB translator: {result2}")
    
    print("\n" + "=" * 60)
    print("TRANSLATING FROM ENGLISH")
    print("=" * 60)
    
    response = "Hi, my name is Akilesh and I am learning NLP. I am from Chennai. Tell me what is the prize of this item?"  # English
    
    for target_lang in ['hi']:
        result = chatbot_translator.translate_from_english(response, target_lang)
        result2 = fb_translator.translate(response, 'en', target_lang)
        print(f"\nOriginal (English): {result['original']}")
        print(f"Target Language: {result['target_language']}")
        print(f"Translation: {result['translation']}")
        print(f"Hindi Translation from FB translator: {result2}")
