"""
Language Translator Module - Agentic LLM-based Translation
Supports both local LLMs (GPT-2, etc.) and OpenAI's ChatGPT API
"""

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod
import os
import re

from dotenv import load_dotenv
load_dotenv()


class LLMTranslator(ABC):
    """Abstract base class for LLM-based translators"""
    
    def __init__(self):
        self.detected_language = None
        self.english_pattern = re.compile(r'^[A-Za-z0-9\s.,!?\'"-]+$')
        self.lang_codes = {
            'en': 'English',
            'hi': 'Hindi',
            'th': 'Thai'
        }

    
    def detect_language(self, text):
        """
        Detect language using character-based heuristics (NOT through the LLM).
        This is fast and doesn't require model inference.
        
        Args:
            text (str): Text to detect language from
        
        Returns:
            str: Detected language code ('en', 'hi', 'th')
        """
        # Devanagari script range (Hindi)
        if any('\u0900' <= c <= '\u097F' for c in text):
            return 'hi'
        
        # Default to English
        return 'en'
    
    ## Latin script for now
    def detect_english(self, text):
        return bool(self.english_pattern.match(text))
    
    @abstractmethod
    def translate_with_llm(self, text, source_lang, target_lang):
        """Translate text using the LLM - implemented by subclasses"""
        pass
    
    def process_query(self, query):
        """
        AGENT STEP 1: Process user query
        - Detect language (character-based) -- optional
        - Store detected language internally
        - Translate to English using LLM if needed
        
        Args:
            query (str): User query in any supported language
        
        Returns:
            str: Query translated to English
        """
        # Step 1: Detect language
        self.detected_language = self.detect_language(query)
        
        # Step 2: Translate to English if needed
        if self.detected_language == 'en':
            return query
        else:
            return self.translate_with_llm(query, self.detected_language, 'en')
    
    def translate_response_back(self, response):
        """
        AGENT STEP 3: Translate response back to user's language
        Uses the stored detected language from process_query()
        
        Args:
            response (str): Response in English (from other tools)
        
        Returns:
            str: Response in user's original language
        """
        if self.detected_language is None:
            raise ValueError("No query has been processed yet. Call process_query() first.")
        
        if self.detected_language == 'en':
            return response
        else:
            return self.translate_with_llm(response, 'en', self.detected_language)
    
    def get_detected_language(self):
        """Get the detected language from the last query processed."""
        return self.detected_language


class LocalLLMTranslator(LLMTranslator):
    """Translator using local LLM models (GPT-2, distilgpt2, etc.)"""
    
    def __init__(self, model_name="gpt2", use_gpu=False):
        """
        Initialize the local LLM translator.
        
        Args:
            model_name (str): HuggingFace model name. Options:
                - "gpt2" (lightweight)
                - "distilgpt2" (smallest)
                - "EleutherAI/gpt-neo-125m" (better quality)
            use_gpu (bool): Whether to use GPU
        """
        super().__init__()
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print(f"Loading local LLM: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Set pad token for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def translate_with_llm(self, text, source_lang, target_lang):
        """Translate text using the local generative LLM via prompt engineering."""
        if source_lang == target_lang:
            return text
        
        source_lang_name = self.lang_codes[source_lang]
        target_lang_name = self.lang_codes[target_lang] 
        
        # Craft the translation prompt
        prompt = f"""Translate the following text 
        from {source_lang_name} to {target_lang_name}. Only output the translation of the text, 
        and nothing else. Here is the following text: {text}"""
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate translation
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + 80,
                num_beams=3,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the translation part
        translation = generated_text.split(f"{target_lang}:")[-1].strip()
        
        return translation if translation else text


class OpenAITranslator(LLMTranslator):
    """Translator using OpenAI's ChatGPT API"""
    
    def __init__(self, api_key=None, model="gpt-4"):
        """
        Initialize the OpenAI translator.
        
        Args:
            api_key (str): OpenAI API key. If None, reads from OPENAI_API_KEY env var
            model (str): Model to use. Options:
                - "gpt-3.5-turbo" (faster, cheaper)
                - "gpt-4" (more capable)
        """
        super().__init__()
        
        try:
            import openai
            self.openai = openai
            print("=" * 70)
            print("AGENTIC TRANSLATOR - OpenAI ChatGPT")
            print("=" * 70)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        self.openai.api_key = self.api_key
        self.model = model
        
        print(f"OpenAI Translator initialized with model: {self.model}")

    
    def translate_with_llm(self, text, _source_lang=None, target_lang='English', reference_text=None):
        """Translate text using OpenAI's ChatGPT API."""
        """
        LLM-driven translation without explicit language codes.

        Args:
            text (str): text to translate
            target (str): "english" OR "original"
            reference_text (str): original user input (for translating back)
        """

        if target_lang in ["English", "en"]:
            system_msg = "You are a professional translator. Detect the input language and translate it to English. Only output the translated text."
            user_msg = f"Translate this text to English: {text}"

        elif target_lang == "original":
            system_msg = "You are a professional translator."
            user_msg = f"""
                Translate the following English text back to the SAME language as this original text.

                Original text: {reference_text}
                English text: {text}

                Only output the translated text.
                """

        # Create the translation prompt
        # prompt = f"""Translate the following text 
        # from {source_lang_name} to {target_lang_name}. Only output the translation of the text, 
        # and nothing else. Here is the following text: {text}"""
            
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": user_msg
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            translation = response.choices[0].message.content.strip()
            return translation if translation else text
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback: return original text if API fails
            return text


class ChatbotTranslator:
    """
    Specialized translator for chatbot queries using agentic LLM.
    
    Supports both local LLMs and OpenAI's ChatGPT.
    
    Usage flow:
    1. Call translate_to_english() with user query
    2. Process the English query with other tools/agents
    3. Call translate_from_english() to get response in user's language
    """
    
    def __init__(self, use_openai=False, api_key=None, model=None, use_gpu=False):
        """
        Initialize the chatbot translator.
        
        Args:
            use_openai (bool): Whether to use OpenAI (True) or local LLM (False)s
            api_key (str): OpenAI API key (only if use_openai=True)
            model (str): Model to use
                - For OpenAI: "gpt-3.5-turbo" or "gpt-4"
                - For local: "gpt2", "distilgpt2", "EleutherAI/gpt-neo-125m"
            use_gpu (bool): Whether to use GPU (only for local LLMs)
        """
        if use_openai:
            # Use OpenAI ChatGPT
            openai_model = model or "gpt-4"
            self.translator = OpenAITranslator(api_key=api_key, model=openai_model)
        else:
            # Use local LLM
            local_model = model or "gpt2"
            self.translator = LocalLLMTranslator(model_name=local_model, use_gpu=use_gpu)
    
    def translate_to_english(self, query):
        """
        STEP 1: Translate user query to English and store detected language.
        
        Args:
            query (str): User query in any supported language
        
        Returns:
            dict: {
                'original': original query text,
                'english_translation': query in English
            }
        """
        english_query = self.translator.process_query(query)
        
        return {
            'original': query,
            'detected_language': self.translator.detected_language,
            'english_translation': english_query
        }
    
    def translate_from_english(self, response, initial_query=None):
        """
        STEP 3: Translate response back to user's language.
        
        Uses the language detected and stored in translate_to_english().
        
        Args:
            response (str): Response in English
        
        Returns:
            str: Response in user's original language
        """
        if not use_openai:
            return self.translator.translate_response_back(response)
        else:
            if initial_query:
                return self.translator.translate_with_llm(response, reference_text=initial_query)
            return response




# Example usage
# if __name__ == "__main__":
#     import sys
    
#     print("=" * 70)
#     print("AGENTIC TRANSLATOR - Local LLM vs OpenAI ChatGPT")
#     print("=" * 70)
    
#     # Choose translator
#     use_openai = True  # Set to True to use OpenAI API
    
#     if use_openai:
#         print("\nInitializing OpenAI ChatGPT translator...")
#         ## give your api key:
#         #  api_key "your key"
#         # Make sure OPENAI_API_KEY is set in your environment
#         translator = ChatbotTranslator(use_openai=True, model="gpt-3", api_key=api_key)
#     else:
#         print("\nInitializing local LLM translator (distilgpt2)...")
#         translator = ChatbotTranslator(use_openai=False, model="distilgpt2", use_gpu=False)
    
#     print("\n" + "=" * 70)
#     print("STEP 1: DETECT LANGUAGE & TRANSLATE TO ENGLISH")
#     print("=" * 70)

#     ## ***********************************************Testing Block****************************************************
#         # Test queries
#     test_queries = [
#         "Hello, how are you?",
#         "नमस्ते, आप कैसे हैं?",
#         "Hi, my name is Akilesh",
#     ]
    
#     results = []
#     if not use_openai:
#         for query in test_queries:
#             result = translator.translate_to_english(query)
#             results.append(result)
#             print(f"\n📝 Original ({result['detected_language'].upper()}): {result['original']}")
#             print(f"🔄 English Translation: {result['english_translation']}")
#     else:
#         for query in test_queries:
#             result = translator.translate_to_english(query)
#             results.append(result)
#             print(f"\n📝 Original: {result['original']}")
#             print(f"🔄 English Translation: {result['english_translation']}")
        
#     print("\n" + "=" * 70)
#     print("STEP 2: OTHER TOOLS PROCESS ENGLISH QUERY")
#     print("=" * 70)
#     print("(Simulating external tool processing...)")
    
#     english_responses = [
#         "Hello! I'm doing great, thank you!",
#         "नमस्ते! I am fine, thank you!",
#         "Nice to meet you!",
#     ]
    
#     print("\n" + "=" * 70)
#     print("STEP 3: TRANSLATE RESPONSE BACK TO USER'S LANGUAGE")
#     print("=" * 70)
    
#     for i, resp in enumerate(english_responses):
#         final_response = translator.translate_from_english(resp)
#         detected_lang = results[i]['detected_language'].upper()
#         print(f"\n✅ Response in {detected_lang}: {final_response}")

    ##*******************************************External usage******************************************************

    ## can be used directly under main function, with just the query and its translation:
    # translator = ChatbotTranslator(use_openai=True, model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    # result = translator.translate_to_english(query)
    ## here, the initial_query is the query given initially by the user, which is used as reference for translating back to original language
    # result = translator.translate_from_english(result, initial_query=query)

    ##***************************************************************************************************************
