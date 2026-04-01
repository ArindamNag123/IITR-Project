"""
Chatbot Integration Example
Shows how to use the LanguageTranslator with a chatbot
"""

from src.ml.translator import ChatbotTranslator


class MultilingualChatbot:
    """
    Example chatbot that supports English, Hindi, and Thai
    """
    
    def __init__(self):
        self.translator = ChatbotTranslator(use_gpu=False)
        self.chat_history = []
    
    def process_user_query(self, user_input):
        """
        Process user input in any supported language
        
        Args:
            user_input (str): User query in English, Hindi, or Thai
        
        Returns:
            dict: Contains original query, English translation, and response
        """
        # Translate to English
        translation_result = self.translator.translate_to_english(user_input)
        english_query = translation_result['english_translation']
        original_language = translation_result['detected_language']
        
        print(f"\n📝 User Input ({original_language.upper()}): {user_input}")
        print(f"🔄 Translated to English: {english_query}")
        
        # Process the English query (you would implement actual chatbot logic here)
        chatbot_response = self._generate_response(english_query)
        
        # Translate response back to original language
        if original_language != 'en':
            response_translation = self.translator.translate_from_english(
                chatbot_response, 
                original_language
            )
            final_response = response_translation['translation']
        else:
            final_response = chatbot_response
        
        print(f"💬 Response ({original_language.upper()}): {final_response}")
        
        return {
            'original_query': user_input,
            'detected_language': original_language,
            'english_query': english_query,
            'response': final_response,
            'response_english': chatbot_response
        }
    
    def _generate_response(self, english_query):
        """
        Generate chatbot response based on English query
        This is a placeholder - implement your actual chatbot logic here
        
        Args:
            english_query (str): Query in English
        
        Returns:
            str: Response in English
        """
        # Simple rule-based responses for demo
        english_query_lower = english_query.lower()
        
        if any(word in english_query_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        
        elif any(word in english_query_lower for word in ['how are you', 'whats up', 'how do you do']):
            return "I'm doing great, thank you for asking! How can I assist you?"
        
        elif any(word in english_query_lower for word in ['thank', 'thanks', 'thankyou']):
            return "You're welcome! Is there anything else I can help you with?"
        
        elif any(word in english_query_lower for word in ['weather', 'temperature']):
            return "I don't have real-time weather data, but you can check a weather website for current conditions."
        
        elif any(word in english_query_lower for word in ['time', 'what time', 'current time']):
            from datetime import datetime
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
        
        else:
            return "I understand your query, but I need more context to provide a helpful response. Could you provide more details?"


def main():
    """Main function to demonstrate the multilingual chatbot"""
    
    print("=" * 70)
    print("MULTILINGUAL CHATBOT - ENGLISH, HINDI, THAI TRANSLATION DEMO")
    print("=" * 70)
    
    chatbot = MultilingualChatbot()
    
    # Example conversations in different languages
    user_queries = [
        "Hello, how are you?",           # English
        "नमस्ते, आप कैसे हैं?",            # Hindi: "Hello, how are you?"
        "Thank you so much!",            # English
        "धन्यवाद!",                       # Hindi: "Thank you!"
        "สวัสดี",                        # Thai: "Hello"
        "What's the time?",              # English
        "समय क्या है?",                   # Hindi: "What time is it?"
    ]
    
    for query in user_queries:
        try:
            chatbot.process_user_query(query)
            print("-" * 70)
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("-" * 70)


if __name__ == "__main__":
    main()
