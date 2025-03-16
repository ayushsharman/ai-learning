import random

intents = {
    'greetings': ['hello', 'hi', 'hey', 'how are you?', 'whats up?'],
    'goodbye': ['bye', 'goodbye', 'see you later', 'take care'],
    'thanks': ['thank you', 'thanks', 'thank you so much'],
}

def get_response(messages):
    messages = messages.lower()
    
    if("hello" in messages or "hi" in messages or "hey" in messages or "whats up?" in messages):
        return random.choice(intents['greetings'])
    elif("bye" in messages or "goodbye" in messages or "see you later" in messages or "take care" in messages):
        return random.choice(intents['goodbye'])
    elif("thank" in messages or "thanks" in messages or "thank you" in messages or "thank you so much" in messages):
        return random.choice(intents['thanks'])
    else:
        return "I'm sorry, I don't understand that."
    
def chatbot():
    print("Hello! I'm a chatbot. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if(user_input.lower() == "exit"):
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print("Chatbot: ", response)
        
if __name__ == "__main__":
    chatbot()
