import google.generativeai as genai

# Set your API key
GOOGLE_API_KEY = "AIzaSyCAtIMQN7gp0p3WwdUF3tU_PLrECvFdskE"
genai.configure(api_key=GOOGLE_API_KEY)

# Use a chat-capable model
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# Start a chat session to maintain context
chat = model.start_chat()

def chat_with_bot():
    print("💬 Gemini Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting chat.")
            break

        response = chat.send_message(user_input)
        print("Gemini:", response.text, "\n")

if __name__ == "__main__":
    chat_with_bot()
