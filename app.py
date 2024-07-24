import os
from dotenv import load_dotenv
import dspy
import json

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI model
turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key=api_key)

# Initialize ColBERTv2
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Configure DSPy settings
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to intents.json
json_path = os.path.join(current_dir, 'intents.json')

# Open the file
with open(json_path, 'r') as f:
    data = json.load(f)


# Load the dataset
def load_dataset(file_name='intents.json'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Dataset loaded successfully. Number of intents: {len(data['intents'])}")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


# Verify the data is loaded correctly
print(f"Number of intents: {len(data['intents'])}")
print(f"First intent: {data['intents'][0]['tag']}")

# Preprocess the data
conversations = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        for response in intent['responses']:
            conversations.append({
                'input': pattern,
                'intent': intent['tag'],
                'response': response
            })

# Shuffle the conversations
import random
random.shuffle(conversations)

# Print a sample to verify
print(f"Sample conversation: {conversations[0]}")
print(f"Total number of conversation samples: {len(conversations)}")

#train and test data
train_data = conversations[:int(len(conversations) * 0.8)]
test_data = conversations[int(len(conversations) * 0.8):]


class IntentClassifier(dspy.Signature):
    """Classify the intent of the user's input."""
    user_input = dspy.InputField()
    intent = dspy.OutputField(desc="The classified intent of the user's input")

class ResponseGenerator(dspy.Signature):
    """Generate a response based on the classified intent."""
    intent = dspy.InputField()
    user_input = dspy.InputField()
    response = dspy.OutputField(desc="An appropriate response to the user's input")

class MentalHealthChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(IntentClassifier)
        self.generator = dspy.Predict(ResponseGenerator)

    def forward(self, user_input):
        intent = self.classifier(user_input=user_input)
        response = self.generator(intent=intent.intent, user_input=user_input)
        return intent.intent, response.response
    
chatbot = MentalHealthChatbot()

def chat_loop():
    print("Mental Health Chatbot: Hello! How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye', 'thanks']:
            print("Mental Health Chatbot: Take care! Remember, help is always available if you need it.")
            break

        intent, response = chatbot(user_input)
        print(f"Mental Health Chatbot: {response}")
        print(f"(Debug - Detected intent: {intent})")

if __name__ == "__main__":
    chat_loop()

def evaluate(chatbot):
    correct_intent = 0
    correct_response = 0
    total = len(test_data)

    for conv in test_data:
        intent, response = chatbot(conv['input'])
        if intent.lower() == conv['intent'].lower():
            correct_intent += 1
        if response.lower() == conv['response'].lower():
            correct_response += 1

    intent_accuracy = correct_intent / total
    response_accuracy = correct_response / total

    print(f"Intent Classification Accuracy: {intent_accuracy:.2f}")
    print(f"Response Generation Accuracy: {response_accuracy:.2f}")

# Run evaluation
#evaluate(chatbot)

import streamlit as st
import dspy
# Import your chatbot class or module here
# from your_chatbot_module import MentalHealthChatbot

# Initialize your chatbot
# Ensure this is done outside of any function to avoid reinitializing on every interaction
#lm = dspy.OpenAI(model='gpt-3.5-turbo')
#dspy.settings.configure(lm=lm)
#chatbot = MentalHealthChatbot()  # Replace with your actual chatbot initialization

def get_chatbot_response(user_input):
    # Modify this function to match your chatbot's interface
    intent, response = chatbot(user_input)
    return response

st.set_page_config(page_title="MindEase", page_icon="🧠", layout="centered")

st.title("MindEase - your mental health chatbot! 🤖💚")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's on your mind?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_chatbot_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information and mental health resources
st.sidebar.title("About")
st.sidebar.info("This chatbot is designed to provide support and information about mental health. Remember, it's not a substitute for professional help.")

st.sidebar.title("Mental Health Resources")
st.sidebar.markdown("""
- iCall Free Counsellng Helpline: 9152987821
- Crisis Text Line: Text HOME to 741741
- [Useful Resources for Mental Health](https://mhfaindia.com/cms/useful-resources-for-mental-health)
- [A Comprehensive Analysis of Mental Health Problems in India](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10460242/)
""")

st.sidebar.title("Disclaimer")
st.sidebar.warning("This chatbot is for informational purposes only. If you're experiencing a mental health emergency, please call your local emergency services or a mental health crisis hotline immediately.")

#test
st.write("Loading dataset...")
dataset = load_dataset()
st.write("Dataset loaded")

st.write("Initializing models...")
# Your model initialization code
st.write("Models initialized")