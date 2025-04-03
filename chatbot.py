from llama_cpp import Llama
import os

# Define the model path (update this if needed)
MODEL_PATH = os.path.expanduser("~/models/mistral-7b-instruct-v0.1.Q5_K_S.gguf")

# Load the model (MPS = Apple Silicon, CPU fallback)
print("Loading model, please wait...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, n_gpu_layers=0)  # Use more threads for faster response

def chat_with_llm(prompt):
    response = llm(prompt, max_tokens=200)
    return response["choices"][0]["text"]

# Simple Menu-Driven Chatbot
while True:
    print("\n1. Chat with LLM")
    print("2. Exit")
    choice = input("Enter choice: ")

    if choice == "1":
        user_input = input("\nYou: ")
        response = chat_with_llm(user_input)
        print("\nLLM:", response.strip())
    elif choice == "2":
        print("Exiting...")
        break
    else:
        print("Invalid choice, try again.")
