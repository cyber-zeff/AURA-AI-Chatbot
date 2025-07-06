from dotenv import load_dotenv
load_dotenv()

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
)
from agents.run import RunConfig
import os
import json

def main():
    set_tracing_disabled(True)

# --- Config ---
MAX_HISTORY_LENGTH = 40
MEMORY_FOLDER = "agents_memory"

# Ensure memory folder exists
os.makedirs(MEMORY_FOLDER, exist_ok=True)

# --- Helpers ---
def get_memory_file(agent_key):
    return os.path.join(MEMORY_FOLDER, f"{agent_key}.json")

def load_conversation(memory_file):
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_conversation(history, memory_file):
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def format_history_for_prompt(history):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

# --- Setup API ---
api_key = os.getenv("GEMINI_API_KEY")
print("API Key Loaded:", bool(api_key))

provider = AsyncOpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=provider,
)

# --- Define All Agents ---
agents = {
    # A-Levels
    "a_lev_phy": Agent(
        name="AURA",
        instructions="You are a professional physics teacher for A'levels Students. You remember everything the user has said earlier in this conversation. Use it to respond naturally.",
        model=model,
    ),
    "a_lev_chem": Agent(
        name="CHEMIX",
        instructions="You are a professional chemistry teacher for A'levels Students. You remember everything the user has said earlier in this conversation. Use it to respond naturally.",
        model=model,
    ),
    "a_lev_math": Agent(
        name="MATHY",
        instructions="You are a professional mathematics teacher for A'levels Students. You remember everything the user has said earlier in this conversation. Use it to respond naturally.",
        model=model,
    ),
    # O-Levels
    "o_lev_phy": Agent(
        name="O-PHY",
        instructions="You are a professional physics teacher for O'levels Students. You remember everything the user has said earlier in this conversation. Use it to respond naturally.",
        model=model,
    ),
    "o_lev_chem": Agent(
        name="O-CHEM",
        instructions="You are a professional chemistry teacher for O'levels Students. You remember everything the user has said earlier in this conversation. Use it to respond naturally.",
        model=model,
    ),
    "o_lev_math": Agent(
        name="O-MATH",
        instructions="You are a professional mathematics teacher for O'levels Students. You remember everything the user has said earlier in this conversation. Use it to respond naturally.",
        model=model,
    ),
}

# --- Prompt for Level ---
print("Select the academic level:")
print("1. A'Level")
print("2. O'Level")
level_choice = input("Enter choice (1/2): ").strip()

if level_choice == "1":
    level_prefix = "a_lev"
elif level_choice == "2":
    level_prefix = "o_lev"
else:
    print("Invalid choice. Exiting.")
    exit()

# --- Prompt for Subject ---
print("\nSelect a subject:")
print("1. Physics")
print("2. Chemistry")
print("3. Math")
subject_choice = input("Enter choice (1/2/3): ").strip()

subject_map = {
    "1": "phy",
    "2": "chem",
    "3": "math",
}

if subject_choice not in subject_map:
    print("Invalid subject choice. Exiting.")
    exit()

subject_suffix = subject_map[subject_choice]
agent_key = f"{level_prefix}_{subject_suffix}"
agent = agents[agent_key]
memory_file = get_memory_file(agent_key)

print(f"\nYou selected: {agent_key.replace('_', ' ').upper()} Agent\n")

# --- Load Memory ---
conversation_history = load_conversation(memory_file)

# --- Run Config ---
config = RunConfig(tracing_disabled=True)

print("Assistant is ready to help you. Type 'quit' to exit.\n")

# --- Chat Loop ---
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Add user input
    conversation_history.append({"role": "user", "content": user_input})

    # Build full prompt from memory
    full_prompt = format_history_for_prompt(conversation_history)

    # Get assistant reply
    response = Runner.run_sync(
        starting_agent=agent, input=full_prompt, run_config=config
    )

    assistant_reply = response.final_output
    print("Assistant:", assistant_reply)

    # Save assistant reply
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # Trim if needed
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]

    # Save to memory file
    save_conversation(conversation_history, memory_file)
