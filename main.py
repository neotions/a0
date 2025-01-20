import sys
import os
import shutil
import subprocess
import importlib.util
import inspect
import uuid

from openai import OpenAI
from plugin_base import Plugin
from plugins.vectorDB import ChromaManager

from includes import *


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory chat state
chat_history = []

# ANSI colors
COLOR_USER = "\033[0m"       # Reset color
COLOR_ASSISTANT = "\033[94m" # Blue for assistant text



def stream_response(prompt: str):
    """
    Stream an OpenAI chat response with line wrapping.
    """
    global chat_history

    # Prepare conversation so far
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for entry in chat_history:
        messages.append({"role": "user", "content": entry["prompt"]})
        messages.append({"role": "assistant", "content": entry["response"]})
    messages.append({"role": "user", "content": prompt})

    # We'll build up the full response
    full_response = ""

    # Print in assistant color
    print(COLOR_ASSISTANT, end="")

    # For line-wrapping logic
    term_width = min(getTermWidth(), 100)
    current_line_length = 0

    # Request streaming from OpenAI
    for chunk in client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    ):
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            full_response += content

            # Split on newline characters first
            lines = content.split("\n")

            for line_index, line in enumerate(lines):
                # If not the very first line in this chunk, move to a new line
                if line_index > 0:
                    sys.stdout.write("\n")
                    current_line_length = 0

                # Split by spaces for word-wrap logic
                words = line.split(" ")

                for i, word in enumerate(words):
                    if i > 0:
                        if current_line_length + 1 + len(word) > term_width:
                            sys.stdout.write("\n")
                            current_line_length = 0
                        else:
                            sys.stdout.write(" ")
                            current_line_length += 1

                    if len(word) + current_line_length > term_width and current_line_length != 0:
                        sys.stdout.write("\n")
                        current_line_length = 0

                    sys.stdout.write(word)
                    current_line_length += len(word)

            sys.stdout.flush()

    # Reset color, then a blank line
    print(COLOR_USER)
    print()

    # Save the conversation
    chat_history.append({"prompt": prompt, "response": full_response})

if __name__ == "__main__":
    print("\na0 assistant powered by \033[33mGPT-4o\033[0m\n")

    # Load all plugins from the plugins folder
    plugins = load_plugins("plugins")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            updated_input = apply_plugin_if_needed(user_input, plugins, chat_history)

            # If the plugin returned None, skip normal flow
            if updated_input is None:
                continue

            # Otherwise, proceed with the updated_input
            print()  # Blank line for clarity
            stream_response(updated_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
