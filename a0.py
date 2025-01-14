import sys
import os
import shutil
import subprocess
from openai import OpenAI

client = OpenAI()

# Persistent chat state
chat_history = []

# ANSI escape codes for colors
COLOR_USER = "\033[0m"       # Reset to default
COLOR_ASSISTANT = "\033[94m" # Blue for assistant text

def get_terminal_width():
    """Return the current width of the terminal; default to 80 on error."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80

def copy_to_clipboard(text):
    """
    Copy given text to the clipboard on macOS, Windows, or Linux
    with built-in commands only.
    """
    platform = sys.platform
    if platform.startswith("darwin"):
        # macOS
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif platform.startswith("win"):
        # Windows
        subprocess.run("clip", universal_newlines=True, input=text, check=True)
    else:
        # Linux (requires xclip installed)
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)

def stream_response(prompt):
    """
    Get an OpenAI response in a streaming fashion, do a best-effort line wrap,
    and preserve newlines (which helps with lists and Markdown formatting).
    """
    global chat_history

    # Prepare the conversation so far
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for entry in chat_history:
        messages.append({"role": "user", "content": entry["prompt"]})
        messages.append({"role": "assistant", "content": entry["response"]})
    messages.append({"role": "user", "content": prompt})

    # We'll accumulate the full response here for later reference
    full_response = ""

    # Print in assistant color
    print(COLOR_ASSISTANT, end="")

    # For line-wrapping logic
    term_width = get_terminal_width()
    current_line_length = 0

    # Request a streaming completion
    for chunk in client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    ):
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            full_response += content

            # Split on newline characters first
            # so we can preserve explicit line breaks
            lines = content.split("\n")

            for line_index, line in enumerate(lines):
                # If it's not the very first line in this chunk,
                # we just hit a newline => move to a new line
                if line_index > 0:
                    sys.stdout.write("\n")
                    current_line_length = 0

                # Now split by spaces to do a simple word wrap
                words = line.split(" ")

                for i, word in enumerate(words):
                    # If not the first word, see if we can fit " word"
                    if i > 0:
                        if current_line_length + 1 + len(word) > term_width:
                            # Start a new line
                            sys.stdout.write("\n")
                            current_line_length = 0
                        else:
                            # Print a space
                            sys.stdout.write(" ")
                            current_line_length += 1

                    # If the word itself is longer than the line can hold
                    # in its remaining space, force a new line first
                    if len(word) + current_line_length > term_width and current_line_length != 0:
                        sys.stdout.write("\n")
                        current_line_length = 0

                    # Print the word
                    sys.stdout.write(word)
                    current_line_length += len(word)

            sys.stdout.flush()

    # Reset color, then a blank line
    print(COLOR_USER)
    print()

    # Store the full (unwrapped) response
    chat_history.append({"prompt": prompt, "response": full_response})

if __name__ == "__main__":
    print("Welcome to the streaming chat!")
    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.")
    print("Type '-c' (without quotes) to copy the last response.\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # If user wants to copy last assistant response
            if user_input == "-c":
                if chat_history:
                    last_response = chat_history[-1]["response"]
                    copy_to_clipboard(last_response)
                    print("Last response copied to clipboard.")
                else:
                    print("No response to copy yet.")
                continue

            print()  # Blank line for clarity
            stream_response(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
