
This document describes how to write a **plugin-based** chat assistant using **OpenAI** and a minimal plugin interface. It includes:

1. Project structure overview  
2. Four key files:
   - **main.py**  
   - **plugin_base.py**  
   - **plugins/__init__.py**  
   - **plugins/example plugins** (e.g. `copy_response.py`, `fix_code.py`)

---

## Project Layout

```
a0/
  ├── main.py
  ├── plugin_base.py
  └── plugins/
      ├── __init__.py
      ├── copy_response.py
      └── fix_code.py
```

- **`main.py`**: The main script that runs the chat loop, loads plugins, and streams responses from OpenAI.  
- **`plugin_base.py`**: Defines the abstract base `Plugin` class that all plugins must extend.  
- **`plugins`** folder: Contains individual plugin files. Each plugin provides a different **command** (flag).

---

## 1. `main.py`

```python
import sys
import os
import shutil
import subprocess
import importlib.util
import inspect

# If you haven't already:
# pip install openai

from openai import OpenAI
from plugin_base import Plugin

# In-memory chat state
chat_history = []

# ANSI colors for printing
COLOR_USER = "\033[0m"       # Reset color
COLOR_ASSISTANT = "\033[94m" # Blue for assistant text

def getTermWidth():
    """Return the current width of the terminal; default to 100 on error."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 100

def copyClipboard(text: str):
    """
    Copy given text to the clipboard on macOS, Windows, or Linux.
    Make sure you have pbcopy (mac), clip (win), or xclip (linux).
    """
    platform = sys.platform
    if platform.startswith("darwin"):
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif platform.startswith("win"):
        subprocess.run("clip", universal_newlines=True, input=text, check=True)
    else:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)

def load_plugins(folder="plugins"):
    """
    Dynamically import all .py files in the given folder,
    find classes that extend `Plugin`, instantiate them,
    and store them keyed by their command (e.g., "-c").
    """
    plugins = {}
    for filename in os.listdir(folder):
        if filename.endswith(".py") and not filename.startswith("__"):
            plugin_path = os.path.join(folder, filename)
            module_name = filename[:-3]  # strip .py extension

            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Inspect each attribute in the module;
            # if it's a subclass of Plugin, instantiate it
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    instance = attr()
                    cmd = instance.command
                    plugins[cmd] = instance

    return plugins

def apply_plugin_if_needed(user_input: str, plugins: dict):
    """
    Check if user_input starts with or equals a known plugin's command.
    If so, call the plugin's run() and handle the return value:
      - If plugin returns None, skip normal OpenAI logic.
      - Otherwise, proceed with the returned text as the new user_input.
    """
    first_token = user_input.split(" ", 1)[0]
    if first_token in plugins:
        plugin = plugins[first_token]
        result = plugin.run(user_input, chat_history, copyClipboard)
        if result is None:
            return None
        else:
            user_input = result

    return user_input

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

    # Replace 'gpt-4o' with your desired model, e.g. 'gpt-3.5-turbo'
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
                # If not the very first line, move to a new line
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

            updated_input = apply_plugin_if_needed(user_input, plugins)

            # If the plugin returned None, skip normal flow
            if updated_input is None:
                continue

            # Otherwise, proceed with the updated_input
            print()  # Blank line for clarity
            stream_response(updated_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
```

---

## 2. `plugin_base.py`

```python
from abc import ABC, abstractmethod

class Plugin(ABC):
    @property
    @abstractmethod
    def command(self) -> str:
        """
        The short flag or string that triggers this plugin. Example: '-c'.
        """
        pass

    @abstractmethod
    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        """
        Perform your custom logic here. Return:
          - None to stop further processing (the plugin handled everything)
          - A modified user_input string if you want the main script to
            continue with that text, eventually calling OpenAI.
        """
        pass
```

---

## 3. `plugins/__init__.py`

```python
# This file can be empty, but must exist so Python treats
# the plugins folder as a package.
```

*(Just create an empty file named `__init__.py` in the `plugins` directory.)*

---

## 4. Example Plugins

### 4a. `plugins/copy_response.py`

```python
from plugin_base import Plugin

class CopyResponsePlugin(Plugin):
    @property
    def command(self) -> str:
        return "-c"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        """
        Copies the last assistant response to the clipboard
        and stops (returns None) so we don't call OpenAI this turn.
        """
        if chat_history:
            last_response = chat_history[-1]["response"]
            copyClipboard(last_response)
            print("Last response copied to clipboard.")
        else:
            print("No response to copy yet.")

        return None
```

### 4b. `plugins/fix_code.py`

```python
from plugin_base import Plugin

class FixCodePlugin(Plugin):
    @property
    def command(self) -> str:
        return "-f"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        """
        Appends instructions so the AI responds with code only.
        We return the new user_input so the main script
        proceeds to call the AI model with these instructions.
        """
        user_input += (
            "!!! ONLY respond with a code fix, no explanation text. "
            "The goal is to be able to copy it right into source code. "
            "Do NOT wrap it in markdown!!!"
        )
        return user_input
```

---

# Usage Instructions

1. **Install Dependencies**  
   From your `a0/` directory (where `main.py` is located):
   ```bash
   pip install openai
   ```
   If you’re on Linux and want clipboard copying:
   ```bash
   sudo apt-get install xclip
   ```
   On macOS, `pbcopy` is built in; on Windows, `clip` is built in.

2. **Set Your API Key**  
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   (Use `set OPENAI_API_KEY=sk-...` on Windows CMD or `$Env:OPENAI_API_KEY="sk-..."` on PowerShell.)

3. **Run the Script**  
   ```bash
   python main.py
   ```

4. **Test the Plugins**  
   - Type `"-c"` to copy the assistant’s last message to your clipboard (if one exists).
   - Type `"-f Hey can you fix this code?"` to instruct the assistant to respond with code only.
   - Type a normal question, e.g. `"Hello assistant!"`, to talk normally with GPT.

5. **Extend by Adding More Plugins**  
   - Create a new `.py` file in `plugins/`  
   - Subclass `Plugin`  
   - Define a unique `command` property (like `"-g"` or `"--help"`)  
   - Implement `run()` logic  

---

## Quick Plugin Development Notes

- The **first token** of user input is matched against the plugin’s `command`.  
  If the user types `"-f"` followed by more text, your plugin’s `run()` will get called.  
- If `run()` returns **`None`**, we **skip** calling OpenAI.  
- If `run()` returns a **string**, that becomes the new user prompt for OpenAI.  
- You can do anything in `run()`: parse arguments, manipulate `chat_history`, or even copy to clipboard.  

---

**That’s it!** Copy this file and save it as `A0PluginDocs.md` for a single downloadable reference.
