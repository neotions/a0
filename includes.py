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

# near the top of main.py:
from plugins.vectorDB import ChromaManager

def store_QA_in_chroma(question: str, answer: str):
    """
    Combine the user's question & assistant's answer into one doc.
    Persist in Chroma so we can retrieve them later.
    """
    coll = ChromaManager.get_collection()
    doc_id = str(uuid.uuid4())
    combined = f"Q: {question}\nA: {answer}"
    coll.add(
        documents=[combined],
        metadatas=[{"source": "QApair"}],
        ids=[doc_id]
    )
    ChromaManager.persist()

def getTermWidth():
    """Return the current width of the terminal; default to 100 on error."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 100

def copyClipboard(text: str):
    """
    Copy given text to the clipboard on macOS, Windows, or Linux.
    """
    platform = sys.platform
    if platform.startswith("darwin"):
        # macOS
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif platform.startswith("win"):
        # Windows
        subprocess.run("clip", universal_newlines=True, input=text, check=True)
    else:
        # Linux (requires xclip)
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)

def load_plugins(folder="plugins"):
    """
    Dynamically import all .py files in the specified folder.
    Look for classes that inherit from Plugin, instantiate them,
    and store them in a dict keyed by their plugin.command property.
    """
    plugins = {}

    for filename in os.listdir(folder):
        if filename.endswith(".py") and not filename.startswith("__"):
            plugin_path = os.path.join(folder, filename)
            module_name = filename[:-3]  # strip .py extension

            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # find classes that inherit from Plugin (but are not Plugin itself)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr) 
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    # instantiate the plugin
                    instance = attr()
                    cmd = instance.command
                    plugins[cmd] = instance
                    # After we instantiate the plugin

    print ("\033[92m" + f'Loaded {len(plugins.items())} plugins' + "\033[0m")
    print()

    return plugins

def apply_plugin_if_needed(user_input: str, plugins: dict, chat_history):
    """
    Check if user_input starts with or equals a known plugin's command.
    If so, call the plugin's run() and handle the return value:
      - If plugin returns None, skip normal OpenAI logic.
      - Otherwise, proceed with the returned text as the new user_input.
    """
    # We’ll split on space to get the first token
    first_token = user_input.split(" ", 1)[0]

    if first_token in plugins:
        plugin = plugins[first_token]
        result = plugin.run(user_input, chat_history, copyClipboard)

        if result is None:
            # Plugin handled it fully, skip
            return None
        else:
            # The plugin might have changed user_input
            user_input = result

    return user_input