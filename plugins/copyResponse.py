from plugin_base import Plugin

class CopyResponsePlugin(Plugin):
    @property
    def command(self) -> str:
        return "-c"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        if chat_history:
            last_response = chat_history[-1]["response"]
            copyClipboard(last_response)
            print("Last response copied to clipboard.")
        else:
            print("No response to copy yet.")

        # Return None to indicate we do NOT continue with normal OpenAI query.
        return None
