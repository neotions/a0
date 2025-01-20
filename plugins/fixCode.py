from plugin_base import Plugin

class FixCodePlugin(Plugin):
    @property
    def command(self) -> str:
        return "-f"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        user_input += (
            "!!! ONLY respond with a code fix, no explanation text. "
            "The goal is to be able to copy it right into source code. "
            "Do NOT wrap it in markdown!!!"
        )
        return user_input
