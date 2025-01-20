from abc import ABC, abstractmethod

class Plugin(ABC):
    @property
    @abstractmethod
    def command(self) -> str:
        """
        The flag or string that triggers this plugin. For example, '-c' or '--export'.
        """
        pass

    @abstractmethod
    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        """
        This method must either return:
          - None, if the plugin handled everything and we do not want 
            to continue with the normal chat flow,
          - OR a modified user_input string, which the main program will
            pass into the normal chat workflow.
        """
        pass
