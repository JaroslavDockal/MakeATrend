# logger.py
import datetime

class Logger:
    # Log levels
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance

    def __init__(self):
        self.log_window = None

    def set_log_window(self, log_window):
        self.log_window = log_window

    def log_message(self, message, level=INFO):
        """
        Logs a message to the log window, console, and optionally a file.

        Args:
            message (str): The message to log.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get level name for the message
        level_names = {
            Logger.DEBUG: "DEBUG",
            Logger.INFO: "INFO",
            Logger.WARNING: "WARNING",
            Logger.ERROR: "ERROR"
        }
        level_name = level_names.get(level, "INFO")

        formatted_message = f"[{timestamp}] {level_name}: {message}"

        # Log to console
        print(formatted_message)

        # Log to the log window if it exists
        if self.log_window:
            self.log_window.add_message(formatted_message, level)

    @staticmethod
    def log_message_static(message, level=INFO):
        """
        Static method to log messages globally.
        Can be called from anywhere in the code.

        Args:
            message (str): The message to log.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
        """
        instance = Logger.get_instance()
        instance.log_message(message, level)