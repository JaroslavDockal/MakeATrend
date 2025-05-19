import os
import datetime

class Logger:
    # Log levels
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    _instance = None

    def __init__(self):
        self.log_window = None
        self._log_file_path = None
        self._setup_log_file()

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

        # Log to a file
        if self._log_file_path:
            with open(self._log_file_path, "a") as log_file:
                log_file.write(formatted_message + "\n")

    def _setup_log_file(self):
        """Create the _logs directory and set up the log file path based on the current date-time."""
        # Create _logs directory if it doesn't exist
        logs_dir = "_logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Generate filename based on current date and time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self._log_file_path = os.path.join(logs_dir, f"Session_{timestamp}.log")

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

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance
