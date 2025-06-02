import os
import datetime
import threading


class Logger:
    # Log levels
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    # Filter for log detail level (0-5, higher = more detail)
    debug_filter = 0
    # Whether to display thread name in log messages
    display_thread_name = True

    _instance = None

    def __init__(self):
        self.log_window = None
        self._log_file_path = None
        self._setup_log_file()

    def set_log_window(self, log_window):
        self.log_window = log_window

    def log_message(self, message, level=INFO, logLevel=5):
        """
        Logs a message to the log window, console, and optionally a file.

        Args:
            message (str): The message to log.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
            logLevel (int): Detail level from 0 (least detailed) to 5 (most detailed)
                            Only messages with logLevel <= debug_filter will be displayed
        """
        # Skip DEBUG messages if their detail level is too high
        if level == Logger.DEBUG and logLevel > Logger.debug_filter:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thread_name = threading.current_thread().name

        # Get level name for the message
        level_names = {
            Logger.DEBUG: "DEBUG",
            Logger.INFO: "INFO",
            Logger.WARNING: "WARNING",
            Logger.ERROR: "ERROR"
        }
        level_name = level_names.get(level, "INFO")

        # Format the message with timestamp, level, and thread name if enabled
        if self.display_thread_name:
            formatted_message = f"[{timestamp}] {level_name}: {message}; Thread: {thread_name}"
        else:
            formatted_message = f"[{timestamp}] {level_name}: {message}"

        # Log to console
        print(formatted_message)

        # Log to the log window if it exists
        if self.log_window:
            self.log_window.add_message(formatted_message, level, logLevel)

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
    def log_message_static(message, level=INFO, logLevel=5):
        """
        Static method to log messages globally.
        Can be called from anywhere in the code.

        Args:
            message (str): The message to log.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
            logLevel (int): Detail level from 0 (least detailed) to 5 (most detailed)
                            Only messages with logLevel <= detail_filter will be displayed
        """
        instance = Logger.get_instance()
        instance.log_message(message, level, logLevel)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance
