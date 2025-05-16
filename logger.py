class Logger:
    @staticmethod
    def log_message_static(message, level=INFO):
        """
        Static method to log messages globally via the SignalViewer instance.
        Can be called from anywhere in the code without a direct instance reference.

        Args:
            message (str): The message to log.
            level (int): Message level (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
        """
        if hasattr(SignalViewer, 'instance') and SignalViewer.instance:
            SignalViewer.instance.log_message(message, level)
        else:
            # Fallback if no instance exists
            level_names = {
                LogWindow.DEBUG: "DEBUG",
                LogWindow.INFO: "INFO",
                LogWindow.WARNING: "WARNING",
                LogWindow.ERROR: "ERROR"
            }
            level_name = level_names.get(level, "INFO")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level_name}: {message}")