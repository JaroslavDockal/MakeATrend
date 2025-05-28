"""
Debug/Recorder parser implementation.

This parser handles the Drive Debug recorder format exactly like the original parser_old.py.
It processes files with "RECORDER VALUES" or "TREND VALUES" headers and proprietary format.
"""

import os
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

from utils.logger import Logger


class DebugParser:
    """
    Parser for Drive Debug recorder format files.
    Replicates the exact behavior of parse_recorder_format from parser_old.py.
    """

    def __init__(self):
        """Initialize the debug parser."""
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions
        """
        return ['.txt', '.rec', '.log', '.dat']

    def is_recorder_format(self, file_path: str) -> bool:
        """
        Check if file is in Drive Debug recorder format.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file appears to be in recorder format
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(1000)  # Read first 1000 chars
                return ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content
        except:
            try:
                with open(file_path, "r", encoding="latin1") as f:
                    content = f.read(1000)
                    return ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content
            except:
                return False

    def parse_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Parse a Drive Debug recorder format file.

        This method replicates the exact behavior of parse_recorder_format from parser_old.py.

        Args:
            file_path: Path to the file to parse

        Returns:
            Tuple containing:
            - np.ndarray: Array of timestamps (float, seconds since UNIX epoch)
            - Dict[str, np.ndarray]: Dictionary of signal name -> signal values as float32 arrays

        Raises:
            ValueError: If the file cannot be parsed as recorder format
        """
        Logger.log_message_static(f"Parser-Debug: Starting to parse file: {os.path.basename(file_path)}", Logger.DEBUG)

        # Read file content with proper encoding handling
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                Logger.log_message_static(f"Parser-Debug: Successfully read file: {os.path.basename(file_path)}", Logger.DEBUG)
        except UnicodeDecodeError:
            Logger.log_message_static(f"Parser-Debug: UTF-8 encoding failed, trying with latin1 for file: {os.path.basename(file_path)}", Logger.WARNING)
            try:
                with open(file_path, "r", encoding="latin1") as f:
                    content = f.read()
            except Exception as e:
                Logger.log_message_static(f"Parser-Debug: Failed to read file {os.path.basename(file_path)}: {str(e)}", Logger.ERROR)
                raise ValueError(f"Cannot read file: {str(e)}")
        except Exception as e:
            Logger.log_message_static(f"Parser-Debug: Failed to read file {os.path.basename(file_path)}: {str(e)}", Logger.ERROR)
            raise ValueError(f"Cannot read file: {str(e)}")

        # Check if this is actually a recorder format
        if not (("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content):
            Logger.log_message_static(f"Parser-Debug: File does not appear to be Drive Debug format: {os.path.basename(file_path)}", Logger.ERROR)
            raise ValueError("File is not in Drive Debug recorder format")

        Logger.log_message_static("Parser-Debug: Parsing Drive Debug format file", Logger.DEBUG)

        lines = content.strip().splitlines()
        item_map = {}
        start_time_str = None
        interval_sec = None
        data_lines = []

        Logger.log_message_static(f"Parser-Debug: File contains {len(lines)} lines", Logger.DEBUG)

        for line in lines:
            line = line.strip()
            if line.startswith("Item "):
                match = re.match(r"Item\s+(\d+)\s*=\s*(.+)", line)
                if match:
                    item_id = int(match.group(1))
                    item_name = match.group(2).strip()
                    item_map[item_id] = item_name
                    Logger.log_message_static(f"Parser-Debug: Found Item {item_id} = {item_name}", Logger.DEBUG)
            elif "Time of Interval" in line:
                start_time_str = line.split(":", 1)[1].strip()
                Logger.log_message_static(f"Parser-Debug: Found start time: {start_time_str}", Logger.DEBUG)
            elif "Interval:" in line:
                m = re.search(r"([\d.]+)\s*sec", line)
                if m:
                    interval_sec = float(m.group(1))
                    Logger.log_message_static(f"Parser-Debug: Found interval: {interval_sec} seconds", Logger.DEBUG)
            elif re.match(r"\s*\d+\s+", line):
                parts = re.split(r'\s+', line)
                try:
                    idx = int(parts[0])
                    values = [float(p.replace(',', '.')) for p in parts[1:]]
                    data_lines.append([idx] + values)
                except ValueError:
                    Logger.log_message_static(f"Parser-Debug: Skipping invalid data line: {line}", Logger.WARNING)
                    continue

        if not start_time_str:
            Logger.log_message_static("Parser-Debug: Missing start time in the file", Logger.ERROR)
            raise ValueError("Invalid recorder format: missing start time.")
        if not interval_sec:
            Logger.log_message_static("Parser-Debug: Missing interval in the file", Logger.ERROR)
            raise ValueError("Invalid recorder format: missing interval.")
        if not item_map:
            Logger.log_message_static("Parser-Debug: No signal items found in the file", Logger.ERROR)
            raise ValueError("Invalid recorder format: no signal items found.")
        if not data_lines:
            Logger.log_message_static("Parser-Debug: No data lines found in the file", Logger.ERROR)
            raise ValueError("Invalid recorder format: no data found.")

        try:
            start_dt = datetime.strptime(start_time_str, "%m/%d/%y %H:%M:%S")
            Logger.log_message_static(f"Parser-Debug: Parsed start time: {start_dt}", Logger.DEBUG)
        except ValueError as e:
            Logger.log_message_static(f"Parser-Debug: Failed to parse start time '{start_time_str}': {str(e)}", Logger.ERROR)
            raise ValueError(f"Invalid start time format: {start_time_str}")

        Logger.log_message_static(f"Parser-Debug: Sorting {len(data_lines)} data lines", Logger.DEBUG)
        data_lines.sort(key=lambda row: row[0])

        Logger.log_message_static("Parser-Debug: Calculating timestamps", Logger.DEBUG)
        timestamps = [(start_dt - timedelta(seconds=row[0] * interval_sec)).timestamp() for row in data_lines]

        signals = {}
        Logger.log_message_static("Parser-Debug: Extracting signal values", Logger.DEBUG)
        for i in range(len(data_lines[0]) - 1):
            name = item_map.get(i + 1, f"Signal {i + 1}")
            signals[name] = [row[i + 1] for row in data_lines]

        for name in signals:
            signals[name] = np.array(signals[name], dtype=np.float32)
            Logger.log_message_static(f"Parser-Debug: Created signal '{name}' with {len(signals[name])} points", Logger.DEBUG)

        Logger.log_message_static(f"Parser-Debug: Successfully parsed {len(signals)} signals from Drive Debug file", Logger.INFO)
        return np.array(timestamps, dtype=np.float64), signals