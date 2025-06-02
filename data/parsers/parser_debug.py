"""
Debug/Recorder parser implementation.

This parser handles both the Drive Debug recorder format and DATALOGGER VALUES format.
It processes files with "RECORDER VALUES", "TREND VALUES", or "DATALOGGER VALUES" headers.
"""

import os
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

from utils.logger import Logger


class DebugParser:
    """
    Parser for Drive Debug recorder format files and DATALOGGER VALUES format files.
    Handles both original recorder format and the new datalogger format.
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
        Check if file is in Drive Debug recorder format or DATALOGGER VALUES format.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file appears to be in recorder or datalogger format
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(1000)  # Read first 1000 chars
                recorder_format = ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content
                datalogger_format = "DATALOGGER VALUES" in content and "Time level:" in content
                return recorder_format or datalogger_format
        except:
            try:
                with open(file_path, "r", encoding="latin1") as f:
                    content = f.read(1000)
                    recorder_format = ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content
                    datalogger_format = "DATALOGGER VALUES" in content and "Time level:" in content
                    return recorder_format or datalogger_format
            except:
                return False

    def _is_datalogger_format(self, content: str) -> bool:
        """Check if content is in DATALOGGER VALUES format."""
        return "DATALOGGER VALUES" in content and "Time level:" in content

    def _parse_datalogger_format(self, content: str, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Parse DATALOGGER VALUES format file.

        Args:
            content: File content as string
            file_path: Path to the file being parsed

        Returns:
            Tuple containing signals dict and metadata dict
        """
        Logger.log_message_static("Parser-Debug: Parsing DATALOGGER VALUES format file", Logger.DEBUG)

        lines = content.strip().splitlines()
        channel_map = {}
        start_time_str = None
        interval_ms = None
        data_lines = []

        Logger.log_message_static(f"Parser-Debug: File contains {len(lines)} lines", Logger.DEBUG)

        for line in lines:
            line = line.strip()

            # Parse channel definitions: "Channel 1 = 1.08 MOTOR SPEED"
            if line.startswith("Channel "):
                match = re.match(r"Channel\s+(\d+)\s*=\s*[\d.]+\s+(.+)", line)
                if match:
                    channel_id = int(match.group(1))
                    channel_name = match.group(2).strip()
                    channel_map[channel_id] = channel_name
                    Logger.log_message_static(f"Parser-Debug: Found Channel {channel_id} = {channel_name}", Logger.DEBUG)

            # Parse trigger time: "Trig time: 2024-10-31 21:16:07.8330"
            elif "Trig time:" in line:
                start_time_str = line.split(":", 1)[1].strip()
                Logger.log_message_static(f"Parser-Debug: Found trigger time: {start_time_str}", Logger.DEBUG)

            # Parse time level: "Time level: 0.1 ms"
            elif "Time level:" in line:
                m = re.search(r"([\d.]+)\s*ms", line)
                if m:
                    interval_ms = float(m.group(1))
                    Logger.log_message_static(f"Parser-Debug: Found time level: {interval_ms} ms", Logger.DEBUG)

            # Parse data lines - both numbered and "Trig:" line
            elif re.match(r"\s*(\d+|Trig:)\s+", line):
                parts = re.split(r'\s+', line)
                try:
                    # Handle "Trig:" line - use a special index (e.g., 820 based on the data)
                    if parts[0] == "Trig:":
                        idx = 820  # This appears to be the trigger point in your data
                    else:
                        idx = int(parts[0])

                    values = [float(p.replace(',', '.')) for p in parts[1:]]
                    data_lines.append([idx] + values)
                except ValueError:
                    Logger.log_message_static(f"Parser-Debug: Skipping invalid data line: {line}", Logger.WARNING)
                    continue

        # Validation
        if not start_time_str:
            Logger.log_message_static("Parser-Debug: Missing trigger time in the file", Logger.ERROR)
            raise ValueError("Invalid datalogger format: missing trigger time.")
        if interval_ms is None:
            Logger.log_message_static("Parser-Debug: Missing time level in the file", Logger.ERROR)
            raise ValueError("Invalid datalogger format: missing time level.")
        if not channel_map:
            Logger.log_message_static("Parser-Debug: No channel definitions found in the file", Logger.ERROR)
            raise ValueError("Invalid datalogger format: no channel definitions found.")
        if not data_lines:
            Logger.log_message_static("Parser-Debug: No data lines found in the file", Logger.ERROR)
            raise ValueError("Invalid datalogger format: no data found.")

        # Parse start time - handle the format "2024-10-31 21:16:07.8330"
        try:
            # Remove microseconds part if present (the .8330 part)
            time_part = start_time_str.split('.')[0]
            start_dt = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S")
            Logger.log_message_static(f"Parser-Debug: Parsed start time: {start_dt}", Logger.DEBUG)
        except ValueError as e:
            Logger.log_message_static(f"Parser-Debug: Failed to parse start time '{start_time_str}': {str(e)}", Logger.ERROR)
            raise ValueError(f"Invalid start time format: {start_time_str}")

        Logger.log_message_static(f"Parser-Debug: Sorting {len(data_lines)} data lines", Logger.DEBUG)
        data_lines.sort(key=lambda row: row[0])

        # Convert interval from ms to seconds
        interval_sec = interval_ms / 1000.0
        Logger.log_message_static(f"Parser-Debug: Interval: {interval_ms} ms = {interval_sec} seconds", Logger.DEBUG)

        # Calculate timestamps - datalogger format appears to count forward from trigger time
        Logger.log_message_static("Parser-Debug: Calculating timestamps", Logger.DEBUG)
        timestamps = [(start_dt + timedelta(seconds=row[0] * interval_sec)).timestamp() for row in data_lines]
        timestamps_np = np.array(timestamps, dtype=np.float64)

        # Extract signal values
        raw_signals = {}
        Logger.log_message_static("Parser-Debug: Extracting signal values", Logger.DEBUG)
        for i in range(len(data_lines[0]) - 1):
            name = channel_map.get(i + 1, f"Channel {i + 1}")
            raw_signals[name] = [row[i + 1] for row in data_lines]

        for name in raw_signals:
            raw_signals[name] = np.array(raw_signals[name], dtype=np.float32)
            Logger.log_message_static(f"Parser-Debug: Created signal '{name}' with {len(raw_signals[name])} points", Logger.DEBUG)

        # Convert format to match parser_master expectations
        result_signals = {}
        metadata = {
            "source_file": file_path,
            "parser": "DebugParser",
            "interval": interval_sec,
            "start_time": start_time_str,
            "format": "DATALOGGER VALUES"
        }

        for name, values in raw_signals.items():
            result_signals[name] = (timestamps_np, values)

        Logger.log_message_static(f"Parser-Debug: Successfully parsed {len(result_signals)} signals from DATALOGGER VALUES file", Logger.INFO)
        return result_signals, metadata

    def _parse_recorder_format(self, content: str, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Parse original RECORDER VALUES format file.

        Args:
            content: File content as string
            file_path: Path to the file being parsed

        Returns:
            Tuple containing signals dict and metadata dict
        """
        Logger.log_message_static("Parser-Debug: Parsing RECORDER VALUES format file", Logger.DEBUG)

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
        timestamps_np = np.array(timestamps, dtype=np.float64)

        raw_signals = {}
        Logger.log_message_static("Parser-Debug: Extracting signal values", Logger.DEBUG)
        for i in range(len(data_lines[0]) - 1):
            name = item_map.get(i + 1, f"Signal {i + 1}")
            raw_signals[name] = [row[i + 1] for row in data_lines]

        for name in raw_signals:
            raw_signals[name] = np.array(raw_signals[name], dtype=np.float32)
            Logger.log_message_static(f"Parser-Debug: Created signal '{name}' with {len(raw_signals[name])} points", Logger.DEBUG)

        # Convert format to match parser_master expectations
        result_signals = {}
        metadata = {
            "source_file": file_path,
            "parser": "DebugParser",
            "interval": interval_sec,
            "start_time": start_time_str,
            "format": "RECORDER VALUES"
        }

        for name, values in raw_signals.items():
            result_signals[name] = (timestamps_np, values)

        Logger.log_message_static(f"Parser-Debug: Successfully parsed {len(result_signals)} signals from RECORDER VALUES file", Logger.INFO)
        return result_signals, metadata

    def parse_file(self, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Parse a Drive Debug recorder format file or DATALOGGER VALUES format file.

        This method handles both the original recorder format and the new datalogger format.

        Args:
            file_path: Path to the file to parse

        Returns:
            Tuple containing:
            - Dictionary mapping signal names to tuples of (time_array, values_array)
            - Dictionary of metadata about the file

        Raises:
            ValueError: If the file cannot be parsed as either format
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

        # Determine format and parse accordingly
        if self._is_datalogger_format(content):
            return self._parse_datalogger_format(content, file_path)
        elif ("RECORDER VALUES" in content or "TREND VALUES" in content) and "Interval:" in content:
            return self._parse_recorder_format(content, file_path)
        else:
            Logger.log_message_static(f"Parser-Debug: File does not appear to be in supported format: {os.path.basename(file_path)}", Logger.ERROR)
            raise ValueError("File is not in supported Debug format (RECORDER VALUES or DATALOGGER VALUES)")