"""
Master parser module that coordinates parsing of data files using specialized parsers.

The master parser intelligently selects parsers based on file extension, then attempts parsing:
1. Extension-specific parser (selected based on file extension)
2. Standard parser (for well-formatted CSV files)
3. Debug parser (for specific debug output formats)
4. Auto parser (fallback with more robust parsing)
"""

import os
from typing import Tuple, Dict, Any, List

import numpy as np

# Import all parsers from the parsers package
from data.parsers import parser_standard, parser_debug, parser_auto
from data.parsers import parser_hdf5, parser_lvm, parser_mat, parser_sql, parser_tmds, parser_excel
from utils.logger import Logger


class ParserMaster:
    """
    Master parser that coordinates multiple specialized parsers.
    """

    def __init__(self):
        # Initialize core parsers
        self.standard_parser = parser_standard.StandardParser()
        self.debug_parser = parser_debug.DebugParser()
        self.auto_parser = parser_auto.AutoParser()

        # Initialize specialized format parsers
        self.hdf5_parser = parser_hdf5.HDF5Parser()
        self.lvm_parser = parser_lvm.LVMParser()
        self.mat_parser = parser_mat.MATParser()
        self.sql_parser = parser_sql.SQLParser()
        self.tmds_parser = parser_tmds.TDMSParser()
        self.excel_parser = parser_excel.ExcelParser()

        # Create a mapping of file extensions to their parsers
        self.extension_to_parsers = self._build_extension_parser_map()

        # List of fallback parsers in order of attempt (used when extension-based selection fails)
        self.fallback_parsers = [
            ("StandardParser", self.standard_parser),
            ("DebugParser", self.debug_parser),
            ("AutoParser", self.auto_parser)
        ]

        # Complete list of all parsers for getting supported extensions
        self.all_parsers = [
            ("StandardParser", self.standard_parser),
            ("DebugParser", self.debug_parser),
            ("AutoParser", self.auto_parser),
            ("HDF5Parser", self.hdf5_parser),
            ("LVMParser", self.lvm_parser),
            ("MATParser", self.mat_parser),
            ("SQLParser", self.sql_parser),
            ("TMDSParser", self.tmds_parser),
            ("ExcelParser", self.excel_parser)
        ]

    def _build_extension_parser_map(self) -> Dict[str, List[Tuple[str, Any]]]:
        """
        Build a mapping of file extensions to their corresponding parsers.

        Returns:
            Dictionary mapping file extensions to lists of (parser_name, parser_instance) tuples
        """
        extension_map = {}

        # Define specialized parsers with their primary extensions
        parser_mappings = [
            (self.hdf5_parser, "HDF5Parser", ['.h5', '.hdf5']),
            (self.lvm_parser, "LVMParser", ['.lvm']),
            (self.mat_parser, "MATParser", ['.mat']),
            (self.sql_parser, "SQLParser", ['.sql', '.db', '.sqlite']),
            (self.tmds_parser, "TMDSParser", ['.tdms']),
            (self.excel_parser, "ExcelParser", ['.xlsx', '.xls', '.ods'])
        ]

        # Build the extension map
        for parser, name, extensions in parser_mappings:
            for ext in extensions:
                ext = ext.lower()
                if ext not in extension_map:
                    extension_map[ext] = []
                extension_map[ext].append((name, parser))

        # Add standard parser for its extensions
        for ext in self.standard_parser.get_supported_extensions():
            ext = ext.lower()
            if ext not in extension_map:
                extension_map[ext] = []
            extension_map[ext].append(("StandardParser", self.standard_parser))

        # Add debug parser for its extensions
        for ext in self.debug_parser.get_supported_extensions():
            ext = ext.lower()
            if ext not in extension_map:
                extension_map[ext] = []
            extension_map[ext].append(("DebugParser", self.debug_parser))

        return extension_map

    def parse_file(self, file_path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Intelligently attempt to parse a file using extension-based parser selection first,
        then fallback to other parsers if needed.

        Args:
            file_path: Path to the file to parse

        Returns:
            Tuple containing:
            - Dictionary mapping signal names to tuples of (time_array, values_array)
            - Dictionary of metadata about the file

        Raises:
            ValueError: If no parser could successfully parse the file
        """
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise FileNotFoundError(error_msg)

        # Get file extension and try the appropriate parsers first
        _, ext = os.path.splitext(file_path.lower())

        # First try parsers specific to this file extension
        if ext in self.extension_to_parsers:
            extension_parsers = self.extension_to_parsers[ext]
            Logger.log_message_static(f"Parser-Master: Found {len(extension_parsers)} specialized parser(s) for extension '{ext}'", Logger.INFO)

            for parser_name, parser in extension_parsers:
                Logger.log_message_static(f"Parser-Master: Attempting to parse file with extension-specific {parser_name}: {file_path}", Logger.INFO)
                try:
                    signals, metadata = parser.parse_file(file_path)
                    Logger.log_message_static(f"Parser-Master: Successfully parsed with {parser_name}", Logger.INFO)
                    return signals, metadata
                except Exception as e:
                    Logger.log_message_static(f"Parser-Master: Extension-specific {parser_name} failed: {str(e)}", Logger.DEBUG)
        else:
            Logger.log_message_static(f"Parser-Master: No specialized parser found for extension '{ext}'", Logger.WARNING)

        # If extension-specific parsers failed or none were found, try the fallback parsers
        Logger.log_message_static("Parser-Master: Trying fallback parsers", Logger.INFO)
        for parser_name, parser in self.fallback_parsers:
            Logger.log_message_static(f"Parser-Master: Attempting to parse file with fallback {parser_name}: {file_path}", Logger.INFO)
            try:
                signals, metadata = parser.parse_file(file_path)
                Logger.log_message_static(f"Parser-Master: Successfully parsed with fallback {parser_name}", Logger.INFO)
                return signals, metadata
            except Exception as e:
                Logger.log_message_static(f"Parser-Master: Fallback {parser_name} failed: {str(e)}",
                                        Logger.DEBUG if parser_name != "AutoParser" else Logger.ERROR)

        # If we get here, all parsers failed
        error_msg = f"Could not parse file {file_path} with any available parser"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    def get_supported_extensions(self) -> List[str]:
        """
        Get a list of all file extensions supported by any parser.

        Returns:
            List of supported file extensions
        """
        extensions = set()
        for _, parser in self.all_parsers:
            try:
                extensions.update(parser.get_supported_extensions())
            except AttributeError:
                # Skip parsers that don't implement get_supported_extensions
                pass
        return sorted(list(extensions))