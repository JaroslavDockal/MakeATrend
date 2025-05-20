"""
Project management module for signal data analysis application.

This module provides functionality for saving and loading project state information
to and from JSON files. The project state typically contains:
- Loaded signal data and references
- View configuration (visible signals, colors, scaling)
- Marker positions and annotations
- Analysis configuration and results

The saved project files enable users to restore their analysis session exactly
as they left it, ensuring continuity in complex signal analysis workflows.

Functions:
    save_project_state: Serialize and save project state to a JSON file
    load_project_state: Load and deserialize project state from a JSON file
"""

import os
import json

from utils.logger import Logger

def save_project_state(file_path, state):
    """
    Saves the project state to a JSON file.

    Args:
        file_path (str): Path to save the project state.
        state (dict): The project state to save.
    """
    Logger.log_message_static(f"Saving project state to {os.path.basename(file_path)}", Logger.INFO)

    try:
        Logger.log_message_static(f"Project state contains {len(state)} entries", Logger.DEBUG)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        Logger.log_message_static("Project state saved successfully", Logger.INFO)
    except Exception as e:
        Logger.log_message_static(f"Failed to save project state: {str(e)}", Logger.ERROR)
        raise IOError(f"Failed to save project state: {e}")

def load_project_state(file_path):
    """
    Loads the project state from a JSON file.

    Args:
        file_path (str): Path to the project state file.

    Returns:
        dict: The loaded project state.
    """
    Logger.log_message_static(f"Loading project state from {os.path.basename(file_path)}", Logger.INFO)

    if not os.path.exists(file_path):
        Logger.log_message_static(f"Project file not found: {file_path}", Logger.ERROR)
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        Logger.log_message_static(f"Project state loaded successfully with {len(state)} entries", Logger.INFO)
        return state
    except json.JSONDecodeError as e:
        Logger.log_message_static(f"Invalid JSON in project file: {str(e)}", Logger.ERROR)
        raise IOError(f"Failed to load project state: Invalid JSON format - {e}")
    except Exception as e:
        Logger.log_message_static(f"Failed to load project state: {str(e)}", Logger.ERROR)
        raise IOError(f"Failed to load project state: {e}")