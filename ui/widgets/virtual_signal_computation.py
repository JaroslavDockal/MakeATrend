"""
Virtual signal computation module.

This module handles computation of virtual signals through expression evaluation
and bit manipulation operations. It supports:
1. Computing signals from mathematical expressions
2. Extracting individual bits from signals
3. Decomposing signals into multiple bit signals
"""

import numpy as np
from utils.logger import Logger
from utils.safe_expression_validator import SafeExpressionValidator


def compute_virtual_signal(expression, time_array, data_signals, aliases=None):
    """
    Compute virtual signal from expression using data signals.

    Args:
        expression (str): Mathematical expression to evaluate
        time_array (numpy.ndarray): Array of timestamps
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples
        aliases (dict, optional): Dictionary mapping aliases to signal names

    Returns:
        tuple: (time_array, result_values) for the computed signal
    """
    Logger.log_message_static(f"Widget-VirtualSignal: Computing virtual signal: '{expression}'", Logger.DEBUG)

    # Create namespace for expression evaluation
    namespace = {}

    if not aliases:
        aliases = {}

    # Add all signals directly if no aliases provided
    for signal_name in data_signals:
        try:
            time_vals, signal_vals = data_signals[signal_name]

            # Use signal name as variable name, replacing spaces with underscores
            var_name = signal_name.replace(' ', '_')
            namespace[var_name] = signal_vals

            Logger.log_message_static(
                f"Widget-VirtualSignal: Added signal '{signal_name}' to namespace as '{var_name}', length={len(signal_vals)}",
                Logger.DEBUG)
        except Exception as e:
            Logger.log_message_static(
                f"Widget-VirtualSignal: Failed to add signal '{signal_name}' to namespace: {str(e)}",
                Logger.WARNING)

    # Add signals with specified aliases
    for alias, signal_name in aliases.items():
        if signal_name not in data_signals:
            error_msg = f"Signal '{signal_name}' not found in data"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        try:
            time_vals, signal_vals = data_signals[signal_name]
            namespace[alias] = signal_vals
            Logger.log_message_static(
                f"Widget-VirtualSignal: Added signal '{signal_name}' to namespace as '{alias}', length={len(signal_vals)}",
                Logger.DEBUG)
        except Exception as e:
            error_msg = f"Error extracting data for signal '{signal_name}': {str(e)}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

    try:
        # Use the safe expression validator to evaluate the expression
        result = SafeExpressionValidator.evaluate_expression(expression, namespace)

        # Check if result is array-like
        if result is None:
            error_msg = "Expression returned None"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        if not hasattr(result, '__len__'):
            Logger.log_message_static(f"Widget-VirtualSignal: Converting scalar {result} to array", Logger.DEBUG)
            result = np.full_like(time_array, result)
        elif not isinstance(result, np.ndarray):
            Logger.log_message_static(f"Widget-VirtualSignal: Converting {type(result)} to array", Logger.DEBUG)
            result = np.array(result)

        # Ensure result has same length as time_array
        if len(result) != len(time_array):
            error_msg = f"Result length ({len(result)}) doesn't match time array length ({len(time_array)})"
            Logger.log_message_static(error_msg, Logger.ERROR)
            raise ValueError(error_msg)

        Logger.log_message_static("Widget-VirtualSignal: Virtual signal computation successful", Logger.DEBUG)
        return time_array, result

    except Exception as e:
        import traceback
        error_msg = f"Failed to compute virtual signal: {str(e)}"
        Logger.log_message_static(error_msg, Logger.ERROR)
        Logger.log_message_static(f"Widget-VirtualSignal: Traceback: {traceback.format_exc()}", Logger.DEBUG)
        raise ValueError(error_msg)


def compute_single_bit_extraction(alias_mapping, data_signals):
    """
    Extracts a single bit from a signal based on bit mode configuration.

    Args:
        alias_mapping (dict): Contains "source_signal", "bit_index" and "_bit_mode" keys
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples

    Returns:
        tuple: (time_array, bit_values) for the extracted bit
    """
    Logger.log_message_static("Widget-VirtualSignal: Computing single bit extraction", Logger.DEBUG)

    source_signal = alias_mapping.get("source_signal")
    bit_index = alias_mapping.get("bit_index")

    if source_signal is None or bit_index is None:
        error_msg = "Missing source_signal or bit_index in bit mode configuration"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Get the signal data
    if source_signal not in data_signals:
        error_msg = f"Signal '{source_signal}' not found in data"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    time_array, values = data_signals[source_signal]

    # Check if the signal contains integer values for bit operations
    if not np.all(np.equal(np.mod(values, 1), 0)):
        error_msg = f"Signal '{source_signal}' contains non-integer values and cannot be used for bit extraction"
        Logger.log_message_static(error_msg, Logger.WARNING)
        raise ValueError(error_msg)

    # Convert to integers to ensure proper bit operations
    values_int = values.astype(np.int64)

    # Extract the specified bit
    bit_mask = 1 << bit_index
    bit_values = ((values_int & bit_mask) > 0).astype(bool)  # Using boolean values instead of 0/1

    Logger.log_message_static(f"Widget-VirtualSignal: Extracted bit {bit_index} from '{source_signal}'", Logger.DEBUG)
    return time_array, bit_values


def compute_bit_decomposition(alias_mapping, data_signals):
    """
    Decomposes a signal into its individual bits.

    Args:
        alias_mapping (dict): Contains "signal" (the signal to decompose) and "bits" (list of bit configurations)
        data_signals (dict): Dictionary of signal data as (time_array, values_array) tuples

    Returns:
        list: List of tuples (bit_alias, time_array, bit_values) for each selected bit
    """
    Logger.log_message_static("Widget-VirtualSignal: Computing bit decomposition", Logger.DEBUG)

    signal_name = alias_mapping.get("signal")
    if not signal_name:
        signal_name = alias_mapping.get("bit_signal_combo", "")  # Fallback for UI mapping

    selected_bits = [item for item in alias_mapping.get("bits", []) if item[1]]  # Filter only checked bits

    if not signal_name:
        error_msg = "No signal specified for bit decomposition"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    if not selected_bits:
        error_msg = "No bits selected for decomposition"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    # Get the signal data
    if signal_name not in data_signals:
        error_msg = f"Signal '{signal_name}' not found in data"
        Logger.log_message_static(error_msg, Logger.ERROR)
        raise ValueError(error_msg)

    time_array, values = data_signals[signal_name]

    # Check if the signal is integer (no decimal part)
    if not np.all(np.equal(np.mod(values, 1), 0)):
        error_msg = f"Signal '{signal_name}' contains non-integer values and cannot be decomposed into bits"
        Logger.log_message_static(error_msg, Logger.WARNING)
        raise ValueError(error_msg)

    # Convert to integers to ensure proper bit operations
    values_int = values.astype(np.int64)

    # Extract each selected bit and create a list of results
    bit_signals = []
    for bit_index, _, bit_alias in selected_bits:
        # Create bit mask and extract the bit
        bit_mask = 1 << bit_index
        # Use boolean values (TRUE/FALSE) instead of 0/1
        bit_values = ((values_int & bit_mask) > 0).astype(bool)

        # Store the result as a tuple (alias, time_array, values)
        bit_signals.append((bit_alias, time_array, bit_values))
        Logger.log_message_static(f"Widget-VirtualSignal: Extracted bit {bit_index} as '{bit_alias}'", Logger.DEBUG)

    return bit_signals