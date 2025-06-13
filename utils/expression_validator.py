"""
Safe expression validator for evaluating user-provided expressions.

This module provides utilities to safely evaluate mathematical expressions
without allowing arbitrary code execution. It includes functionality for:
1. Safely evaluating expressions within a restricted environment
2. Providing a set of allowed mathematical and NumPy functions
3. Handling errors and exceptions in expression evaluation
4. Supporting array operations, trigonometric, statistical, and signal processing functions
"""

import numpy as np
import re
from utils.logger import Logger


class SafeExpressionValidator:
    """
    Provides safe evaluation of mathematical expressions.

    This class allows evaluating user-provided expressions in a restricted
    environment with only approved mathematical operations. It supports a wide
    range of NumPy functions for signal processing and data analysis.
    """

    @staticmethod
    def get_safe_functions():
        """
        Returns a dictionary of allowed functions for safe expression evaluation.

        Returns:
            dict: Dictionary of safe function names to function objects
        """
        # Basic constants
        constants = {
            'pi': np.pi,
            'e': np.e,
            'nan': np.nan,
            'inf': np.inf
        }

        # Basic math functions
        basic_math = {
            'abs': np.abs,
            'sqrt': np.sqrt,
            'pow': np.power,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'log2': np.log2,
            'floor': np.floor,
            'ceil': np.ceil,
            'round': np.round,
            'clip': np.clip,
            'sign': np.sign
        }

        # Trigonometric functions
        trig = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'arctan2': np.arctan2,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
            'deg2rad': np.deg2rad,
            'rad2deg': np.rad2deg
        }

        # Statistical functions
        stats = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'var': np.var,
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
            'percentile': np.percentile,
            'histogram': np.histogram
        }

        # Signal processing functions
        signal_proc = {
            'diff': np.diff,
            'gradient': np.gradient,
            'cumsum': np.cumsum,
            'convolve': np.convolve,
            'correlate': np.correlate,
            'unwrap': np.unwrap
        }

        # Array operations
        array_ops = {
            'array': np.array,
            'zeros': np.zeros,
            'ones': np.ones,
            'linspace': np.linspace,
            'arange': np.arange,
            'concatenate': np.concatenate,
            'reshape': np.reshape,
            'transpose': np.transpose
        }

        # Combine all categories
        safe_funcs = {}
        safe_funcs.update(constants)
        safe_funcs.update(basic_math)
        safe_funcs.update(trig)
        safe_funcs.update(stats)
        safe_funcs.update(signal_proc)
        safe_funcs.update(array_ops)

        return safe_funcs

    @staticmethod
    def is_expression_safe(expression):
        """
        Performs basic validation to check if an expression appears safe.

        This method checks for common patterns that might indicate attempts
        to execute arbitrary code.

        Args:
            expression (str): The expression to validate

        Returns:
            bool: True if expression appears safe, False otherwise
        """
        if expression is None or not isinstance(expression, str):
            return False

        # Trim whitespace
        expression = expression.strip()

        # Empty expression
        if not expression:
            return False

        # Check for dangerous patterns
        dangerous_patterns = [
            r'__[a-zA-Z]+__',            # Dunder methods
            r'import\s+',                 # Import statements
            r'exec\s*\(',                 # exec() function
            r'eval\s*\(',                 # eval() function
            r'compile\s*\(',              # compile() function
            r'globals\s*\(',              # globals() function
            r'locals\s*\(',               # locals() function
            r'getattr\s*\(',              # getattr() function
            r'setattr\s*\(',              # setattr() function
            r'delattr\s*\(',              # delattr() function
            r'open\s*\(',                 # file operations
            r'file\s*\(',                 # file operations
            r'os\.',                      # os module
            r'sys\.',                     # sys module
            r'subprocess\.',              # subprocess module
            r'lambda\s+',                 # lambda functions
            r'for\s+.*\s+in',             # for loops
            r'while\s+',                  # while loops
            r'def\s+',                    # function definitions
            r'class\s+'                   # class definitions
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression):
                Logger.log_message_static(f"SafeValidator: Unsafe pattern detected in expression: '{pattern}'", Logger.WARNING)
                return False

        return True

    @staticmethod
    def evaluate_expression(expression, namespace):
        """
        Safely evaluates an expression in a restricted environment.

        Args:
            expression (str): The expression to evaluate
            namespace (dict): Dictionary of variables available to the expression

        Returns:
            object: Result of the expression evaluation

        Raises:
            ValueError: If the expression evaluation fails or is deemed unsafe
        """
        try:
            # First check if the expression is safe
            if not SafeExpressionValidator.is_expression_safe(expression):
                error_msg = f"Expression validation failed: potentially unsafe expression"
                Logger.log_message_static(error_msg, Logger.ERROR)
                raise ValueError(error_msg)

            # Add safe functions to namespace
            safe_functions = SafeExpressionValidator.get_safe_functions()
            namespace.update(safe_functions)

            Logger.log_message_static(
                f"SafeValidator: Added {len(safe_functions)} safe functions to namespace",
                Logger.DEBUG)

            # Safely evaluate the expression without access to built-ins
            Logger.log_message_static(f"SafeValidator: Evaluating expression: '{expression}'", Logger.DEBUG)
            result = eval(expression, {"__builtins__": {}}, namespace)

            # Log result type for debugging
            Logger.log_message_static(f"SafeValidator: Expression result type: {type(result)}", Logger.DEBUG)

            return result

        except Exception as e:
            import traceback
            error_msg = f"Failed to evaluate expression: {str(e)}"
            Logger.log_message_static(error_msg, Logger.ERROR)
            Logger.log_message_static(f"SafeValidator: Traceback: {traceback.format_exc()}", Logger.DEBUG)
            raise ValueError(error_msg)


def validate_expression(expression, allowed_names=None):
    """
    Validates if an expression is safe and syntactically correct.

    Args:
        expression (str): The expression to validate
        allowed_names (list): List of allowed variable names/aliases

    Returns:
        tuple: (is_valid, error_message)
    """
    if not SafeExpressionValidator.is_expression_safe(expression):
        return False, "Expression contains unsafe elements"

    # Additional validation as needed
    # This can leverage the existing SafeExpressionValidator functionality

    return True, ""


def validate_signal_name(name, existing_names=None):
    """
    Validates a signal name for correctness and uniqueness.

    Args:
        name (str): The signal name to validate
        existing_names (list): List of existing signal names to check against

    Returns:
        tuple: (is_valid, error_message)
    """
    if not name or not name.strip():
        return False, "Signal name cannot be empty"

    # Check for allowed characters
    pattern = r'^[A-Za-z0-9_\-\.\[\] ]+$'
    if not re.match(pattern, name):
        return False, "Signal name can only contain letters, numbers, spaces, and _ - . [ ]"

    # Check for duplicates
    if existing_names and name in existing_names:
        return False, f"Signal name '{name}' already exists"

    return True, ""