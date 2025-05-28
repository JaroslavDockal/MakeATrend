"""
Safe expression validator for evaluating user-provided expressions.

This module provides utilities to safely evaluate mathematical expressions
without allowing arbitrary code execution. It includes functionality for:
1. Safely evaluating expressions within a restricted environment
2. Providing a set of allowed mathematical and NumPy functions
3. Handling errors and exceptions in expression evaluation
"""

import numpy as np
from utils.logger import Logger


class SafeExpressionValidator:
    """
    Provides safe evaluation of mathematical expressions.

    This class allows evaluating user-provided expressions in a restricted
    environment with only approved mathematical operations.
    """

    @staticmethod
    def get_safe_functions():
        """
        Returns a dictionary of allowed functions for safe expression evaluation.

        Returns:
            dict: Dictionary of safe function names to function objects
        """
        return {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log,
            'exp': np.exp, 'pi': np.pi, 'e': np.e
        }

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
            ValueError: If the expression evaluation fails
        """
        try:
            # Add safe functions to namespace
            safe_functions = SafeExpressionValidator.get_safe_functions()
            namespace.update(safe_functions)

            Logger.log_message_static(
                f"SafeValidator: Added safe NumPy functions to namespace: {list(safe_functions.keys())}",
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