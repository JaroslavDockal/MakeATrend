"""
Module with predefined colors for signals.
Provides a set of optimized colors for distinguishing signals on a dark background.
"""
import hashlib
import random

from .logger import Logger


class SignalColors:
    """
    A class providing a predefined set of colors for signal visualization.

    The colors are optimized for readability on dark backgrounds. They offer high
    contrast and are visually distinct from one another to ensure clear differentiation
    between signals.
    """

    # Highly distinct color palette optimized for dark backgrounds and visual separation
    # Hexadecimal format used in PyQtGraph
    COLORS = [
        '#FF3300',  # Bright Red
        '#33FF00',  # Bright Green
        '#0066FF',  # Strong Blue
        '#FFCC00',  # Gold
        '#CC00FF',  # Violet
        '#00FFCC',  # Turquoise
        '#FF6600',  # Orange
        '#00CCFF',  # Sky Blue
        '#FF0099',  # Hot Pink
        '#66FF33',  # Lime
        '#9900FF',  # Purple
        '#FFFF00',  # Yellow
        '#FF99CC',  # Light Pink
        '#00FF66',  # Mint Green
        '#3399FF',  # Azure
        '#FF9900',  # Amber
        '#00FFFF',  # Cyan
        '#FF66FF',  # Magenta
        '#CCFF00',  # Chartreuse
        '#FFFFFF',  # White
        '#800080',  # Deep Purple
        '#FF6347',  # Tomato Red
        '#8A2BE2',  # Blue Violet
        '#ADFF2F',  # Green Yellow
        '#FF4500',  # Orange Red
        '#2E8B57',  # Sea Green
        '#FFD700',  # Gold
        '#D2691E',  # Chocolate
        '#8B0000',  # Dark Red
    ]

    @classmethod
    def get_color(cls, index: int) -> str:
        """
        Return a color based on the given index.

        Colors are reused cyclically if the index exceeds the number of defined colors.

        Args:
            index (int): The index of the desired color.

        Returns:
            str: Hexadecimal color code (e.g., '#ff0000').
        """
        Logger.log_message_static(f"Requesting color at index {index}", Logger.DEBUG)

        if not isinstance(index, int):
            Logger.log_message_static(f"Non-integer index {index} provided to get_color, attempting conversion", Logger.WARNING)
            try:
                index = int(index)
            except (ValueError, TypeError) as e:
                Logger.log_message_static(f"Could not convert index to integer: {str(e)}", Logger.ERROR)
                index = 0

        if index < 0:
            Logger.log_message_static(f"Negative index {index} provided, using modulo to normalize", Logger.WARNING)

        color_count = len(cls.COLORS)
        normalized_index = index % color_count

        if index >= color_count:
            Logger.log_message_static(f"Color index {index} exceeds available colors ({color_count}), wrapping to index {normalized_index}", Logger.DEBUG)

        color = cls.COLORS[normalized_index]
        Logger.log_message_static(f"Returning color {color} for index {index}", Logger.DEBUG)
        return color

    @classmethod
    def get_color_for_name(cls, name: str) -> str:
        """
        Generate a distinct color for a given signal name with maximum variation.

        Creates highly saturated, bright colors that stand out clearly on dark backgrounds.

        Uses the signal name as a base but introduces multiple randomization
        factors to ensure even similar names get different colors.

        Args:
            name (str): Name of the signal.

        Returns:
            str: Hexadecimal color code (e.g., '#ff0000').
        """
        Logger.log_message_static(f"Generating color for signal '{name}'", Logger.DEBUG)

        if not name:
            Logger.log_message_static("Empty signal name provided, using fallback color", Logger.WARNING)
            return cls.COLORS[0]

        try:
            # Use the name to create a hash
            name_length = len(name)
            Logger.log_message_static(f"Signal name length: {name_length}", Logger.DEBUG)

            prefix = name[:min(3, len(name))]
            suffix = name[-min(3, len(name)):]
            Logger.log_message_static(f"Signal name prefix: '{prefix}', suffix: '{suffix}'", Logger.DEBUG)

            # Create a unique seed from name characteristics
            char_sum = sum(ord(c) for c in name)
            name_hash = hash(name)
            prefix_suffix_hash = hash(prefix + suffix)
            unique_chars = len(set(name))
            length_component = name_length * 31

            Logger.log_message_static(f"Name hash components calculated", Logger.DEBUG)

            # Mix the components with xor operations for better distribution
            mixed_hash = char_sum ^ name_hash ^ prefix_suffix_hash ^ (unique_chars << 8) ^ (length_component << 16)
            Logger.log_message_static(f"Mixed hash value: {mixed_hash}", Logger.DEBUG)

            # Create more vibrant colors by using extreme values in HSV space
            # Convert hash to HSV (Hue, Saturation, Value) color space parameters
            h = abs(mixed_hash) % 360  # Hue (0-359)
            s = 90 + (abs(mixed_hash >> 8) % 10)  # Saturation (90-99%)
            v = 90 + (abs(mixed_hash >> 16) % 10)  # Value/Brightness (90-99%)

            Logger.log_message_static(f"HSV components - H: {h}, S: {s}, V: {v}", Logger.DEBUG)

            # Convert HSV to RGB
            # Based on HSV to RGB conversion algorithm
            c = (v / 100) * (s / 100)
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = (v / 100) - c

            if 0 <= h < 60:
                r, g, b = c, x, 0
            elif 60 <= h < 120:
                r, g, b = x, c, 0
            elif 120 <= h < 180:
                r, g, b = 0, c, x
            elif 180 <= h < 240:
                r, g, b = 0, x, c
            elif 240 <= h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            # Convert to 8-bit RGB values
            r = int(255 * (r + m))
            g = int(255 * (g + m))
            b = int(255 * (b + m))

            Logger.log_message_static(f"Final RGB components - R: {r}, G: {g}, B: {b}", Logger.DEBUG)

            # Generate the color
            color = f"#{r:02x}{g:02x}{b:02x}"
            Logger.log_message_static(f"Color generated for '{name}': {color}", Logger.DEBUG)

            return color

        except Exception as e:
            Logger.log_message_static(f"Error generating color for signal '{name}': {str(e)}", Logger.ERROR)
            import traceback
            Logger.log_message_static(f"Color generation traceback: {traceback.format_exc()}", Logger.DEBUG)
            # Fallback to a default color in case of error
            return cls.COLORS[0]

    @classmethod
    def random_color(cls) -> str:
        """
        Generate a random color for a signal, which is useful if the predefined colors are exhausted.

        Returns:
            str: Random Hexadecimal color code (e.g., '#ff0000').
        """
        Logger.log_message_static("Generating random color", Logger.DEBUG)

        try:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            Logger.log_message_static(f"Generated random RGB values: R={r}, G={g}, B={b}", Logger.DEBUG)

            # Check if the color is too dark for visibility on dark backgrounds
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            if brightness < 50:
                Logger.log_message_static(f"Random color too dark (brightness={brightness}), increasing luminance", Logger.WARNING)
                # Make it brighter by increasing all components
                r = min(255, r + 100)
                g = min(255, g + 100)
                b = min(255, b + 100)
                Logger.log_message_static(f"Adjusted RGB values: R={r}, G={g}, B={b}", Logger.DEBUG)

            random_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            Logger.log_message_static(f"Generated random color: {random_color}", Logger.INFO)
            return random_color

        except Exception as e:
            Logger.log_message_static(f"Error generating random color: {str(e)}", Logger.ERROR)
            # Fallback to a bright color that should be visible
            return "#FF0000"  # Bright red as fallback