"""
Module with predefined colors for signals.
Provides a set of optimized colors for distinguishing signals on a dark background.
"""

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
        return cls.COLORS[index % len(cls.COLORS)]

    @classmethod
    def get_color_for_name(cls, name: str) -> str:
        """
        Generate a consistent color for a given signal name.

        A hash of the signal name is used to determine the color index, ensuring
        that the same name always maps to the same color.

        Args:
            name (str): Name of the signal.

        Returns:
            str: Hexadecimal color code (e.g., '#ff0000').
        """
        hash_value = hash(name)
        index = abs(hash_value) % len(cls.COLORS)
        return cls.COLORS[index]
