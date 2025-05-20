"""
Module with predefined colors for signals.
Provides a set of optimized colors for distinguishing signals on a dark background.
"""
from distinctipy import get_colors
import numpy as np

from utils.logger import Logger


class SignalColors:
    """
    A class providing a predefined set of colors for signal visualization.

    The colors are optimized for readability on dark backgrounds. They offer high
    contrast and are visually distinct from one another to ensure clear differentiation
    between signals.
    """
    _color_map = {}
    _used_colors = []
    _initialized = False
    _fallback_color = "#FF3300"

    @classmethod
    def initialize(cls):
        """
        Initializes a Glasbey-like base palette with 50 highly distinct colors.
        """
        if cls._initialized:
            Logger.log_message_static("SignalColors already initialized", Logger.DEBUG)
            return

        Logger.log_message_static("Initializing Glasbey base palette", Logger.INFO)
        base_colors = get_colors(50)
        for idx, color in enumerate(base_colors):
            hex_color = cls._rgb_to_hex(color)
            cls._used_colors.append(color)
            cls._color_map[f"__PREGEN__{idx}"] = hex_color
        cls._initialized = True

    @classmethod
    def get_color_for_name(cls, name: str) -> str:
        """
        Returns a consistent and distinct color for the given signal name.
        If name is not known, generates a new color as distinct as possible.
        """
        if name in cls._color_map:
            return cls._color_map[name]

        Logger.log_message_static(f"Assigning new color to '{name}'", Logger.DEBUG)
        try:
            new_color = cls._generate_distinct_color()
            cls._used_colors.append(new_color)
            hex_color = cls._rgb_to_hex(new_color)
            cls._color_map[name] = hex_color
            Logger.log_message_static(f"Color assigned to '{name}': {hex_color}", Logger.DEBUG)
            return hex_color
        except Exception as e:
            Logger.log_message_static(f"Color assignment failed for '{name}': {e}", Logger.ERROR)
            return cls._fallback_color

    @staticmethod
    def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    @classmethod
    def _generate_distinct_color(cls) -> tuple[float, float, float]:
        """
        Generate a color maximally distinct from _used_colors using Glasbey-style distance metric,
        and reject too-dark candidates (unsuitable for dark background).
        """
        Logger.log_message_static("Generating distinct color (with brightness check)...", Logger.DEBUG)

        # Sample candidate space
        for attempt in range(5):  # up to 5 attempts to find a valid color
            candidates = np.random.rand(1000, 3)

            # Remove too-dark colors (perceived luminance under 0.25)
            def luminance(rgb):
                r, g, b = rgb
                return 0.2126 * r + 0.7152 * g + 0.0722 * b  # Rec. 709 luminance

            candidates = np.array([c for c in candidates if luminance(c) >= 0.25])

            if len(candidates) == 0:
                Logger.log_message_static("All candidates filtered out (too dark), retrying...", Logger.WARNING)
                continue

            used = np.array(cls._used_colors)
            if len(used) == 0:
                return tuple(candidates[0])

            # Compute perceptual distance
            dists = np.min(np.linalg.norm(candidates[:, None, :] - used[None, :, :], axis=2), axis=1)

            best_idx = np.argmax(dists)
            best_color = tuple(candidates[best_idx])
            Logger.log_message_static(f"New distinct color selected: {cls._rgb_to_hex(best_color)}", Logger.DEBUG)
            return best_color

        Logger.log_message_static("Failed to find suitable color after filtering, returning fallback", Logger.ERROR)
        return (1.0, 0.0, 0.0)  # fallback = bright red
