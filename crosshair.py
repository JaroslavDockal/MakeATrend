"""
Crosshair overlay for the signal plot.
Displays interactive vertical and horizontal lines with labels at mouse position.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from datetime import datetime
from logger import Logger

class Crosshair:
    """
    Class representing an interactive crosshair with labels for both X and Y axes,
    including support for right Y-axis and timestamp formatting.

    Attributes:
        viewbox (pg.ViewBox): The plot viewbox where the crosshair is drawn.
        enabled (bool): Whether the crosshair is currently active.
    """

    def __init__(self, viewbox: pg.ViewBox):
        """
        Initializes the crosshair overlay.

        Args:
            viewbox (pg.ViewBox): The ViewBox to attach the crosshair to.
        """
        Logger.log_message_static("Initializing crosshair overlay", Logger.DEBUG)
        self.viewbox = viewbox
        self.enabled = False

        # Dashed cross lines
        Logger.log_message_static("Creating vertical and horizontal infinite lines", Logger.DEBUG)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DashLine))
        self.viewbox.addItem(self.vline, ignoreBounds=True)
        self.viewbox.addItem(self.hline, ignoreBounds=True)

        # Labels
        Logger.log_message_static("Creating text labels for coordinates", Logger.DEBUG)
        self.label_x = pg.TextItem(anchor=(1, 1), color=(200, 200, 200))
        self.label_y_left = pg.TextItem(anchor=(0, 0), color=(200, 200, 200))
        self.label_y_right = pg.TextItem(anchor=(1, 0), color=(200, 200, 200))
        self.viewbox.addItem(self.label_x)
        self.viewbox.addItem(self.label_y_left)
        self.viewbox.addItem(self.label_y_right)

        # Signal proxy for mouse movement
        scene = self.viewbox.scene()
        if isinstance(scene, pg.GraphicsScene):
            Logger.log_message_static("Setting up mouse movement signal proxy with rate limiting", Logger.DEBUG)
            self.proxy = pg.SignalProxy(
                scene.sigMouseMoved,
                rateLimit=60,
                slot=self.mouse_moved
            )
        else:
            Logger.log_message_static("Warning: ViewBox scene is not a GraphicsScene, mouse tracking may not work", Logger.WARNING)

        self.mouse_locked = False
        self.local_tz = datetime.now().astimezone().tzinfo
        Logger.log_message_static(f"Using local timezone: {self.local_tz}", Logger.DEBUG)

        Logger.log_message_static("Initially hiding crosshair elements", Logger.DEBUG)
        self.hide()
        Logger.log_message_static("Crosshair initialization complete", Logger.DEBUG)

    def toggle(self, state: bool):
        """
        Enable or disable the crosshair.

        Args:
            state (bool): True = enable, False = disable
        """
        Logger.log_message_static(f"Toggling crosshair to {'enabled' if state else 'disabled'}", Logger.INFO)
        self.enabled = state
        self.viewbox.setMouseEnabled(x=not state, y=not state)
        if state:
            Logger.log_message_static("Disabling mouse zoom/pan for crosshair mode", Logger.DEBUG)
            # workaround to prevent zooming on first activation
            QTimer.singleShot(0, lambda: self.viewbox.setMouseEnabled(x=False, y=False))
            self.show()
        else:
            Logger.log_message_static("Re-enabling mouse zoom/pan", Logger.DEBUG)
            self.hide()

    def show(self):
        """Show crosshair lines and labels."""
        Logger.log_message_static("Showing crosshair elements", Logger.DEBUG)
        self.vline.show()
        self.hline.show()
        self.label_x.show()
        self.label_y_left.show()
        self.label_y_right.show()

    def hide(self):
        """Hide all crosshair elements."""
        Logger.log_message_static("Hiding crosshair elements", Logger.DEBUG)
        self.vline.hide()
        self.hline.hide()
        self.label_x.hide()
        self.label_y_left.hide()
        self.label_y_right.hide()

    def mouse_moved(self, evt):
        """
        Callback for mouse movement over the plot area.

        Args:
            evt: Event with scene position info.
        """
        if not self.enabled:
            return

        pos = evt[0]
        if not self.viewbox.sceneBoundingRect().contains(pos):
            return

        # Prevent mouse-zooming (redundant fallback)
        self.viewbox.setMouseEnabled(x=False, y=False)

        try:
            mouse_point = self.viewbox.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()

            Logger.log_message_static(f"Crosshair position: x={x:.3f}, y={y:.3f}", Logger.DEBUG)

            # Move lines
            self.vline.setPos(x)
            self.hline.setPos(y)

            # Update labels
            view_range = self.viewbox.viewRange()
            self.label_x.setPos(x, view_range[1][0])
            self.label_y_left.setPos(view_range[0][0], y)
            self.label_y_right.setPos(view_range[0][1], y)

            try:
                time_str = datetime.fromtimestamp(x, tz=self.local_tz).strftime("%H:%M:%S.%f")[:-3]
                self.label_x.setText(f"X: {time_str}")
                Logger.log_message_static(f"Formatted timestamp: {time_str}", Logger.DEBUG)
            except Exception as e:
                Logger.log_message_static(f"Failed to format timestamp from value {x}: {str(e)}", Logger.WARNING)
                self.label_x.setText("X: -")

            self.label_y_left.setText(f"Y: {y:.3f}")
            self.label_y_right.setText(f"Y: {y:.3f}")

        except Exception as e:
            Logger.log_message_static(f"Error updating crosshair: {str(e)}", Logger.ERROR)