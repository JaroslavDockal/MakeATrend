"""
Crosshair overlay for the signal plot.
Displays interactive vertical and horizontal lines with labels at mouse position.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from datetime import datetime


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
        self.viewbox = viewbox
        self.enabled = False

        # Dashed cross lines
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((200, 200, 200), style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((200, 200, 200), style=Qt.DashLine))
        self.viewbox.addItem(self.vline, ignoreBounds=True)
        self.viewbox.addItem(self.hline, ignoreBounds=True)

        # Labels
        self.label_x = pg.TextItem(anchor=(1, 1), color=(200, 200, 200))
        self.label_y_left = pg.TextItem(anchor=(0, 0), color=(200, 200, 200))
        self.label_y_right = pg.TextItem(anchor=(1, 0), color=(200, 200, 200))
        self.viewbox.addItem(self.label_x)
        self.viewbox.addItem(self.label_y_left)
        self.viewbox.addItem(self.label_y_right)

        # Signal proxy for mouse movement
        self.proxy = pg.SignalProxy(
            self.viewbox.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouse_moved
        )

        self.mouse_locked = False
        self.local_tz = datetime.now().astimezone().tzinfo

        self.hide()

    def toggle(self, state: bool):
        """
        Enable or disable the crosshair.

        Args:
            state (bool): True = enable, False = disable
        """
        self.enabled = state
        self.viewbox.setMouseEnabled(x=not state, y=not state)
        if state:
            # workaround to prevent zooming on first activation
            QTimer.singleShot(0, lambda: self.viewbox.setMouseEnabled(x=False, y=False))
            self.show()
        else:
            self.hide()

    def show(self):
        """Show crosshair lines and labels."""
        self.vline.show()
        self.hline.show()
        self.label_x.show()
        self.label_y_left.show()
        self.label_y_right.show()

    def hide(self):
        """Hide all crosshair elements."""
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

        mouse_point = self.viewbox.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

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
        except Exception:
            self.label_x.setText("X: -")

        self.label_y_left.setText(f"Y: {y:.3f}")
        self.label_y_right.setText(f"Y: {y:.3f}")
