"""
Crosshair overlay for the signal plot.
Displays interactive vertical and horizontal lines with labels at mouse position.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from datetime import datetime


class Crosshair:
    """
    Class representing an interactive crosshair that shows position labels.

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

        # Vertical and horizontal dashed lines
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((200, 200, 200), style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((200, 200, 200), style=Qt.DashLine))
        self.viewbox.addItem(self.vline, ignoreBounds=True)
        self.viewbox.addItem(self.hline, ignoreBounds=True)

        # Text labels for X and Y positions
        self.label_x = pg.TextItem(anchor=(1, 1), color=(200, 200, 200))
        self.label_y = pg.TextItem(anchor=(0, 0), color=(200, 200, 200))
        self.label_y.setText("")  # init empty
        self.viewbox.addItem(self.label_x)
        self.viewbox.addItem(self.label_y)

        # Signal proxy for mouse movement
        self.proxy = pg.SignalProxy(
            self.viewbox.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouse_moved
        )

        self.hide()

    def toggle(self, state: bool):
        """
        Enable or disable the crosshair.

        Args:
            state (bool): True to enable, False to disable.
        """
        self.enabled = state
        self.viewbox.setMouseEnabled(x=not state, y=not state)
        if state:
            self.show()
        else:
            self.hide()

    def show(self):
        """Show all crosshair elements."""
        self.vline.show()
        self.hline.show()
        self.label_x.show()
        self.label_y.show()

    def hide(self):
        """Hide all crosshair elements."""
        self.vline.hide()
        self.hline.hide()
        self.label_x.hide()
        self.label_y.hide()

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

        # Map scene position to plot coordinates
        mouse_point = self.viewbox.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        # Move crosshair lines
        self.vline.setPos(x)
        self.hline.setPos(y)

        # Update label positions
        view_range = self.viewbox.viewRange()
        self.label_x.setPos(x, view_range[1][0])  # bottom X
        self.label_y.setPos(view_range[0][0], y)  # left Y

        # Format X value as timestamp if possible
        try:
            time_str = datetime.fromtimestamp(x).strftime("%H:%M:%S.%f")[:-3]
            self.label_x.setText(f"X: {time_str}")
        except Exception:
            self.label_x.setText("X: -")

        # Format Y value as number
        self.label_y.setText(f"Y: {y:.3f}")
