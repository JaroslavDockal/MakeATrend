"""
Crosshair overlay logic for displaying X and Y axis values in the plot.
"""

from PySide6.QtCore import Qt
import pyqtgraph as pg


class Crosshair:
    """
    Class managing crosshair overlay in a PyQtGraph ViewBox.

    Attributes:
        viewbox (pg.ViewBox): The plot area viewbox.
        enabled (bool): Whether crosshair is active.
    """

    def __init__(self, viewbox):
        """
        Initializes the Crosshair object.

        Args:
            viewbox (pg.ViewBox): ViewBox to overlay crosshair on.
        """
        self.viewbox = viewbox
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((150, 150, 150), style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((150, 150, 150), style=Qt.DashLine))
        self.viewbox.addItem(self.vline, ignoreBounds=True)
        self.viewbox.addItem(self.hline, ignoreBounds=True)

        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        self.label_x = pg.TextItem(anchor=(1, 1), color=(200, 200, 200))
        self.label_y = pg.TextItem(anchor=(1, 0), color=(200, 200, 200))
        self.viewbox.addItem(self.label_x)
        self.viewbox.addItem(self.label_y)

        self.enabled = False
        self.last_pos = None
        self.hide()

    def toggle(self, state):
        """
        Enables or disables the crosshair.

        Args:
            state (bool): True to enable, False to disable.
        """
        self.enabled = state
        if state:
            self.last_pos = self.viewbox.mapSceneToView(
                self.viewbox.mapToScene(self.viewbox.width() // 2, self.viewbox.height() // 2)
            )
            self.show()
            self.update_crosshair()
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
        Handler for mouse movement events.

        Args:
            evt (tuple): PyQtGraph mouse move event.
        """
        if not self.enabled:
            return
        pos = evt[0]
        if self.viewbox.sceneBoundingRect().contains(pos):
            mouse_point = self.viewbox.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            self.vline.setPos(x)
            self.hline.setPos(y)
            self.label_x.setPos(x, self.viewbox.viewRange()[1][0])
            self.label_y.setPos(self.viewbox.viewRange()[0][0], y)

            try:
                timestamp = datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S.%f")[:-3]
                self.label_x.setText(f"{timestamp}")
            except Exception:
                self.label_x.setText(f"{x:.2f}")

            self.label_y.setText(f"{y:.2f}")

    def update_crosshair(self):
        """Updates the crosshair position and label values."""
        if not self.enabled or not self.last_pos:
            return
        x = self.last_pos.x()
        y = self.last_pos.y()
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.label_x.setPos(x, self.viewbox.viewRange()[1][0])
        self.label_y.setPos(self.viewbox.viewRange()[0][0], y)
        self.label_x.setText(f"X: {x:.2f}")
        self.label_y.setText(f"Y: {y:.2f}")
