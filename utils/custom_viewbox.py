import pyqtgraph as pg

class CustomViewBox(pg.ViewBox):
    """
    Custom ViewBox that disables mouse wheel zooming when locked externally.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crosshair_locked = False

    def wheelEvent(self, ev):
        """
        Override wheelEvent to block zoom if crosshair is active.
        """
        if not self.crosshair_locked:
            super().wheelEvent(ev)

    def setCrosshairLock(self, locked: bool):
        """
        Lock or unlock zooming via mouse wheel.

        Args:
            locked (bool): If True, disables wheel-based zoom.
        """
        self.crosshair_locked = locked
        self.setMouseEnabled(x=not locked, y=not locked)
