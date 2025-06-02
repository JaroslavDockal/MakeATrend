"""
Setup functions for the plot area in the SignalViewer.
"""
import pyqtgraph as pg

from PySide6.QtWidgets import QPushButton, QGraphicsProxyWidget


def setup_plot_area(viewer):
    """Sets up the main plot area and floating button."""
    setup_floating_button(viewer)


def setup_floating_button(viewer):
    """Sets up the floating button for hiding/showing the control panel."""
    viewer.show_panel_btn = QPushButton("☰")
    viewer.show_panel_btn.setFixedSize(30, 30)
    viewer.show_panel_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold; border: none;")
    viewer.show_panel_btn.clicked.connect(lambda: viewer.toggle_panel_btn.setChecked(False))
    viewer.show_panel_btn.setToolTip("Show control panel")
    viewer.show_panel_btn.setVisible(False)

    viewer.proxy_btn = QGraphicsProxyWidget()
    viewer.proxy_btn.setWidget(viewer.show_panel_btn)
    viewer.plot_widget.scene().addItem(viewer.proxy_btn)

    def update_button_pos():
        viewer.proxy_btn.setPos(viewer.plot_widget.width() - 40, 10)

    viewer.plot_widget.resizeEvent = lambda event: (
        pg.PlotWidget.resizeEvent(viewer.plot_widget, event), update_button_pos()
    )
    update_button_pos()


def setup_axes(viewer):
    """Sets up the multiple axes in the plot area."""
    viewer.viewboxes = {
        'Left': viewer.main_view,
        'Right': pg.ViewBox(),
        'Digital': pg.ViewBox()
    }
    viewer.plot_widget.scene().addItem(viewer.viewboxes['Right'])
    viewer.plot_widget.scene().addItem(viewer.viewboxes['Digital'])
    viewer.viewboxes['Right'].setXLink(viewer.main_view)
    viewer.viewboxes['Digital'].setXLink(viewer.main_view)
    viewer.viewboxes['Digital'].setYRange(-0.1, 1.1, padding=0.1)

    viewer.signal_axis_map = {k: [] for k in viewer.viewboxes}

    digital_axis = pg.AxisItem('right')
    digital_axis.setTicks([[(0, "F"), (1, "T")]])

    viewer.axis_labels = {
        'Left': viewer.plot_widget.getAxis('left'),
        'Right': viewer.plot_widget.getAxis('right'),
        'Digital': digital_axis
    }

    viewer.plot_widget.showAxis('right')
    viewer.axis_labels['Right'].linkToView(viewer.viewboxes['Right'])
    viewer.plot_widget.getPlotItem().layout.addItem(viewer.axis_labels['Digital'], 2, 4)
    viewer.axis_labels['Digital'].linkToView(viewer.viewboxes['Digital'])

    digital_background = pg.QtWidgets.QGraphicsRectItem()
    digital_background.setPen(pg.mkPen(None))
    digital_background.setBrush(pg.mkBrush(20, 20, 20, 50))  # Subtle dark background
    viewer.viewboxes['Digital'].addItem(digital_background)

    def sync_views():
        geom = viewer.main_view.sceneBoundingRect()
        viewer.viewboxes['Right'].setGeometry(geom)
        viewer.viewboxes['Digital'].setGeometry(geom)
        viewer.viewboxes['Right'].linkedViewChanged(viewer.main_view, viewer.viewboxes['Right'].XAxis)
        viewer.viewboxes['Digital'].linkedViewChanged(viewer.main_view, viewer.viewboxes['Digital'].XAxis)

    viewer.main_view.sigResized.connect(sync_views)