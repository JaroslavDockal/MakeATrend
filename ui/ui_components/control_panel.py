"""
Setup functions for the control panel in the SignalViewer.
"""
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox,
    QScrollArea, QWidget, QSpinBox, QLineEdit
)

from analysis.analysis_dialog import show_analysis_dialog


def setup_control_panel(viewer):
    """Sets up the entire control panel with all its components."""
    layout = QVBoxLayout(viewer.control_panel)

    setup_button_section(viewer, layout)
    setup_checkboxes(viewer, layout)
    setup_filter_section(viewer, layout)

    viewer.scroll = QScrollArea()
    viewer.scroll.setWidgetResizable(True)
    viewer.scroll_content = QWidget()
    viewer.scroll.setWidget(viewer.scroll_content)
    viewer.scroll_layout = QVBoxLayout(viewer.scroll_content)
    layout.addWidget(viewer.scroll)

    setup_cursors(viewer)


def setup_button_section(viewer, layout):
    """Sets up the button section at the top of the control panel."""
    button_widget = QWidget()
    button_layout = QHBoxLayout(button_widget)

    # Create two columns
    left_column = QVBoxLayout()
    right_column = QVBoxLayout()

    load_btn = QPushButton("Load Files")
    load_btn.clicked.connect(lambda: viewer.load_data(multiple=True))
    load_btn.setToolTip("Open file dialog to select and load one or more files")
    left_column.addWidget(load_btn)

    viewer.toggle_mode_btn = QPushButton("Advanced Mode")
    viewer.toggle_mode_btn.setCheckable(True)
    viewer.toggle_mode_btn.toggled.connect(viewer.toggle_complex_mode)
    viewer.toggle_mode_btn.setToolTip("Show additional signal controls (color, width, axis selection)")
    left_column.addWidget(viewer.toggle_mode_btn)

    viewer.toggle_panel_btn = QPushButton("Hide Panel")
    viewer.toggle_panel_btn.setCheckable(True)
    viewer.toggle_panel_btn.toggled.connect(viewer.toggle_right_panel)
    viewer.toggle_panel_btn.setToolTip("Hide control panel")
    left_column.addWidget(viewer.toggle_panel_btn)

    export_btn = QPushButton("Export Graph")
    export_btn.clicked.connect(viewer.export_graph_via_utils)
    export_btn.setToolTip("Save current graph as image (PNG) or PDF file")
    right_column.addWidget(export_btn)

    analysis_btn = QPushButton("Signal Analysis")
    analysis_btn.clicked.connect(lambda: show_analysis_dialog(viewer))
    analysis_btn.setToolTip("Open signal analysis tools (FFT, statistics, etc.)")
    right_column.addWidget(analysis_btn)

    virtual_btn = QPushButton("Add Virtual Signal")
    virtual_btn.clicked.connect(viewer.add_virtual_signal)
    virtual_btn.setToolTip("Create calculated signals using expressions with existing signals")
    right_column.addWidget(virtual_btn)

    button_layout.addLayout(left_column)
    button_layout.addLayout(right_column)

    layout.addWidget(button_widget)


def setup_checkboxes(viewer, layout):
    """Sets up the checkboxes section in the control panel."""
    checkbox_container = QWidget()
    checkbox_layout = QHBoxLayout(checkbox_container)

    col1 = QVBoxLayout()
    col2 = QVBoxLayout()

    viewer.white_background_chk = QCheckBox("White Background")
    viewer.white_background_chk.setChecked(False)
    viewer.white_background_chk.toggled.connect(viewer.toggle_background)
    viewer.white_background_chk.setToolTip("Switch to white/black scale for better printouts")
    col1.addWidget(viewer.white_background_chk)

    viewer.toggle_grid_chk = QCheckBox("Show Grid")
    viewer.toggle_grid_chk.setChecked(True)
    viewer.toggle_grid_chk.toggled.connect(viewer.toggle_grid)
    viewer.toggle_grid_chk.setToolTip("Show/hide grid lines on graph")
    col1.addWidget(viewer.toggle_grid_chk)

    viewer.cursor_a_chk = QCheckBox("Show Cursor A")
    viewer.cursor_a_chk.toggled.connect(lambda s: viewer.toggle_cursor(viewer.cursor_a, s))
    viewer.cursor_a_chk.setToolTip("Show/hide magenta vertical cursor line")
    col1.addWidget(viewer.cursor_a_chk)

    viewer.dock_cursor_info_chk = QCheckBox("Dock Cursor Info")
    viewer.dock_cursor_info_chk.toggled.connect(viewer.toggle_cursor_info_mode)
    viewer.dock_cursor_info_chk.setToolTip("Show cursor measurements directly in control panel")
    col1.addWidget(viewer.dock_cursor_info_chk)

    viewer.downsample_chk = QCheckBox("Downsample")
    viewer.downsample_chk.setChecked(False)
    viewer.downsample_chk.setToolTip("Reduce data points for better performance with large datasets")
    col1.addWidget(viewer.downsample_chk)

    #TODO not working - might be used for something else. Leave as placeholder
    viewer.log_scale_chk = QCheckBox("Logaritmic Scale")
    viewer.log_scale_chk.setChecked(False)
    viewer.log_scale_chk.toggled.connect(viewer.toggle_axis_scale)
    viewer.log_scale_chk.setToolTip("Switch to axis logaritmic/Decimal format.")
    col2.addWidget(viewer.log_scale_chk)

    viewer.toggle_crosshair_chk = QCheckBox("Show Crosshair")
    viewer.toggle_crosshair_chk.toggled.connect(viewer.toggle_crosshair)
    viewer.toggle_crosshair_chk.setToolTip("Show/hide cursor crosshair following mouse")
    col2.addWidget(viewer.toggle_crosshair_chk)

    viewer.cursor_b_chk = QCheckBox("Show Cursor B")
    viewer.cursor_b_chk.toggled.connect(lambda s: viewer.toggle_cursor(viewer.cursor_b, s))
    viewer.cursor_b_chk.setToolTip("Show/hide cyan vertical cursor line")
    col2.addWidget(viewer.cursor_b_chk)

    viewer.show_log_chk = QCheckBox("Show Log")
    viewer.show_log_chk.toggled.connect(viewer.toggle_log_window)
    viewer.show_log_chk.setToolTip("Show application log messages")
    col2.addWidget(viewer.show_log_chk)

    viewer.downsample_points = QSpinBox()
    viewer.downsample_points.setRange(100, 100000)
    viewer.downsample_points.setValue(5000)
    viewer.downsample_points.setEnabled(True)
    viewer.downsample_points.setToolTip("Maximum number of points to display per signal")
    col2.addWidget(viewer.downsample_points)

    checkbox_layout.addLayout(col1)
    checkbox_layout.addLayout(col2)
    layout.addWidget(checkbox_container)


def setup_filter_section(viewer, layout):
    """Sets up the signal filter section in the control panel."""
    viewer.filter_box = QLineEdit()
    viewer.filter_box.setPlaceholderText("Filter signals...")
    viewer.filter_box.textChanged.connect(viewer.apply_signal_filter)
    viewer.filter_box.setToolTip("Type to filter signal list by name")
    layout.addWidget(viewer.filter_box)

    layout.addWidget(QLabel("Signals:"))


def setup_cursors(viewer):
    """Sets up cursor lines and cursor info dialog."""
    from ui.widgets.cursor_info import CursorInfoDialog
    from ui.widgets.crosshair import Crosshair

    viewer.cursor_a = pg.InfiniteLine(angle=90, movable=True, pen='m')
    viewer.cursor_b = pg.InfiniteLine(angle=90, movable=True, pen='c')
    viewer.cursor_a.setVisible(False)
    viewer.cursor_b.setVisible(False)
    viewer.plot_widget.addItem(viewer.cursor_a)
    viewer.plot_widget.addItem(viewer.cursor_b)

    viewer.cursor_a.sigPositionChanged.connect(viewer.update_cursor_info)
    viewer.cursor_b.sigPositionChanged.connect(viewer.update_cursor_info)

    viewer.cursor_info = CursorInfoDialog(viewer)
    viewer.crosshair = Crosshair(viewer.main_view)