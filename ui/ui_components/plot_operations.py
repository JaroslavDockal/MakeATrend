"""
Signal plotting and management operations for the CSV Signal Viewer.
"""
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QComboBox, QPushButton, QSpinBox, QColorDialog

from data.signal_utils import is_digital_signal
from utils.signal_colors import SignalColors


def build_signal_row(self, name):
    """
    Constructs UI row for a given signal.

    Args:
        name (str): Name of the signal.

    Returns:
        QWidget: The row widget.
    """
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    cb = QCheckBox(name)
    cb.stateChanged.connect(self.toggle_signal)

    signal_color = SignalColors.get_color_for_name(name)

    color_btn = QPushButton("Color")
    color_btn.setStyleSheet(f"background-color: {signal_color}")
    color_btn.clicked.connect(lambda _, b=color_btn: self.pick_color(b))
    color_btn.setVisible(False)

    width_spin = QSpinBox()
    width_spin.setRange(1, 10)
    width_spin.setValue(2)
    width_spin.setVisible(False)

    axis_cb = QComboBox()
    axis_cb.addItems(['Left', 'Right'])
    axis_cb.setVisible(False)

    row_layout.addWidget(cb)
    row_layout.addWidget(axis_cb)
    row_layout.addWidget(color_btn)
    row_layout.addWidget(width_spin)

    self.signal_widgets[name] = {
        'checkbox': cb,
        'axis': axis_cb,
        'color_btn': color_btn,
        'width': width_spin,
        'row': row
    }

    return row


def clear_signals(self):
    """
    Removes all signal plots and resets widgets.
    """
    self.log_message("Components-PlotOp: Clearing all signals from plot", self.DEBUG)
    for curve in self.curves.values():
        for vb in self.viewboxes.values():
            vb.removeItem(curve)

    self.curves.clear()
    self.signal_axis_map = {k: [] for k in self.viewboxes}
    self.signal_styles.clear()
    self.color_counter = 0  # Reset color counter

    for i in reversed(range(self.scroll_layout.count())):
        widget = self.scroll_layout.itemAt(i).widget()
        if widget:
            widget.setParent(None)

    self.signal_widgets.clear()
    self.update_axis_labels()


def toggle_signal(self):
    """
    Handles toggling of a signal. Adds or removes the signal from the plot.
    Automatically assigns boolean signals to the Digital axis with step-style rendering.
    """
    cb = self.sender()
    for name, widgets in self.signal_widgets.items():
        if widgets['checkbox'] == cb:
            if cb.isChecked():
                self.log_message(f"Components-PlotOp: Displaying signal: {name}", self.DEBUG)
                # Použít barvu z tlačítka, které již bylo inicializováno pomocí SignalColors
                color = widgets['color_btn'].palette().button().color().name()
                width = 2
                axis = 'Left'
                if self.complex_mode:
                    width = widgets['width'].value()
                    axis = widgets['axis'].currentText()

                if is_digital_signal(self.data_signals[name]):
                    axis = 'Digital'
                    style = dict(stepMode=True, width=3)
                    pen = pg.mkPen(color=color, **style)
                    self.viewboxes['Digital'].setYRange(-0.1, 1.1, padding=0.1)
                    # Convert TRUE/FALSE to 1.0/0.0
                    time_arr, value_arr = self.data_signals[name]
                    self.log_message(f"Components-PlotOp: Original values for {name}: {value_arr}", self.DEBUG)
                    numeric_values = np.array(
                        [1.0 if str(val).upper() == 'TRUE' else 0.0 for val in value_arr if val is not None],
                        dtype=np.float32)
                    self.log_message(f"Components-PlotOp: Converted values for {name}: {value_arr}", self.DEBUG)

                    # Use the converted numeric values instead of the original ones
                    curve = pg.PlotCurveItem(x=time_arr, y=numeric_values, pen=pen)

                    # Force the x-axis to show proper time range for digital signals
                    if len(time_arr) > 0:
                        self.main_view.setXRange(min(time_arr), max(time_arr), padding=0.1)
                        self.plot_widget.getAxis('bottom').setVisible(True)
                else:
                    style = dict(width=width)
                    pen = pg.mkPen(color=color, **style)
                    time_arr, value_arr = self.data_signals[name]

                    # Downsample data before plotting
                    # Only downsample if checkbox is checked
                    if hasattr(self, 'downsample_chk') and self.downsample_chk.isChecked():
                        max_points = self.downsample_points.value()
                        time_arr, value_arr = self.downsample_signal(time_arr, value_arr, max_points)

                    curve = pg.PlotCurveItem(x=time_arr, y=value_arr, pen=pen)

                self.viewboxes[axis].addItem(curve)
                self.curves[name] = curve
                self.signal_axis_map[axis].append(name)
                self.signal_styles[name] = (axis, color, style['width'])

                widgets['checkbox'].setStyleSheet(f"color: {color};")
                # Force a refresh of the plot
                self.plot_widget.plotItem.update()
            else:
                self.log_message(f"Components-PlotOp: Hiding signal: {name}", self.DEBUG)
                curve = self.curves.pop(name, None)
                if curve:
                    axis = self.signal_styles[name][0]
                    self.viewboxes[axis].removeItem(curve)
                    self.signal_axis_map[axis].remove(name)
                    del self.signal_styles[name]
                widgets['checkbox'].setStyleSheet("color: white;")
            self.update_axis_labels()
            break


def update_axis_labels(self):
    """
    Updates the Y-axis labels with colored names of plotted signals in one row.
    """
    self.log_message(
        f"Updating axis labels with {len(self.signal_axis_map['Left'])} left, {len(self.signal_axis_map['Right'])} right, {len(self.signal_axis_map['Digital'])} digital signals",
        self.DEBUG)
    for axis, label in self.axis_labels.items():
        names = self.signal_axis_map.get(axis, [])
        html_parts = []
        for name in names:
            base_name = name.split('[')[0].strip()
            _, color, _ = self.signal_styles.get(name, (None, "#FFFFFF", 2))
            html_parts.append(f'<span style="color:{color}">{base_name}</span>')

        # Create centered HTML layout with CSS
        if html_parts:
            html_text = f'<div style="text-align: center; width: 100%;">{", ".join(html_parts)}</div>'
        else:
            html_text = f'<div style="text-align: center; width: 100%;">{axis}</div>'

        label.setLabel(text=html_text, html=True)  # Set html=True to ensure HTML is processed


def pick_color(self, btn):
    """
    Opens color dialog to select line color.

    Args:
        btn (QPushButton): The button to apply color to.
    """
    # Find which signal this button belongs to
    signal_name = None
    for name, widgets in self.signal_widgets.items():
        if widgets['color_btn'] == btn:
            signal_name = name
            break

    color = QColorDialog.getColor()
    if color.isValid():
        btn.setStyleSheet(f"background-color: {color.name()}")
        self.log_message(f"Components-PlotOp: Changed signal color for '{signal_name}' to {color.name()}", self.DEBUG)

        # If signal is currently shown, update its color
        if signal_name in self.curves:
            curve = self.curves[signal_name]
            axis, _, width = self.signal_styles[signal_name]
            pen = pg.mkPen(color=color.name(), width=width)
            curve.setPen(pen)

            # Update signal style info
            self.signal_styles[signal_name] = (axis, color.name(), width)
            self.update_axis_labels()


def apply_signal_filter(self, text: str):
    """
    Filters signal rows in the panel based on user input.

    Args:
        text (str): Filter string (case-insensitive).
    """
    self.signal_filter_text = text.lower().strip()
    for name, widgets in self.signal_widgets.items():
        row_widget = widgets.get('row')
        visible = self.signal_filter_text in name.lower()
        row_widget.setVisible(visible)


def downsample_signal(self, time_arr, value_arr, max_points):
    """
    Downsamples time and value arrays to have at most max_points.
    Preserves shape and important features while reducing memory usage.

    Args:
        time_arr (np.ndarray): Original time array
        value_arr (np.ndarray): Original values array
        max_points (int): Maximum number of points to keep

    Returns:
        tuple: (downsampled_time, downsampled_values)
    """
    if len(time_arr) <= max_points:
        return time_arr, value_arr

    self.log_message(f"Components-PlotOp: Downsampling signal from {len(time_arr)} to ~{max_points} points", self.DEBUG)

    # Calculate stride for even sampling
    stride = len(time_arr) // max_points

    if stride > 2:
        self.log_message(
            f"Heavy downsampling applied - using stride of {stride} (original: {len(time_arr)}, target: {max_points})",
            self.WARNING)

    # Use stride-based sampling to reduce points
    ds_time = time_arr[::stride]
    ds_values = value_arr[::stride]

    # Ensure we keep the last point for proper range representation
    if len(time_arr) > 0 and ds_time[-1] != time_arr[-1]:
        ds_time = np.append(ds_time, time_arr[-1])
        ds_values = np.append(ds_values, value_arr[-1])

    self.log_message(f"Components-PlotOp: Downsampled signal from {len(time_arr)} to {len(ds_time)} points", self.DEBUG)

    return ds_time, ds_values