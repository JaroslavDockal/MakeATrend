"""
Control panel operational functions for the CSV Signal Viewer.
"""
import datetime
import numpy as np

from data.signal_utils import find_nearest_index


def toggle_complex_mode(self, state):
    """
    Enables or disables advanced plotting controls.

    Args:
        state (bool): True to enable advanced options.
    """
    self.log_message(f"Advanced mode {'enabled' if state else 'disabled'}", self.INFO)
    self.complex_mode = state
    for widgets in self.signal_widgets.values():
        widgets['axis'].setVisible(state)
        widgets['color_btn'].setVisible(state)
        widgets['width'].setVisible(state)


def toggle_right_panel(self, checked):
    """
    Show/hide the entire control panel.

    Args:
        visible (bool): Whether to show the panel.
    """
    self.log_message(f"Control panel {'hidden' if checked else 'shown'}", self.DEBUG)
    self.control_panel.setVisible(not checked)
    self.show_panel_btn.setVisible(checked)


def toggle_cursor_info_mode(self, docked):
    """
    Moves cursor info to control panel or keeps it in its own window.

    Args:
        docked (bool): If True, docks the info panel.
    """
    self.log_message(f"Cursor info panel {'docked' if docked else 'undocked'}", self.DEBUG)

    if docked:
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget and widget != self.cursor_info:
                widget.setParent(None)
        self.scroll_layout.addWidget(self.cursor_info)
    else:
        self.scroll_layout.removeWidget(self.cursor_info)
        self.cursor_info.setParent(None)

        for name, widgets in self.signal_widgets.items():
            self.scroll_layout.addWidget(widgets['row'])


def toggle_crosshair(self, state):
    """
    Toggles the crosshair visibility.

    Args:
        state (bool): True to show, False to hide.
    """
    self.log_message(f"Crosshair {'enabled' if state else 'disabled'}", self.DEBUG)
    self.crosshair.toggle(state)


def toggle_grid(self, state: bool):
    """
    Enables/disables X and Y grid lines on the plot.

    Args:
        state (bool): True = show grid, False = hide.
    """
    self.log_message(f"Grid lines {'shown' if state else 'hidden'}", self.DEBUG)
    self.plot_widget.showGrid(x=state, y=state)


def toggle_cursor(self, cursor, state):
    """
    Shows or hides a specific vertical cursor line.

    Args:
        cursor (pg.InfiniteLine): The cursor line.
        state (bool): Visibility flag.
    """
    self.log_message(f"{'Showing' if state else 'Hiding'} cursor {'A' if cursor == self.cursor_a else 'B'}", self.DEBUG)
    cursor.setVisible(state)
    if state:
        try:
            mid = None
            for name, (time_arr, _) in self.data_signals.items():
                if time_arr is not None and len(time_arr) > 0:
                    mid = time_arr[len(time_arr) // 2]
                    break
            if mid is not None:
                cursor.setPos(mid)
        except Exception:
            pass
    self.cursor_info.setVisible(self.cursor_a.isVisible() or self.cursor_b.isVisible())
    self.update_cursor_info()


def update_cursor_info(self):
    """
    Refreshes the floating or docked cursor data panel.
    Now supports individual time vectors for each signal.
    """
    if not self.cursor_info.isVisible():
        return

    has_a = self.cursor_a.isVisible()
    has_b = self.cursor_b.isVisible()

    t_a = self.cursor_a.value() if has_a else None
    t_b = self.cursor_b.value() if has_b else None

    fmt = "%H:%M:%S.%f"
    try:
        s_a = datetime.datetime.fromtimestamp(t_a).strftime(fmt)[:-3] if has_a else "-"
        s_b = datetime.datetime.fromtimestamp(t_b).strftime(fmt)[:-3] if has_b else "-"
    except Exception:
        s_a, s_b = "-", "-"

    try:
        date_a = datetime.datetime.fromtimestamp(t_a) if has_a else None
        date_b = datetime.datetime.fromtimestamp(t_b) if has_b else None
        # Use the earlier date if both are valid, otherwise use whichever is available
        if date_a and date_b:
            date = min(date_a, date_b)
        elif date_a:
            date = date_a
        elif date_b:
            date = date_b
        date_str = date.strftime("%d-%b-%Y") if date else "Unknown"
    except Exception:
        date_str = "Unknown"

    self.log_message(f"Updating cursor info: A={s_a}, B={s_b}", self.DEBUG)

    def get_vals(t, enabled):
        vals = {}
        if not enabled:
            return vals
        for name in self.curves:
            time_arr, value_arr = self.data_signals.get(name, (None, None))
            if time_arr is not None and value_arr is not None:
                try:
                    idx = find_nearest_index(time_arr, t)
                    vals[name] = value_arr[idx]
                except Exception:
                    vals[name] = np.nan
        return vals

    v_a = get_vals(t_a, has_a)
    v_b = get_vals(t_b, has_b)

    self.cursor_info.update_data(s_a, s_b, v_a, v_b, has_a, has_b, date_str)


def toggle_log_window(self, state):
    """
    Show or hide the log window.

    Args:
        state (bool): True to show, False to hide.
    """
    self.log_message(f"Log window {'shown' if state else 'hidden'}", self.DEBUG)

    if state:
        self.log_window.show()
        # Connect log window to logger
        self.logger.set_log_window(self.log_window)
    else:
        self.log_window.hide()