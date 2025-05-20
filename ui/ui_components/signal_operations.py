"""
Signal operations including virtual signals and graph export for the CSV Signal Viewer.
"""
from PySide6.QtWidgets import QMessageBox, QFileDialog

from data.graph_export import export_graph
from analysis.analysis_dialog import show_analysis_dialog


def add_virtual_signal(self):
    """
    Opens a dialog for creating a new virtual signal from an expression.
    The user defines a name and an expression based on existing signals.
    If the expression is valid, the new signal is added and plotted.
    """
    self.log_message("Opening virtual signal creation dialog", self.DEBUG)

    signal_names = list(self.data_signals.keys())

    if not signal_names:
        QMessageBox.warning(self, "Virtual Signal", "No signals loaded. Load some signals first.")
        self.log_message("Cannot create virtual signal: No signals loaded", self.WARNING)
        return

    from ui.widgets.virtual_signal_dialog import VirtualSignalDialog
    dialog = VirtualSignalDialog(signal_names, self)
    if dialog.exec():
        signal_name, expression, alias_mapping = dialog.get_result()

        if expression.strip() == "":
            self.log_message("Empty expression provided for virtual signal", self.WARNING)
            return

        try:
            # Use the dedicated compute_virtual_signal function from virtual_signal_dialog
            from ui.widgets.virtual_signal_dialog import compute_virtual_signal

            # Compute the virtual signal
            self.log_message(f"Computing virtual signal '{signal_name}' with expression: {expression}", self.DEBUG)
            time_array, values = compute_virtual_signal(expression, alias_mapping, self.data_signals)
            self.log_message(f"Virtual signal calculation complete: {len(values)} points generated", self.DEBUG)

            # Add the virtual signal to the data dictionary
            self.data_signals[signal_name] = (time_array, values)

            # Create UI row for the new signal
            row = self.build_signal_row(signal_name)
            self.scroll_layout.addWidget(row)

            # Auto-select the new signal
            self.signal_widgets[signal_name]['checkbox'].setChecked(True)

            QMessageBox.information(self, "Virtual Signal",
                                    f"Virtual signal '{signal_name}' created successfully.")
            self.log_message(f"Virtual signal '{signal_name}' created successfully with expression: {expression}",
                             self.INFO)
        except Exception as e:
            QMessageBox.critical(self, "Virtual Signal Error", str(e))


def export_graph_simple(self):
    self.log_message("Starting graph export...", self.INFO)
    try:
        from pyqtgraph.exporters import ImageExporter
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", "graph.png", "PNG Images (*.png)"
        )
        if file_path:
            exporter = ImageExporter(self.plot_widget.plotItem)
            exporter.export(file_path)
            self.log_message(f"Graph exported successfully to {file_path}", self.INFO)
        else:
            self.log_message("Graph export cancelled by user", self.DEBUG)
    except ImportError:
        self.log_message("pyqtgraph.exporters not available. Cannot export graph.", self.ERROR)
    except Exception as e:
        self.log_message(f"Graph export failed: {str(e)}", self.ERROR)


def export_graph_via_utils(self):
    """
    Wrapper method to call the export_graph function from utils.
    """
    self.log_message("Exporting graph via utils...", self.DEBUG)
    export_graph(self.plot_widget, self)


def open_analysis_dialog(self):
    show_analysis_dialog(self)