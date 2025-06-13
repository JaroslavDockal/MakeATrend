"""
Movable legend widget for the SignalViewer.
Displays signal names with color indicators that can be repositioned by dragging.
"""
import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItem
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPen, QBrush, QFont, QColor


class MovableLegend(QGraphicsRectItem):
    """
    A draggable legend that displays active signals with their colors.
    Can be positioned anywhere in the plot area and updates automatically
    when signals are toggled on/off.
    """

    def __init__(self, viewer):
        """
        Initialize the movable legend.

        Args:
            viewer: The main SignalViewer instance
        """
        super().__init__()
        self.viewer = viewer

        # Make the legend movable and selectable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Visual styling
        self.setPen(QPen(QColor(200, 200, 200), 0))
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.setOpacity(1.0)

        # Layout constants
        self.line_height = 26
        self.text_height = 26
        self.padding = 0
        self.min_width = 120

        # Text items for signal names
        self.signal_texts = {}

        # Default position (top-right area)
        self.default_x = 60
        self.default_y = 60
        self.setPos(self.default_x, self.default_y)

        # Initially hidden
        self.setVisible(False)

    def update_legend(self):
        """
        Updates the legend content based on currently visible signals.
        Automatically resizes the legend and repositions text items.
        """
        # Clear existing text items
        self._clear_text_items()

        # Get active signals
        active_signals = self._get_active_signals()

        if not active_signals:
            self.setVisible(False)
            return

        # Create new text items and calculate dimensions
        max_width = self._create_signal_texts(active_signals)
        legend_height = len(active_signals) * self.line_height + 2 * self.padding

        # Resize legend background
        self.setRect(QRectF(0, 0, max_width, legend_height))

        # Show the legend
        self.setVisible(True)

    def _clear_text_items(self):
        """Remove all existing text items and backgrounds from the scene."""
        for items in self.signal_texts.values():
            if items['text'].scene():
                items['text'].scene().removeItem(items['text'])
            if items['background'].scene():
                items['background'].scene().removeItem(items['background'])
        self.signal_texts.clear()

    def _get_active_signals(self):
        """
        Get list of currently visible signals with their colors.

        Returns:
            list: Tuples of (signal_name, color_string)
        """
        active_signals = []
        for signal_name, curve in self.viewer.curves.items():
            if curve.isVisible():
                color = self.viewer.signal_styles[signal_name][1]
                active_signals.append((signal_name, color))
        return active_signals

    def _create_signal_texts(self, active_signals):
        """
        Create text items for each active signal with uniform width and center alignment.
        All legend entries will have the same width based on the longest signal name,
        with shorter names centered within their allocated space.

        Args:
            active_signals (list): List of (signal_name, color) tuples

        Returns:
            int: Uniform width used for all legend entries
        """
        y_offset = self.padding

        # First pass: calculate maximum width needed
        max_text_width = 0
        for signal_name, color in active_signals:
            text_width = len(signal_name) * 8
            max_text_width = max(max_text_width, text_width)

        uniform_width = max_text_width + 40  # Add padding for color box and spacing

        for signal_name, color in active_signals:
            # Create background rectangle positioned relative to legend
            bg_rect = pg.QtWidgets.QGraphicsRectItem(
                self.pos().x(), self.pos().y() + y_offset, uniform_width, self.text_height
            )
            bg_rect.setBrush(pg.functions.mkBrush(color))
            bg_rect.setPen(pg.mkPen(None))

            # Add background to scene
            self.viewer.plot_widget.scene().addItem(bg_rect)

            # Create text item
            text_item = pg.TextItem(
                text=signal_name,
                color='black',
                anchor=(0.5, 0.5)
            )
            text_item.setFont(QFont("Arial", 11, QFont.Weight.Bold))

            # Position text at center of background
            text_item.setPos(
                self.pos().x() + uniform_width / 2,
                self.pos().y() + y_offset + self.text_height / 2
            )

            self.viewer.plot_widget.scene().addItem(text_item)
            self.signal_texts[signal_name] = {
                'text': text_item,
                'background': bg_rect
            }

            y_offset += self.line_height

        return uniform_width

    def _truncate_name(self, signal_name, max_length=20):
        """
        Truncate signal name if it's too long.

        Args:
            signal_name (str): Original signal name
            max_length (int): Maximum allowed length

        Returns:
            str: Truncated name with ellipsis if needed
        """
        return signal_name

    def _position_text_item(self, text_item, y_offset):
        """
        Position a text item relative to the legend.

        Args:
            text_item: The text item to position
            y_offset (int): Vertical offset from legend top
        """
        text_item.setPos(
            self.pos().x() + self.padding,
            self.pos().y() + y_offset + self.line_height // 2
        )

    def paint(self, painter, option, widget=None):
        """
        Custom paint method to draw the legend background.
        Color backgrounds are now handled by TextItem fill property.

        Args:
            painter: QPainter instance
            option: Style options
            widget: Widget being painted
        """
        # Draw only the background rectangle - no color indicators needed
        #super().paint(painter, option, widget)
        pass

    def _draw_color_indicators(self, painter):
        """
        Draw colored rectangles next to each signal name.

        Args:
            painter: QPainter instance
        """
        y_offset = self.padding

        for signal_name, curve in self.viewer.curves.items():
            if curve.isVisible() and signal_name in self.signal_texts:
                color = self.viewer.signal_styles[signal_name][1]

                # Create color brush
                color_brush = QBrush()
                color_brush.setStyle(Qt.BrushStyle.SolidPattern)
                color_brush.setColor(pg.functions.mkColor(color))

                # Draw colored rectangle
                color_rect = QRectF(
                    self.padding,
                    y_offset + (self.line_height - self.color_box_height) // 2,
                    self.color_box_width,
                    self.color_box_height
                )
                painter.fillRect(color_rect, color_brush)

                # Draw border around color box
                painter.setPen(QPen(QColor(128, 128, 128), 1))
                painter.drawRect(color_rect)

                y_offset += self.line_height

    def mousePressEvent(self, event):
        """
        Handle mouse press events for selection and dragging.

        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSelected(True)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events during dragging.
        Updates text positions to follow the legend.

        Args:
            event: Mouse event
        """
        super().mouseMoveEvent(event)
        self._update_text_positions()

    def itemChange(self, change, value):
        """
        Handle item changes, particularly position changes.

        Args:
            change: Type of change
            value: New value

        Returns:
            The processed value
        """
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Constrain legend to plot area
            new_pos = value
            plot_rect = self.viewer.plot_widget.plotItem.vb.sceneBoundingRect()
            legend_rect = self.boundingRect()

            # Keep legend within plot bounds
            if new_pos.x() < 0:
                new_pos.setX(0)
            elif new_pos.x() + legend_rect.width() > plot_rect.width():
                new_pos.setX(plot_rect.width() - legend_rect.width())

            if new_pos.y() < 0:
                new_pos.setY(0)
            elif new_pos.y() + legend_rect.height() > plot_rect.height():
                new_pos.setY(plot_rect.height() - legend_rect.height())

            return new_pos

        return super().itemChange(change, value)

    def _update_text_positions(self):
        """Update positions of all text items and backgrounds to follow the legend."""
        if not self.signal_texts:
            return

        # Recalculate uniform width
        max_text_width = 0
        for signal_name in self.signal_texts.keys():
            text_width = len(signal_name) * 8
            max_text_width = max(max_text_width, text_width)
        uniform_width = max_text_width + 40

        y_offset = self.padding
        for items in self.signal_texts.values():
            # Update background position
            items['background'].setPos(self.pos().x(), self.pos().y())
            items['background'].setRect(0, y_offset, uniform_width, self.text_height)

            # Update text position
            items['text'].setPos(
                self.pos().x() + uniform_width / 2,
                self.pos().y() + y_offset + self.text_height / 2
            )
            y_offset += self.line_height

    def reset_position(self):
        """Reset legend to its default position."""
        self.setPos(self.default_x, self.default_y)
        self._update_text_positions()

    def set_visible(self, visible):
        """
        Show or hide the legend and all its text items and backgrounds.

        Args:
            visible (bool): Whether to show the legend
        """
        self.setVisible(visible)
        for items in self.signal_texts.values():
            items['text'].setVisible(visible)
            items['background'].setVisible(visible)

    def _position_text_item_centered(self, text_item, y_offset, width):
        """
        Position a text item centered within the given width.

        Args:
            text_item: The text item to position
            y_offset (int): Vertical offset from legend top
            width (int): Total width for centering
        """
        text_item.setPos(
            self.pos().x() + width / 2,
            self.pos().y() + y_offset + self.line_height // 2
        )