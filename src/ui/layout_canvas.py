"""
Interactive Page Layout Presentation Canvas for GemGraph.
Implements GraphPad Prism-style presentation page layouts, interactive graph placeholders,
8-handle resize transformation, live high-DPI graph snapshots, and smooth zoom/pan controls.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, QSizeF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from src.core.layout import PageLayout, Placeholder
from src.ui.dialogs import AssignGraphDialog


class LayoutPageItem(QGraphicsItem):
    """
    Renders the physical white page boundary with drop shadow on neutral workspace background.
    """

    def __init__(self, width: float, height: float, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.page_width = float(width)
        self.page_height = float(height)
        self.setZValue(-100)

    def boundingRect(self) -> QRectF:
        return QRectF(-20, -20, self.page_width + 40, self.page_height + 40)

    def paint(self, painter: QPainter, option: Any, widget: Optional[QWidget] = None) -> None:
        # Drop shadow layers
        for offset, alpha in [(8, 20), (5, 35), (2, 50)]:
            shadow_rect = QRectF(offset, offset, self.page_width, self.page_height)
            painter.fillRect(shadow_rect, QColor(0, 0, 0, alpha))

        # Physical white page surface
        page_rect = QRectF(0, 0, self.page_width, self.page_height)
        painter.fillRect(page_rect, QColor(255, 255, 255))
        painter.setPen(QPen(QColor(210, 210, 210), 1.0))
        painter.drawRect(page_rect)


class ResizeHandleItem(QGraphicsRectItem):
    """
    Interactive resize handle around selected placeholder items.
    """

    HANDLE_SIZE = 8.0

    def __init__(self, handle_index: int, parent_placeholder: GraphPlaceholderItem) -> None:
        super().__init__(
            -self.HANDLE_SIZE / 2.0,
            -self.HANDLE_SIZE / 2.0,
            self.HANDLE_SIZE,
            self.HANDLE_SIZE,
            parent_placeholder,
        )
        self.handle_index = handle_index
        self.parent_placeholder = parent_placeholder
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(QColor(0, 120, 212), 1.5))
        self.setZValue(10)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

        # Assign cursors
        cursors = {
            0: Qt.CursorShape.SizeFDiagCursor,  # Top-Left
            1: Qt.CursorShape.SizeVerCursor,    # Top-Center
            2: Qt.CursorShape.SizeBDiagCursor,  # Top-Right
            3: Qt.CursorShape.SizeHorCursor,    # Middle-Left
            4: Qt.CursorShape.SizeHorCursor,    # Middle-Right
            5: Qt.CursorShape.SizeBDiagCursor,  # Bottom-Left
            6: Qt.CursorShape.SizeVerCursor,    # Bottom-Center
            7: Qt.CursorShape.SizeFDiagCursor,  # Bottom-Right
        }
        self.setCursor(cursors.get(handle_index, Qt.CursorShape.ArrowCursor))
        self.drag_start_pos: Optional[QPointF] = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.scenePos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.drag_start_pos is not None:
            shift_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.parent_placeholder.handle_resize(self.handle_index, event.scenePos(), shift_pressed)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class GraphPlaceholderItem(QGraphicsRectItem):
    """
    Interactive graph placeholder displaying empty dashed state or live graph snapshot,
    supporting double-click assignment, 8 resize handles, and aspect-ratio preservation.
    """

    def __init__(
        self,
        placeholder: Placeholder,
        page_width: float,
        page_height: float,
        document: Any,
        canvas: LayoutCanvas,
    ) -> None:
        super().__init__()
        self.placeholder = placeholder
        self.page_width = page_width
        self.page_height = page_height
        self.document = document
        self.canvas = canvas

        self.cached_pixmap: Optional[QPixmap] = None

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Create 8 resize handles
        self.handles: List[ResizeHandleItem] = [
            ResizeHandleItem(i, self) for i in range(8)
        ]

        self._sync_geometry_from_placeholder()
        self._update_handle_positions()
        self._update_handle_visibility()

    def _sync_geometry_from_placeholder(self) -> None:
        rx, ry, rw, rh = self.placeholder.rect
        px = rx * self.page_width
        py = ry * self.page_height
        pw = rw * self.page_width
        ph = rh * self.page_height
        self.setPos(px, py)
        self.setRect(0, 0, pw, ph)

    def _sync_placeholder_from_geometry(self) -> None:
        pos = self.pos()
        rect = self.rect()
        rx = pos.x() / self.page_width
        ry = pos.y() / self.page_height
        rw = rect.width() / self.page_width
        rh = rect.height() / self.page_height

        # Clamp normalized values between 0.0 and 1.0
        rx = max(0.0, min(1.0 - rw, rx))
        ry = max(0.0, min(1.0 - rh, ry))
        rw = max(0.02, min(1.0, rw))
        rh = max(0.02, min(1.0, rh))

        self.placeholder.rect = (round(rx, 4), round(ry, 4), round(rw, 4), round(rh, 4))

    def _update_handle_positions(self) -> None:
        r = self.rect()
        w, h = r.width(), r.height()
        positions = [
            (0.0, 0.0),          # 0: TL
            (w / 2.0, 0.0),      # 1: TC
            (w, 0.0),            # 2: TR
            (0.0, h / 2.0),      # 3: ML
            (w, h / 2.0),        # 4: MR
            (0.0, h),            # 5: BL
            (w / 2.0, h),        # 6: BC
            (w, h),              # 7: BR
        ]
        for idx, (hx, hy) in enumerate(positions):
            self.handles[idx].setPos(hx, hy)

    def _update_handle_visibility(self) -> None:
        sel = self.isSelected()
        for h in self.handles:
            h.setVisible(sel)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            for h in self.handles:
                h.setVisible(bool(value))
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos = value
            r = self.rect()
            # Constrain movement within physical page boundaries
            clamped_x = max(0.0, min(self.page_width - r.width(), new_pos.x()))
            clamped_y = max(0.0, min(self.page_height - r.height(), new_pos.y()))
            return QPointF(clamped_x, clamped_y)
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._sync_placeholder_from_geometry()
        return super().itemChange(change, value)

    def invalidate_cache(self) -> None:
        """Forces recreation of cached high-DPI graph snapshot."""
        self.cached_pixmap = None
        self.update()

    def handle_resize(self, handle_index: int, scene_pos: QPointF, shift_pressed: bool = False) -> None:
        """Resizes the placeholder rect based on handle drag, with aspect ratio locking support."""
        pos = self.pos()
        r = self.rect()
        cur_w, cur_h = r.width(), r.height()
        aspect = cur_w / max(1.0, cur_h)

        dx = scene_pos.x() - (pos.x() + r.width())
        dy = scene_pos.y() - (pos.y() + r.height())

        right = pos.x() + cur_w
        bottom = pos.y() + cur_h

        new_x = pos.x()
        new_y = pos.y()
        new_w = cur_w
        new_h = cur_h

        # Compute unconstrained rectangle from handle
        if handle_index == 7:  # BR
            new_w = max(40.0, scene_pos.x() - pos.x())
            new_h = max(40.0, scene_pos.y() - pos.y())
            if self.placeholder.locked_aspect and not shift_pressed:
                new_h = new_w / aspect
        elif handle_index == 6:  # BC
            new_h = max(40.0, scene_pos.y() - pos.y())
            if self.placeholder.locked_aspect and not shift_pressed:
                new_w = new_h * aspect
                new_x = (pos.x() + cur_w / 2.0) - new_w / 2.0
        elif handle_index == 5:  # BL
            new_x = min(right - 40.0, scene_pos.x())
            new_w = right - new_x
            new_h = max(40.0, scene_pos.y() - pos.y())
            if self.placeholder.locked_aspect and not shift_pressed:
                new_h = new_w / aspect
        elif handle_index == 4:  # MR
            new_w = max(40.0, scene_pos.x() - pos.x())
            if self.placeholder.locked_aspect and not shift_pressed:
                new_h = new_w / aspect
                new_y = (pos.y() + cur_h / 2.0) - new_h / 2.0
        elif handle_index == 3:  # ML
            new_x = min(right - 40.0, scene_pos.x())
            new_w = right - new_x
            if self.placeholder.locked_aspect and not shift_pressed:
                new_h = new_w / aspect
                new_y = (pos.y() + cur_h / 2.0) - new_h / 2.0
        elif handle_index == 2:  # TR
            new_w = max(40.0, scene_pos.x() - pos.x())
            new_y = min(bottom - 40.0, scene_pos.y())
            new_h = bottom - new_y
            if self.placeholder.locked_aspect and not shift_pressed:
                new_h = new_w / aspect
                new_y = bottom - new_h
        elif handle_index == 1:  # TC
            new_y = min(bottom - 40.0, scene_pos.y())
            new_h = bottom - new_y
            if self.placeholder.locked_aspect and not shift_pressed:
                new_w = new_h * aspect
                new_x = (pos.x() + cur_w / 2.0) - new_w / 2.0
        elif handle_index == 0:  # TL
            new_x = min(right - 40.0, scene_pos.x())
            new_y = min(bottom - 40.0, scene_pos.y())
            new_w = right - new_x
            new_h = bottom - new_y
            if self.placeholder.locked_aspect and not shift_pressed:
                new_h = new_w / aspect
                new_y = bottom - new_h

        # Page boundary constraints
        new_x = max(0.0, min(self.page_width - 40.0, new_x))
        new_y = max(0.0, min(self.page_height - 40.0, new_y))
        new_w = min(self.page_width - new_x, new_w)
        new_h = min(self.page_height - new_y, new_h)

        self.setRect(0, 0, new_w, new_h)
        self.setPos(new_x, new_y)
        self._update_handle_positions()
        self._sync_placeholder_from_geometry()
        self.invalidate_cache()
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._sync_placeholder_from_geometry()
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        """Opens AssignGraphDialog to assign, swap, or unassign active graphs."""
        dialog = AssignGraphDialog(
            self.document,
            current_graph_id=self.placeholder.graph_id,
            parent=self.canvas,
        )
        if dialog.exec():
            graph_id, title = dialog.get_selected_graph()
            if graph_id:
                self.placeholder.assign_graph(graph_id, title)
            else:
                self.placeholder.unassign_graph()
            self.invalidate_cache()
            self.canvas.placeholderAssigned.emit(self.placeholder.slot_id)
        event.accept()

    def paint(self, painter: QPainter, option: Any, widget: Optional[QWidget] = None) -> None:
        r = self.rect()

        if self.placeholder.graph_id and self.document is not None:
            # Assigned state: render graph snapshot
            if self.cached_pixmap is None:
                graph_sheet = self.document.get_sheet(self.placeholder.graph_id)
                if graph_sheet:
                    try:
                        from src.core.office_exporter import render_graph_to_png_bytes
                        png_data = render_graph_to_png_bytes(
                            graph_sheet,
                            document=self.document,
                            width=max(100, int(r.width() * 2)),
                            height=max(100, int(r.height() * 2)),
                        )
                        img = QImage.fromData(png_data)
                        if not img.isNull():
                            self.cached_pixmap = QPixmap.fromImage(img)
                    except Exception:
                        self.cached_pixmap = None

            if self.cached_pixmap is not None:
                painter.save()
                painter.setClipRect(r)
                scale = max(0.1, min(5.0, getattr(self.placeholder, "custom_scale", 1.0)))
                if abs(scale - 1.0) > 1e-4:
                    cx = r.center().x()
                    cy = r.center().y()
                    sw = r.width() * scale
                    sh = r.height() * scale
                    draw_rect = QRectF(cx - sw / 2.0, cy - sh / 2.0, sw, sh)
                else:
                    draw_rect = r
                painter.drawPixmap(draw_rect.toRect(), self.cached_pixmap)
                painter.restore()
                painter.setPen(QPen(QColor(215, 215, 215), 1.0))
                painter.drawRect(r)
            else:
                # Fallback if render failed
                self._paint_empty_state(painter, r)
        else:
            # Empty state
            self._paint_empty_state(painter, r)

        # Highlight outline when selected
        if self.isSelected():
            painter.setPen(QPen(QColor(0, 120, 212), 2.0, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)

    def _paint_empty_state(self, painter: QPainter, r: QRectF) -> None:
        painter.fillRect(r, QColor(248, 249, 250))
        pen = QPen(QColor(170, 175, 180), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(r)

        painter.setPen(QColor(110, 115, 120))
        font = QFont("Helvetica", 10)
        font.setItalic(True)
        painter.setFont(font)
        text = f"Slot {self.placeholder.slot_id}\nDouble-click to assign graph"
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)


class LayoutCanvas(QGraphicsView):
    """
    High-performance QGraphicsView presentation canvas rendering standardized page formats,
    multi-graph layouts, drag-to-align interactions, and smooth zoom/pan controls.
    """

    placeholderAssigned = pyqtSignal(int)
    layoutModifiedSignal = pyqtSignal()
    selectionChangedSignal = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.current_layout: Optional[PageLayout] = None
        self.current_document: Any = None
        self.page_item: Optional[LayoutPageItem] = None
        self.placeholder_items: List[GraphPlaceholderItem] = []

        # View configuration
        self.setBackgroundBrush(QBrush(QColor(235, 237, 240)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

        self.panning = False
        self.pan_start = QPointF()

    def _on_scene_selection_changed(self) -> None:
        try:
            self.selectionChangedSignal.emit()
        except (RuntimeError, AttributeError):
            pass

    def set_layout(self, layout: PageLayout, document: Any) -> None:
        """Loads and visualizes a PageLayout document model."""
        self.current_layout = layout
        self.current_document = document
        self.rebuild_scene()

    def rebuild_scene(self) -> None:
        """Constructs physical page surface, interactive graph placeholders, and annotations."""
        self.scene.clear()
        self.placeholder_items.clear()

        if self.current_layout is None:
            return

        page_w, page_h = self.current_layout.get_dimensions_pt()

        # Add page surface
        self.page_item = LayoutPageItem(page_w, page_h)
        self.scene.addItem(self.page_item)
        self.scene.setSceneRect(-60, -60, page_w + 120, page_h + 120)

        # Add graph placeholders
        for p in self.current_layout.placeholders:
            item = GraphPlaceholderItem(
                placeholder=p,
                page_width=page_w,
                page_height=page_h,
                document=self.current_document,
                canvas=self,
            )
            self.scene.addItem(item)
            self.placeholder_items.append(item)

        # Add annotations (text panels A/B/C/D, connector lines)
        for ann in getattr(self.current_layout, "annotations", []):
            ann_type = ann.get("type", "text")
            if ann_type == "text":
                text = ann.get("text", "")
                ax = float(ann.get("x", 0.0)) * page_w
                ay = float(ann.get("y", 0.0)) * page_h
                text_item = self.scene.addText(text)
                font_size = int(ann.get("font_size", 14))
                font = QFont(ann.get("font_family", "Helvetica"), font_size)
                font.setBold(bool(ann.get("bold", True)))
                text_item.setFont(font)
                text_item.setDefaultTextColor(QColor(ann.get("color", "#000000")))
                text_item.setPos(ax, ay)
                text_item.setZValue(5)
            elif ann_type in ("line", "connector"):
                x1 = float(ann.get("x1", 0.0)) * page_w
                y1 = float(ann.get("y1", 0.0)) * page_h
                x2 = float(ann.get("x2", 1.0)) * page_w
                y2 = float(ann.get("y2", 1.0)) * page_h
                pen_color = QColor(ann.get("color", "#000000"))
                pen_w = max(1.0, float(ann.get("width", 1.5)))
                pen = QPen(pen_color, pen_w)
                if ann.get("style") == "dash":
                    pen.setStyle(Qt.PenStyle.DashLine)
                line_item = self.scene.addLine(x1, y1, x2, y2, pen)
                line_item.setZValue(5)

        self.fit_page_to_window()

    def get_selected_placeholders(self) -> List[Placeholder]:
        """Returns the list of Placeholder models currently selected on canvas."""
        selected: List[Placeholder] = []
        for item in self.placeholder_items:
            if item.isSelected():
                selected.append(item.placeholder)
        return selected

    def refresh_snapshots(self) -> None:
        """Refreshes all placeholder snapshots to synchronize with modified data or plots."""
        for item in self.placeholder_items:
            item.invalidate_cache()
        self.scene.update()

    def fit_page_to_window(self) -> None:
        """Fits the entire physical page cleanly within the current view window."""
        if self.page_item is not None:
            self.fitInView(self.page_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth zooming with mouse wheel."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.panning:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
