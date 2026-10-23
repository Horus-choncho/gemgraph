"""
Interactive Page Layout Presentation Canvas for GemGraph.
Implements GraphPad Prism-style presentation page layouts, interactive graph placeholders,
8-handle resize transformation, live high-DPI graph snapshots, rich text legend items,
and smooth zoom/pan controls.
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSizeF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
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
    QGraphicsTextItem,
    QGraphicsView,
    QMenu,
    QWidget,
)

from src.core.layout import PageLayout, Placeholder
from src.ui.dialogs import AssignGraphDialog, PageSetupDialog


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
        self._handle_start_rect: Optional[Tuple[float, float, float, float]] = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.scenePos()
            self._handle_start_rect = self.parent_placeholder.placeholder.rect
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
            old_rect = self._handle_start_rect
            new_rect = self.parent_placeholder.placeholder.rect
            if old_rect is not None and old_rect != new_rect:
                layout_view = getattr(self.parent_placeholder.canvas, "layout_view", None)
                if layout_view is not None and hasattr(layout_view, "push_move_resize_command"):
                    layout_view.push_move_resize_command(self.parent_placeholder.placeholder, old_rect, new_rect)
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
        self.page_width = float(page_width)
        self.page_height = float(page_height)
        self.document = document
        self.canvas = canvas
        self._syncing_geometry = False
        self._drag_start_rect: Optional[Tuple[float, float, float, float]] = None

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
        self._syncing_geometry = True
        try:
            rx, ry, rw, rh = self.placeholder.rect
            px = rx * self.page_width
            py = ry * self.page_height
            pw = rw * self.page_width
            ph = rh * self.page_height
            self.setRect(0, 0, pw, ph)
            self.setPos(px, py)
        finally:
            self._syncing_geometry = False

    def _sync_placeholder_from_geometry(self) -> None:
        if getattr(self, "_syncing_geometry", False):
            return
        if self.page_width <= 0 or self.page_height <= 0:
            return
        pos = self.pos()
        rect = self.rect()
        rw = rect.width() / self.page_width
        rh = rect.height() / self.page_height
        rx = pos.x() / self.page_width
        ry = pos.y() / self.page_height

        # Clamp normalized values between 0.0 and 1.0
        rw = max(0.02, min(1.0, rw))
        rh = max(0.02, min(1.0, rh))
        rx = max(0.0, min(1.0 - rw, rx))
        ry = max(0.0, min(1.0 - rh, ry))

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
            clamped_x = max(0.0, min(self.page_width - r.width(), new_pos.x()))
            clamped_y = max(0.0, min(self.page_height - r.height(), new_pos.y()))
            return QPointF(clamped_x, clamped_y)
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not getattr(self, "_syncing_geometry", False):
                self._sync_placeholder_from_geometry()
        return super().itemChange(change, value)

    def invalidate_cache(self) -> None:
        """Forces recreation of cached high-DPI graph snapshot."""
        self.cached_pixmap = None
        self.update()

    def update_snapshot(self) -> None:
        """Forces an immediate snapshot refresh with the latest data from upstream table."""
        if self.placeholder.graph_id and self.document is not None:
            graph_sheet = self.document.get_sheet(self.placeholder.graph_id)
            if graph_sheet:
                try:
                    from src.core.office_exporter import render_graph_to_png_bytes
                    r = self.rect()
                    w = max(600, int(r.width() * 2))
                    h = max(450, int(r.height() * 2))
                    png_data = render_graph_to_png_bytes(
                        graph_sheet,
                        document=self.document,
                        width=w,
                        height=h,
                    )
                    img = QImage.fromData(png_data)
                    if not img.isNull():
                        self.cached_pixmap = QPixmap.fromImage(img)
                        self.update()
                        return
                except Exception:
                    pass
        self.cached_pixmap = None
        self.update()

    def handle_resize(self, handle_index: int, scene_pos: QPointF, shift_pressed: bool = False) -> None:
        """Resizes the placeholder rect based on handle drag, with aspect ratio locking support."""
        pos = self.pos()
        r = self.rect()
        cur_w, cur_h = r.width(), r.height()
        aspect = cur_w / max(1.0, cur_h)

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

        self._syncing_geometry = True
        try:
            self.setRect(0, 0, new_w, new_h)
            self.setPos(new_x, new_y)
        finally:
            self._syncing_geometry = False

        self._update_handle_positions()
        self._sync_placeholder_from_geometry()
        self.invalidate_cache()
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_rect = self.placeholder.rect
            event.accept()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._sync_placeholder_from_geometry()
        old_rect = getattr(self, "_drag_start_rect", None)
        new_rect = self.placeholder.rect
        if old_rect is not None and old_rect != new_rect:
            layout_view = getattr(self.canvas, "layout_view", None)
            if layout_view is not None and hasattr(layout_view, "push_move_resize_command"):
                layout_view.push_move_resize_command(self.placeholder, old_rect, new_rect)
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def open_assign_dialog(self) -> None:
        """Opens AssignGraphDialog to assign, swap, or unassign active graphs."""
        dialog = AssignGraphDialog(
            self.document,
            current_graph_id=self.placeholder.graph_id,
            current_rotation=getattr(self.placeholder, "rotation", 0),
            current_scale=getattr(self.placeholder, "custom_scale", 1.0),
            parent=self.canvas,
        )
        if dialog.exec():
            graph_id, title = dialog.get_selected_graph()
            rotation = dialog.get_selected_rotation()
            scale = dialog.get_selected_scale()
            layout_view = getattr(self.canvas, "layout_view", None)
            if layout_view is not None and hasattr(layout_view, "push_assign_graph_command"):
                layout_view.push_assign_graph_command(self.placeholder, graph_id, title, rotation=rotation, scale=scale)
            else:
                if graph_id:
                    self.placeholder.assign_graph(graph_id, title)
                else:
                    self.placeholder.unassign_graph()
                self.placeholder.rotation = rotation
                self.placeholder.custom_scale = scale
                self.invalidate_cache()
                self.canvas.placeholderAssigned.emit(self.placeholder.slot_id)
                if hasattr(self.canvas, "layoutModifiedSignal"):
                    self.canvas.layoutModifiedSignal.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        """Opens AssignGraphDialog on mouse double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_assign_dialog()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def paint(self, painter: QPainter, option: Any, widget: Optional[QWidget] = None) -> None:
        r = self.rect()

        if self.placeholder.graph_id and self.document is not None:
            # Assigned state: render graph snapshot
            if self.cached_pixmap is None:
                self.update_snapshot()

            if self.cached_pixmap is not None:
                painter.save()
                painter.setClipRect(r)
                rot = getattr(self.placeholder, "rotation", 0)
                if rot:
                    painter.translate(r.center())
                    painter.rotate(rot)
                    painter.translate(-r.center())
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
                self._paint_empty_state(painter, r)
        else:
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


class TextResizeHandleItem(QGraphicsRectItem):
    """
    Interactive corner/edge resize grip for LayoutTextItem.
    """

    HANDLE_SIZE = 8.0

    def __init__(self, handle_index: int, parent_text_item: LayoutTextItem) -> None:
        super().__init__(
            -self.HANDLE_SIZE / 2.0,
            -self.HANDLE_SIZE / 2.0,
            self.HANDLE_SIZE,
            self.HANDLE_SIZE,
            parent_text_item,
        )
        self.handle_index = handle_index
        self.parent_text_item = parent_text_item
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(QColor(0, 120, 212), 1.5))
        self.setZValue(20)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

        cursors = {
            0: Qt.CursorShape.SizeFDiagCursor,  # TL
            1: Qt.CursorShape.SizeBDiagCursor,  # TR
            2: Qt.CursorShape.SizeHorCursor,    # MR
            3: Qt.CursorShape.SizeBDiagCursor,  # BL
            4: Qt.CursorShape.SizeFDiagCursor,  # BR
        }
        self.setCursor(cursors.get(handle_index, Qt.CursorShape.ArrowCursor))
        self.drag_start_pos: Optional[QPointF] = None
        self._start_width: float = 0.0
        self._start_pos: Optional[QPointF] = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.scenePos()
            self._start_width = self.parent_text_item.boundingRect().width()
            self._start_pos = self.parent_text_item.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.drag_start_pos is not None:
            self.parent_text_item.handle_resize(
                self.handle_index,
                event.scenePos(),
                self.drag_start_pos,
                self._start_width,
                self._start_pos,
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = None
            self.parent_text_item.finish_resize()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class LayoutTextItem(QGraphicsTextItem):
    """
    Floating rich text box item on page layout canvas.
    Supports inline editing, bold/italic formatting, font family and size adjustment,
    interactive dragging, subtle dashed selection border, dynamic resize handles, and undo integration.
    """

    def __init__(
        self,
        annotation_data: Dict[str, Any],
        page_width: float,
        page_height: float,
        canvas: LayoutCanvas,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.annotation = annotation_data
        self.page_width = float(page_width)
        self.page_height = float(page_height)
        self.canvas = canvas
        self._syncing_geometry = False
        self._drag_start_pos: Optional[Tuple[float, float]] = None
        self._resize_old_state: Optional[Dict[str, Any]] = None

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setZValue(15)

        self._apply_annotation_properties()
        self._sync_geometry_from_annotation()

        # Create 5 transform handles (TL, TR, MR, BL, BR)
        self.handles: List[TextResizeHandleItem] = [
            TextResizeHandleItem(i, self) for i in range(5)
        ]
        self._update_handle_positions()
        self._update_handle_visibility()

        self.document().contentsChanged.connect(self._on_contents_changed)

    def _apply_annotation_properties(self) -> None:
        font_family = self.annotation.get("font_family", "Helvetica")
        font_size = int(self.annotation.get("font_size", 14))
        bold = bool(self.annotation.get("bold", False))
        italic = bool(self.annotation.get("italic", False))
        color = QColor(self.annotation.get("color", "#000000"))

        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        self.setFont(font)
        self.setDefaultTextColor(color)

        html = self.annotation.get("html")
        text = self.annotation.get("text", "")
        if html:
            self.setHtml(html)
        elif text:
            self.setPlainText(text)
        else:
            self.setPlainText("Text Annotation")

    def _sync_geometry_from_annotation(self) -> None:
        self._syncing_geometry = True
        try:
            ax = float(self.annotation.get("x", 0.1)) * self.page_width
            ay = float(self.annotation.get("y", 0.1)) * self.page_height
            self.setPos(ax, ay)
            w_norm = self.annotation.get("w", self.annotation.get("width"))
            if w_norm is not None and float(w_norm) > 0:
                self.setTextWidth(float(w_norm) * self.page_width)
            else:
                self.setTextWidth(-1)
        finally:
            self._syncing_geometry = False
        self._update_handle_positions()

    def _sync_annotation_from_geometry(self) -> None:
        if self._syncing_geometry or self.page_width <= 0 or self.page_height <= 0:
            return
        pos = self.pos()
        nx = max(0.0, min(1.0, pos.x() / self.page_width))
        ny = max(0.0, min(1.0, pos.y() / self.page_height))
        self.annotation["x"] = round(nx, 4)
        self.annotation["y"] = round(ny, 4)
        br = self.boundingRect()
        nw = max(0.01, min(1.0, br.width() / self.page_width))
        nh = max(0.01, min(1.0, br.height() / self.page_height))
        self.annotation["w"] = round(nw, 4)
        self.annotation["width"] = self.annotation["w"]
        self.annotation["h"] = round(nh, 4)
        self.annotation["height"] = self.annotation["h"]

    def _update_handle_positions(self) -> None:
        br = self.boundingRect()
        w, h = br.width(), br.height()
        positions = [
            (0.0, 0.0),       # 0: TL
            (w, 0.0),         # 1: TR
            (w, h / 2.0),     # 2: MR
            (0.0, h),         # 3: BL
            (w, h),           # 4: BR
        ]
        if hasattr(self, "handles"):
            for idx, (hx, hy) in enumerate(positions):
                if idx < len(self.handles):
                    self.handles[idx].setPos(hx, hy)

    def _update_handle_visibility(self, visible: Optional[bool] = None) -> None:
        if visible is None:
            visible = self.isSelected()
        if hasattr(self, "handles"):
            for h in self.handles:
                h.setVisible(visible)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._syncing_geometry:
                self._sync_annotation_from_geometry()
                if hasattr(self.canvas, "layoutModifiedSignal"):
                    self.canvas.layoutModifiedSignal.emit()
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self._update_handle_visibility(bool(value))
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        if self.textInteractionFlags() == Qt.TextInteractionFlag.NoTextInteraction:
            if hasattr(event, "button") and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start_pos = (float(self.annotation.get("x", 0.1)), float(self.annotation.get("y", 0.1)))
        try:
            super().mousePressEvent(event)
        except (TypeError, RuntimeError):
            pass

    def mouseReleaseEvent(self, event) -> None:
        try:
            super().mouseReleaseEvent(event)
        except (TypeError, RuntimeError):
            pass
        if self.textInteractionFlags() == Qt.TextInteractionFlag.NoTextInteraction:
            self._sync_annotation_from_geometry()
            old_pos = getattr(self, "_drag_start_pos", None)
            new_pos = (float(self.annotation.get("x", 0.1)), float(self.annotation.get("y", 0.1)))
            if old_pos is not None and (abs(old_pos[0] - new_pos[0]) > 1e-4 or abs(old_pos[1] - new_pos[1]) > 1e-4):
                layout_view = getattr(self.canvas, "layout_view", None)
                if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
                    old_data = {"x": old_pos[0], "y": old_pos[1]}
                    new_data = {"x": new_pos[0], "y": new_pos[1]}
                    layout_view.push_edit_text_command(self.annotation, old_data, new_data)
            self._drag_start_pos = None
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        if hasattr(event, "button") and event.button() == Qt.MouseButton.LeftButton:
            self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.setCursor(Qt.CursorShape.IBeamCursor)
        try:
            super().mouseDoubleClickEvent(event)
        except (TypeError, RuntimeError):
            pass

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.clearFocus()
            self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            cursor = self.textCursor()
            cursor.clearSelection()
            self.setTextCursor(cursor)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:
        self._initial_edit_data = {
            "text": self.toPlainText(),
            "html": self.toHtml(),
        }
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().focusOutEvent(event)
        init_data = getattr(self, "_initial_edit_data", None)
        if init_data is not None:
            cur_text = self.toPlainText()
            cur_html = self.toHtml()
            if cur_text != init_data["text"] or cur_html != init_data["html"]:
                layout_view = getattr(self.canvas, "layout_view", None)
                if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
                    layout_view.push_edit_text_command(
                        self.annotation,
                        init_data,
                        {"text": cur_text, "html": cur_html},
                    )

    def handle_resize(
        self,
        handle_index: int,
        scene_pos: QPointF,
        drag_start: QPointF,
        start_width: float,
        start_pos: Optional[QPointF],
    ) -> None:
        """Adjusts QTextDocument text width dynamically based on handle drag, re-wrapping text."""
        if getattr(self, "_resize_old_state", None) is None:
            self._resize_old_state = {
                "w": self.annotation.get("w"),
                "width": self.annotation.get("width"),
                "h": self.annotation.get("h"),
                "height": self.annotation.get("height"),
                "x": self.annotation.get("x"),
                "y": self.annotation.get("y"),
            }
        delta_x = scene_pos.x() - drag_start.x()

        if handle_index in (1, 2, 4):  # TR, MR, BR
            new_width = max(30.0, start_width + delta_x)
            self.setTextWidth(new_width)
        elif handle_index in (0, 3):  # TL, BL
            new_width = max(30.0, start_width - delta_x)
            if start_pos is not None:
                new_x = start_pos.x() + (start_width - new_width)
                self.setPos(new_x, start_pos.y())
            self.setTextWidth(new_width)

        self._sync_annotation_from_geometry()
        self._update_handle_positions()
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def finish_resize(self) -> None:
        old_state = getattr(self, "_resize_old_state", None)
        self._resize_old_state = None
        self._sync_annotation_from_geometry()
        self._update_handle_positions()
        if old_state is not None:
            new_state = {
                "w": self.annotation.get("w"),
                "width": self.annotation.get("width"),
                "h": self.annotation.get("h"),
                "height": self.annotation.get("height"),
                "x": self.annotation.get("x"),
                "y": self.annotation.get("y"),
            }
            if old_state != new_state:
                layout_view = getattr(self.canvas, "layout_view", None)
                if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
                    layout_view.push_edit_text_command(self.annotation, old_state, new_state)
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def _on_contents_changed(self) -> None:
        self.annotation["text"] = self.toPlainText()
        self.annotation["html"] = self.toHtml()
        self._sync_annotation_from_geometry()
        self._update_handle_positions()
        if hasattr(self.canvas, "layoutModifiedSignal"):
            self.canvas.layoutModifiedSignal.emit()

    def set_font_size(self, size: int) -> None:
        old_size = self.annotation.get("font_size", 13)
        if old_size == size:
            return
        layout_view = getattr(self.canvas, "layout_view", None)
        if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
            layout_view.push_edit_text_command(self.annotation, {"font_size": old_size}, {"font_size": size})
        else:
            self.annotation["font_size"] = size
            f = self.font()
            f.setPointSize(size)
            self.setFont(f)
            self.update()

    def set_font_bold(self, bold: bool) -> None:
        old_bold = self.annotation.get("bold", False)
        if old_bold == bold:
            return
        layout_view = getattr(self.canvas, "layout_view", None)
        if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
            layout_view.push_edit_text_command(self.annotation, {"bold": old_bold}, {"bold": bold})
        else:
            self.annotation["bold"] = bold
            f = self.font()
            f.setBold(bold)
            self.setFont(f)
            self.update()

    def set_font_italic(self, italic: bool) -> None:
        old_italic = self.annotation.get("italic", False)
        if old_italic == italic:
            return
        layout_view = getattr(self.canvas, "layout_view", None)
        if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
            layout_view.push_edit_text_command(self.annotation, {"italic": old_italic}, {"italic": italic})
        else:
            self.annotation["italic"] = italic
            f = self.font()
            f.setItalic(italic)
            self.setFont(f)
            self.update()

    def set_font_family(self, family: str) -> None:
        old_family = self.annotation.get("font_family", "Helvetica")
        if old_family == family:
            return
        layout_view = getattr(self.canvas, "layout_view", None)
        if layout_view is not None and hasattr(layout_view, "push_edit_text_command"):
            layout_view.push_edit_text_command(self.annotation, {"font_family": old_family}, {"font_family": family})
        else:
            self.annotation["font_family"] = family
            f = self.font()
            f.setFamily(family)
            self.setFont(f)
            self.update()

    def paint(self, painter: QPainter, option: Any, widget: Optional[QWidget] = None) -> None:
        super().paint(painter, option, widget)
        if self.isSelected():
            painter.save()
            br = self.boundingRect()
            pen = QPen(QColor(0, 120, 212), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(br)
            painter.restore()


class LayoutCanvas(QGraphicsView):
    """
    High-performance QGraphicsView presentation canvas rendering standardized page formats,
    multi-graph layouts, drag-to-align interactions, context menus, and smooth zoom/pan controls.
    """

    placeholderAssigned = pyqtSignal(int)
    layoutModifiedSignal = pyqtSignal()
    selectionChangedSignal = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.layout_view: Optional[Any] = parent if hasattr(parent, "undo_stack") else None
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.current_layout: Optional[PageLayout] = None
        self.current_document: Any = None
        self.page_item: Optional[LayoutPageItem] = None
        self.placeholder_items: List[GraphPlaceholderItem] = []
        self.text_items: List[LayoutTextItem] = []

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
        self.document = document
        self.rebuild_scene()

    def rebuild_scene(self) -> None:
        """Constructs physical page surface, interactive graph placeholders, and annotations."""
        self.scene.clear()
        self.placeholder_items.clear()
        self.text_items.clear()

        if self.current_layout is None:
            return

        page_w, page_h = self.current_layout.get_dimensions_pt()

        # Update physical page scene rect
        self.page_item = LayoutPageItem(page_w, page_h)
        self.scene.addItem(self.page_item)
        self.scene.setSceneRect(-40, -40, page_w + 80, page_h + 80)

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

        # Add annotations (text boxes, connector lines)
        for ann in getattr(self.current_layout, "annotations", []):
            ann_type = ann.get("type", "text")
            if ann_type == "text":
                text_item = LayoutTextItem(
                    annotation_data=ann,
                    page_width=page_w,
                    page_height=page_h,
                    canvas=self,
                )
                self.scene.addItem(text_item)
                self.text_items.append(text_item)
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
            item.update_snapshot()
        self.scene.update()

    def refresh_hot_placeholders(self) -> None:
        """Refreshes placeholders bound to unfrozen (hot) graphs."""
        doc = getattr(self, "current_document", None) or getattr(self, "document", None)
        for item in self.placeholder_items:
            gid = getattr(item.placeholder, "graph_id", None)
            if gid and doc:
                graph = doc.get_sheet(gid)
                if graph and not getattr(graph, "is_frozen", False):
                    item.invalidate_cache()
                    item.update_snapshot()
        self.scene.update()

    def fit_page_to_window(self) -> None:
        """Fits the entire physical page cleanly within the current view window without compounding zoom."""
        if self.current_layout is not None and self.page_item is not None:
            page_w, page_h = self.current_layout.get_dimensions_pt()
            page_rect = QRectF(0, 0, page_w, page_h)
            self.resetTransform()
            self.fitInView(page_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def contextMenuEvent(self, event) -> None:
        """Displays layout canvas right-click context menu."""
        if self.current_layout is None:
            return

        menu = QMenu(self)
        act_add_placeholder = menu.addAction("➕ Add Graph Placeholder")
        act_add_text = menu.addAction("📝 Add Text Box / Legend")
        menu.addSeparator()
        act_page_setup = menu.addAction("📄 Page Setup...")

        chosen = menu.exec(event.globalPos())
        if chosen == act_add_placeholder:
            self.action_add_placeholder_at(event.pos())
        elif chosen == act_add_text:
            self.action_add_text_at(event.pos())
        elif chosen == act_page_setup:
            self.action_page_setup()

    def action_add_placeholder_at(self, viewport_pos: QPoint) -> None:
        """Adds a new Graph Placeholder at the cursor location matching existing dimensions."""
        if self.current_layout is None:
            return
        page_w, page_h = self.current_layout.get_dimensions_pt()
        scene_pos = self.mapToScene(viewport_pos)

        # Match size of existing placeholders on page or default to 40% x 40%
        if self.current_layout.placeholders:
            first_p = self.current_layout.placeholders[0]
            def_w = first_p.rect[2]
            def_h = first_p.rect[3]
        else:
            def_w = 0.40
            def_h = 0.40

        norm_x = scene_pos.x() / page_w
        norm_y = scene_pos.y() / page_h
        norm_x = max(0.02, min(1.0 - def_w - 0.02, norm_x))
        norm_y = max(0.02, min(1.0 - def_h - 0.02, norm_y))

        existing_slots = [
            p.slot_id for p in self.current_layout.placeholders
            if isinstance(p.slot_id, int)
        ]
        new_slot_id = max(existing_slots, default=0) + 1

        new_p = Placeholder(
            slot_id=new_slot_id,
            rect=(round(norm_x, 4), round(norm_y, 4), round(def_w, 4), round(def_h, 4)),
        )

        layout_view = getattr(self, "layout_view", None)
        if layout_view is not None and hasattr(layout_view, "push_add_placeholder_command"):
            layout_view.push_add_placeholder_command(new_p)
        else:
            self.current_layout.add_placeholder(new_p)
            self.rebuild_scene()
            self.layoutModifiedSignal.emit()

    def action_add_text_at(self, viewport_pos: Optional[QPoint] = None) -> None:
        """Adds a floating rich text box at the cursor position or default top-left."""
        if self.current_layout is None:
            return
        page_w, page_h = self.current_layout.get_dimensions_pt()
        if viewport_pos is not None:
            scene_pos = self.mapToScene(viewport_pos)
            nx = max(0.04, min(0.75, scene_pos.x() / page_w))
            ny = max(0.04, min(0.85, scene_pos.y() / page_h))
        else:
            nx, ny = 0.08, 0.08

        new_ann: Dict[str, Any] = {
            "id": f"text_{uuid.uuid4().hex[:8]}",
            "type": "text",
            "text": "Figure 1. Panel Description",
            "html": "<b>Figure 1.</b> Panel Description",
            "x": round(nx, 4),
            "y": round(ny, 4),
            "font_size": 13,
            "font_family": "Helvetica",
            "bold": False,
            "italic": False,
            "color": "#000000",
        }

        layout_view = getattr(self, "layout_view", None)
        if layout_view is not None and hasattr(layout_view, "push_add_text_command"):
            layout_view.push_add_text_command(new_ann)
        else:
            if not hasattr(self.current_layout, "annotations") or self.current_layout.annotations is None:
                self.current_layout.annotations = []
            self.current_layout.annotations.append(new_ann)
            self.rebuild_scene()
            self.layoutModifiedSignal.emit()

    def action_page_setup(self) -> None:
        """Opens Page Setup dialog allowing user to switch page format and orientation."""
        if self.current_layout is None:
            return
        dialog = PageSetupDialog(
            current_format=self.current_layout.page_format,
            current_orientation=self.current_layout.orientation,
            parent=self,
        )
        if dialog.exec():
            fmt, orient = dialog.get_settings()
            self.current_layout.page_format = fmt
            self.current_layout.orientation = orient
            layout_view = getattr(self, "layout_view", None)
            if layout_view is not None:
                layout_view.set_layout(self.current_layout, self.current_document)
            else:
                self.rebuild_scene()
            self.layoutModifiedSignal.emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Dispatches double-click events directly to any targeted placeholder slot."""
        scene_pos = self.mapToScene(event.pos())
        items = self.scene.items(scene_pos)
        target = None
        for it in items:
            cur = it
            while cur is not None and not isinstance(cur, GraphPlaceholderItem):
                cur = cur.parentItem()
            if isinstance(cur, GraphPlaceholderItem):
                target = cur
                break
        if target is not None:
            target.open_assign_dialog()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handles canvas key shortcuts: Delete key deletes selected placeholders or text items."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # Check selected items
            deleted = False
            for it in list(self.placeholder_items):
                if it.isSelected() and self.current_layout:
                    layout_view = getattr(self, "layout_view", None)
                    if layout_view is not None and hasattr(layout_view, "push_delete_placeholder_command"):
                        layout_view.push_delete_placeholder_command(it.placeholder)
                    else:
                        self.current_layout.remove_placeholder(it.placeholder.slot_id)
                        self.rebuild_scene()
                    deleted = True
                    break
            if not deleted:
                for t in list(self.text_items):
                    if t.isSelected() and self.current_layout:
                        layout_view = getattr(self, "layout_view", None)
                        if layout_view is not None and hasattr(layout_view, "push_delete_text_command"):
                            layout_view.push_delete_text_command(t.annotation)
                        else:
                            self.current_layout.annotations = [a for a in self.current_layout.annotations if a.get("id") != t.annotation.get("id")]
                            self.rebuild_scene()
                        deleted = True
                        break
            if deleted:
                self.layoutModifiedSignal.emit()
                event.accept()
                return

        super().keyPressEvent(event)
