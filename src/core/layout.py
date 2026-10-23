"""
Page Layout Data Model and Manipulation Engine for GemGraph.
Supports GraphPad Prism-style presentation page layouts, multi-graph assembly,
grid templates, alignment/equalization math, and high-DPI offscreen rendering.
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QBuffer, QIODevice, QPointF, QRectF, QSizeF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPageLayout, QPageSize, QPdfWriter, QPen

from src.core.sheet import BaseSheet, SheetType


# Physical page sizes in points (72 points per inch)
PAGE_DIMENSIONS_PT: Dict[str, Tuple[float, float]] = {
    "A4": (595.28, 841.89),       # 210mm x 297mm
    "Letter": (612.0, 792.0),     # 8.5" x 11.0"
    "Slide 16:9": (960.0, 540.0), # 13.33" x 7.5"
    "Slide 4:3": (720.0, 540.0),  # 10.0" x 7.5"
}

PAGE_DIMENSIONS_INCH: Dict[str, Tuple[float, float]] = {
    "A4": (8.27, 11.69),
    "Letter": (8.5, 11.0),
    "Slide 16:9": (13.33, 7.5),
    "Slide 4:3": (10.0, 7.5),
}


class Placeholder:
    """
    Normalized graph placeholder within a page layout.
    Stores normalized (0.0 - 1.0) page coordinates [x, y, w, h] and graph link.
    """

    def __init__(
        self,
        slot_id: Any = 0,
        rect: Tuple[float, float, float, float] = (0.1, 0.1, 0.4, 0.4),
        graph_id: Optional[str] = None,
        linked_graph_title: Optional[str] = None,
        custom_scale: float = 1.0,
        locked_aspect: bool = True,
        rotation: int = 0,
    ) -> None:
        self.slot_id = slot_id
        rx = max(0.0, min(1.0, float(rect[0])))
        ry = max(0.0, min(1.0, float(rect[1])))
        rw = max(0.01, min(1.0, float(rect[2])))
        rh = max(0.01, min(1.0, float(rect[3])))
        self.rect = (rx, ry, rw, rh)
        self.graph_id = graph_id
        self.linked_graph_title = linked_graph_title
        self.custom_scale = float(custom_scale)
        self.locked_aspect = bool(locked_aspect)
        self.rotation = int(rotation)

    @property
    def aspect_ratio_locked(self) -> bool:
        return self.locked_aspect

    @aspect_ratio_locked.setter
    def aspect_ratio_locked(self, val: bool) -> None:
        self.locked_aspect = bool(val)

    def assign_graph(self, graph_id: str, title: Optional[str] = None) -> None:
        """Assigns or replaces the linked graph for this placeholder slot."""
        self.graph_id = graph_id
        self.linked_graph_title = title

    def unassign_graph(self) -> None:
        """Sever graph link, resetting placeholder back to empty state."""
        self.graph_id = None
        self.linked_graph_title = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "rect": list(self.rect),
            "graph_id": self.graph_id,
            "linked_graph_title": self.linked_graph_title,
            "custom_scale": self.custom_scale,
            "locked_aspect": self.locked_aspect,
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Placeholder:
        raw_rect = data.get("rect", [0.1, 0.1, 0.4, 0.4])
        rect_tuple = (float(raw_rect[0]), float(raw_rect[1]), float(raw_rect[2]), float(raw_rect[3]))
        return cls(
            slot_id=data.get("slot_id", 0),
            rect=rect_tuple,
            graph_id=data.get("graph_id"),
            linked_graph_title=data.get("linked_graph_title"),
            custom_scale=float(data.get("custom_scale", 1.0)),
            locked_aspect=bool(data.get("locked_aspect", True)),
            rotation=int(data.get("rotation", 0)),
        )


class PageLayout(BaseSheet):
    """
    Presentation page layout holding multi-graph placeholders, standardized page bounds,
    and freeform annotations for publication and slide export.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "Layout 1",
        page_format: str = "A4",
        orientation: str = "Portrait",
        placeholders: Optional[List[Placeholder]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        layout_id = id if id is not None else f"layout_{uuid.uuid4().hex[:8]}"
        super().__init__(sheet_id=layout_id, title=name, sheet_type=SheetType.LAYOUT)
        self.page_format = page_format
        self.orientation = orientation
        self.placeholders: List[Placeholder] = placeholders if placeholders is not None else []
        self.annotations: List[Dict[str, Any]] = annotations if annotations is not None else []

    @property
    def id(self) -> str:
        return self.sheet_id

    @id.setter
    def id(self, val: str) -> None:
        self.sheet_id = val

    @property
    def name(self) -> str:
        return self.title

    @name.setter
    def name(self, val: str) -> None:
        self.title = val

    def get_dimensions_pt(self) -> Tuple[float, float]:
        """Returns physical dimensions in points (width, height) accounting for orientation."""
        base_w, base_h = PAGE_DIMENSIONS_PT.get(self.page_format, (595.28, 841.89))
        if self.orientation.lower() == "landscape":
            return max(base_w, base_h), min(base_w, base_h)
        else:
            return min(base_w, base_h), max(base_w, base_h)

    def get_dimensions_in(self) -> Tuple[float, float]:
        """Returns physical dimensions in inches (width, height) accounting for orientation."""
        base_w, base_h = PAGE_DIMENSIONS_INCH.get(self.page_format, (8.27, 11.69))
        if self.orientation.lower() == "landscape":
            return max(base_w, base_h), min(base_w, base_h)
        else:
            return min(base_w, base_h), max(base_w, base_h)

    def get_dimensions_px(self, dpi: int = 300) -> Tuple[int, int]:
        """Returns physical dimensions in pixels at target DPI."""
        in_w, in_h = self.get_dimensions_in()
        return int(round(in_w * dpi)), int(round(in_h * dpi))

    def add_placeholder(self, placeholder: Placeholder) -> None:
        self.placeholders.append(placeholder)

    def remove_placeholder(self, slot_id: int) -> bool:
        initial_len = len(self.placeholders)
        self.placeholders = [p for p in self.placeholders if p.slot_id != slot_id]
        return len(self.placeholders) < initial_len

    def get_placeholder(self, slot_id: int) -> Optional[Placeholder]:
        for p in self.placeholders:
            if p.slot_id == slot_id:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "id": self.id,
            "name": self.name,
            "page_format": self.page_format,
            "orientation": self.orientation,
            "placeholders": [p.to_dict() for p in self.placeholders],
            "annotations": [dict(a) for a in self.annotations],
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageLayout:
        layout_id = data.get("id") or data.get("sheet_id") or f"layout_{uuid.uuid4().hex[:8]}"
        name = data.get("name") or data.get("title") or "Layout"
        page_format = data.get("page_format", "A4")
        orientation = data.get("orientation", "Portrait")
        placeholders_raw = data.get("placeholders", [])
        placeholders = [Placeholder.from_dict(p) for p in placeholders_raw]
        annotations = [dict(a) for a in data.get("annotations", [])]

        layout = cls(
            id=layout_id,
            name=name,
            page_format=page_format,
            orientation=orientation,
            placeholders=placeholders,
            annotations=annotations,
        )
        layout.is_frozen = data.get("is_frozen", False)
        layout.is_orphaned = data.get("is_orphaned", False)
        layout.is_pinned = data.get("is_pinned", False)
        layout.parent_sheet_ids = list(data.get("parent_sheet_ids", []))
        layout.child_sheet_ids = list(data.get("child_sheet_ids", []))
        return layout


# =============================================================================
# Grid Templates & Presets
# =============================================================================

def create_grid_placeholders(
    rows: int,
    cols: int,
    margin_x: float = 0.08,
    margin_y: float = 0.08,
    spacing_x: float = 0.04,
    spacing_y: float = 0.04,
) -> List[Placeholder]:
    """
    Generates a regular grid of M x N placeholders with strict coordinate bounds within (0, 1).
    """
    rows = max(1, rows)
    cols = max(1, cols)
    avail_w = 1.0 - 2 * margin_x - (cols - 1) * spacing_x
    avail_h = 1.0 - 2 * margin_y - (rows - 1) * spacing_y
    w = max(0.05, avail_w / cols)
    h = max(0.05, avail_h / rows)

    placeholders: List[Placeholder] = []
    slot = 1
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (w + spacing_x)
            y = margin_y + r * (h + spacing_y)
            placeholders.append(
                Placeholder(
                    slot_id=slot,
                    rect=(round(x, 4), round(y, 4), round(w, 4), round(h, 4)),
                )
            )
            slot += 1
    return placeholders


def create_template_placeholders(
    preset_name: str,
    rows: int = 1,
    cols: int = 1,
) -> List[Placeholder]:
    """
    Factory function producing standard layout template presets:
    - "1 Graph"
    - "2 Horizontal"
    - "2 Vertical"
    - "3 (1 Top / 2 Bottom)"
    - "4 (2x2 Quadrants)"
    - "6 (2x3)"
    - "8 (2x4)"
    - "Custom Grid"
    """
    norm_name = preset_name.strip().lower()

    if "1" in norm_name and ("graph" in norm_name or norm_name == "1"):
        return [
            Placeholder(slot_id=1, rect=(0.10, 0.10, 0.80, 0.80))
        ]

    elif "2" in norm_name and "horizontal" in norm_name:
        return create_grid_placeholders(rows=1, cols=2, margin_x=0.06, margin_y=0.15, spacing_x=0.04)

    elif "2" in norm_name and "vertical" in norm_name:
        return create_grid_placeholders(rows=2, cols=1, margin_x=0.15, margin_y=0.06, spacing_y=0.04)

    elif "3" in norm_name:
        # 1 Top, 2 Bottom
        top_w = 0.60
        top_h = 0.40
        top_x = (1.0 - top_w) / 2.0
        top_y = 0.06

        bot_w = 0.40
        bot_h = 0.40
        bot_spacing = 0.06
        bot_y = 0.52
        bot_x1 = (1.0 - (2 * bot_w + bot_spacing)) / 2.0
        bot_x2 = bot_x1 + bot_w + bot_spacing

        return [
            Placeholder(slot_id=1, rect=(round(top_x, 4), round(top_y, 4), round(top_w, 4), round(top_h, 4))),
            Placeholder(slot_id=2, rect=(round(bot_x1, 4), round(bot_y, 4), round(bot_w, 4), round(bot_h, 4))),
            Placeholder(slot_id=3, rect=(round(bot_x2, 4), round(bot_y, 4), round(bot_w, 4), round(bot_h, 4))),
        ]

    elif "4" in norm_name:
        return create_grid_placeholders(rows=2, cols=2, margin_x=0.06, margin_y=0.06, spacing_x=0.04, spacing_y=0.04)

    elif "6" in norm_name:
        return create_grid_placeholders(rows=2, cols=3, margin_x=0.05, margin_y=0.06, spacing_x=0.03, spacing_y=0.04)

    elif "8" in norm_name:
        return create_grid_placeholders(rows=2, cols=4, margin_x=0.04, margin_y=0.06, spacing_x=0.025, spacing_y=0.04)

    else:
        # Custom Grid
        return create_grid_placeholders(rows=max(1, rows), cols=max(1, cols))


# =============================================================================
# Alignment & Equalization Routines
# =============================================================================

def align_placeholders(
    placeholders: List[Placeholder],
    alignment: str = "left",
    mode: Optional[str] = None,
) -> None:
    """
    Aligns a list of placeholders along a target reference axis.
    Alignments supported:
    - 'left': align to minimum X
    - 'center': align horizontal centers to mean center
    - 'right': align right edges to maximum right edge
    - 'top': align to minimum Y
    - 'middle': align vertical centers to mean center
    - 'bottom': align bottom edges to maximum bottom edge
    """
    if len(placeholders) < 2:
        return

    alignment = (mode if mode is not None else alignment).lower().strip()

    if alignment in ("left", "lefts"):
        min_x = min(p.rect[0] for p in placeholders)
        for p in placeholders:
            p.rect = (min_x, p.rect[1], p.rect[2], p.rect[3])

    elif alignment in ("center", "centers"):
        mean_cx = sum(p.rect[0] + p.rect[2] / 2.0 for p in placeholders) / len(placeholders)
        for p in placeholders:
            new_x = max(0.0, min(1.0 - p.rect[2], mean_cx - p.rect[2] / 2.0))
            p.rect = (round(new_x, 4), p.rect[1], p.rect[2], p.rect[3])

    elif alignment in ("right", "rights"):
        max_r = max(p.rect[0] + p.rect[2] for p in placeholders)
        for p in placeholders:
            new_x = max(0.0, min(1.0 - p.rect[2], max_r - p.rect[2]))
            p.rect = (round(new_x, 4), p.rect[1], p.rect[2], p.rect[3])

    elif alignment in ("top", "tops"):
        min_y = min(p.rect[1] for p in placeholders)
        for p in placeholders:
            p.rect = (p.rect[0], min_y, p.rect[2], p.rect[3])

    elif alignment in ("middle", "middles"):
        mean_cy = sum(p.rect[1] + p.rect[3] / 2.0 for p in placeholders) / len(placeholders)
        for p in placeholders:
            new_y = max(0.0, min(1.0 - p.rect[3], mean_cy - p.rect[3] / 2.0))
            p.rect = (p.rect[0], round(new_y, 4), p.rect[2], p.rect[3])

    elif alignment in ("bottom", "bottoms"):
        max_b = max(p.rect[1] + p.rect[3] for p in placeholders)
        for p in placeholders:
            new_y = max(0.0, min(1.0 - p.rect[3], max_b - p.rect[3]))
            p.rect = (p.rect[0], round(new_y, 4), p.rect[2], p.rect[3])


def equalize_placeholders(placeholders: List[Placeholder], mode: str) -> None:
    """
    Equalizes dimensions of placeholders using the first placeholder as reference:
    - 'width' / 'widths': sets all widths equal to reference
    - 'height' / 'heights': sets all heights equal to reference
    - 'both': sets both widths and heights equal to reference
    """
    if len(placeholders) < 2:
        return

    ref = placeholders[0]
    ref_w, ref_h = ref.rect[2], ref.rect[3]
    mode = mode.lower().strip()

    for p in placeholders[1:]:
        new_w = ref_w if mode in ("width", "widths", "both") else p.rect[2]
        new_h = ref_h if mode in ("height", "heights", "both") else p.rect[3]
        new_x = min(p.rect[0], max(0.0, 1.0 - new_w))
        new_y = min(p.rect[1], max(0.0, 1.0 - new_h))
        p.rect = (round(new_x, 4), round(new_y, 4), round(new_w, 4), round(new_h, 4))


# =============================================================================
# Offscreen Composite Layout Renderer
# =============================================================================

def render_layout_to_image(
    layout: PageLayout,
    document: Optional[Any] = None,
    dpi: int = 300,
    graph_canvas: Optional[Any] = None,
) -> QImage:
    """
    Renders a complete PageLayout composite offscreen to a publication-grade QImage
    at specified DPI (150, 300, 600 DPI) preserving physical page dimensions.
    """
    in_w, in_h = layout.get_dimensions_in()
    px_w = int(round(in_w * dpi))
    px_h = int(round(in_h * dpi))

    img = QImage(px_w, px_h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.white)

    painter = QPainter(img)
    try:
        if not painter.isActive():
            return img
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        from src.core.office_exporter import render_graph_to_png_bytes

        for p in layout.placeholders:
            rx, ry, rw, rh = p.rect
            target_rect = QRectF(rx * px_w, ry * px_h, rw * px_w, rh * px_h)

            rendered = False
            if p.graph_id and document is not None:
                graph_sheet = document.get_sheet(p.graph_id)
                if graph_sheet is not None:
                    try:
                        slot_w = max(100, int(target_rect.width()))
                        slot_h = max(100, int(target_rect.height()))
                        png_data = render_graph_to_png_bytes(
                            graph_sheet,
                            document=document,
                            graph_canvas=graph_canvas,
                            width=slot_w,
                            height=slot_h,
                        )
                        graph_img = QImage.fromData(png_data)
                        if not graph_img.isNull():
                            painter.save()
                            painter.setClipRect(target_rect)
                            rot = getattr(p, "rotation", 0)
                            if rot:
                                painter.translate(target_rect.center())
                                painter.rotate(rot)
                                painter.translate(-target_rect.center())
                            scale = max(0.2, min(5.0, getattr(p, "custom_scale", 1.0)))
                            if abs(scale - 1.0) > 1e-4:
                                cx = target_rect.center().x()
                                cy = target_rect.center().y()
                                sw = target_rect.width() * scale
                                sh = target_rect.height() * scale
                                draw_rect = QRectF(cx - sw / 2.0, cy - sh / 2.0, sw, sh)
                            else:
                                draw_rect = target_rect
                            painter.drawImage(draw_rect, graph_img)
                            painter.restore()
                            rendered = True
                    except Exception:
                        rendered = False

            if not rendered:
                # Draw empty state placeholder frame
                pen = QPen(QColor(180, 180, 180), max(1.5, dpi / 150.0), Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(248, 249, 250)))
                painter.drawRect(target_rect)

                painter.setPen(QColor(120, 120, 120))
                font = QFont("Helvetica", max(9, int(11 * dpi / 150.0)))
                painter.setFont(font)
                label_text = p.linked_graph_title or f"Slot {p.slot_id}: Double-click to assign graph"
                painter.drawText(target_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, label_text)

        # Draw annotations (Figure letter panels A, B, C, D, titles, connector lines)
        for ann in layout.annotations:
            ann_type = ann.get("type", "text")
            if ann_type == "text":
                text = ann.get("text", "")
                ax = float(ann.get("x", 0.0)) * px_w
                ay = float(ann.get("y", 0.0)) * px_h
                aw = float(ann.get("w", 0.2)) * px_w
                ah = float(ann.get("h", 0.05)) * px_h
                font_size = int(ann.get("font_size", 16) * dpi / 72.0)
                font = QFont(ann.get("font_family", "Helvetica"), font_size)
                font.setBold(bool(ann.get("bold", True)))
                painter.setFont(font)
                painter.setPen(QColor(ann.get("color", "#000000")))
                painter.drawText(QRectF(ax, ay, aw, ah), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            elif ann_type in ("line", "connector"):
                x1 = float(ann.get("x1", 0.0)) * px_w
                y1 = float(ann.get("y1", 0.0)) * px_h
                x2 = float(ann.get("x2", 1.0)) * px_w
                y2 = float(ann.get("y2", 1.0)) * px_h
                pen_color = QColor(ann.get("color", "#000000"))
                pen_w = max(1.0, float(ann.get("width", 1.5)) * dpi / 72.0)
                pen = QPen(pen_color, pen_w)
                if ann.get("style") == "dash":
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    finally:
        if painter.isActive():
            painter.end()

    return img


def render_layout_to_png_bytes(
    layout: PageLayout,
    document: Optional[Any] = None,
    dpi: int = 300,
    graph_canvas: Optional[Any] = None,
) -> bytes:
    """Exports PageLayout to high-DPI PNG bytes in memory."""
    img = render_layout_to_image(layout, document, dpi=dpi, graph_canvas=graph_canvas)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    data = bytes(buf.data())
    buf.close()
    return data


def render_layout_to_pdf(
    layout: PageLayout,
    document: Optional[Any] = None,
    file_path: str = "",
    dpi: int = 300,
    graph_canvas: Optional[Any] = None,
) -> str:
    """Exports PageLayout to a physical PDF document matching layout dimensions and 300 DPI."""
    writer = QPdfWriter(file_path)
    fmt_lower = layout.page_format.lower()
    pt_w, pt_h = layout.get_dimensions_pt()
    if "letter" in fmt_lower:
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
    elif "a4" in fmt_lower:
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    else:
        writer.setPageSize(QPageSize(QSizeF(pt_w, pt_h), QPageSize.Unit.Point))

    if layout.orientation.lower() == "landscape":
        writer.setPageOrientation(QPageLayout.Orientation.Landscape)
    else:
        writer.setPageOrientation(QPageLayout.Orientation.Portrait)

    writer.setResolution(dpi)

    img = render_layout_to_image(layout, document=document, dpi=dpi, graph_canvas=graph_canvas)

    painter = QPainter(writer)
    try:
        if painter.isActive():
            page_rect = QRectF(0, 0, writer.width(), writer.height())
            painter.drawImage(page_rect, img)
    finally:
        if painter.isActive():
            painter.end()

    return file_path


def render_layout_to_svg(
    layout: PageLayout,
    document: Optional[Any] = None,
    file_path: str = "",
    graph_canvas: Optional[Any] = None,
) -> str:
    """Exports PageLayout to scalable vector graphics (.svg)."""
    from PyQt6.QtSvg import QSvgGenerator
    pt_w, pt_h = layout.get_dimensions_pt()

    generator = QSvgGenerator()
    generator.setFileName(file_path)
    generator.setSize(QSizeF(pt_w, pt_h).toSize())
    generator.setViewBox(QRectF(0, 0, pt_w, pt_h))
    generator.setTitle(layout.name)
    generator.setDescription("Generated by GemGraph")

    painter = QPainter(generator)
    try:
        if not painter.isActive():
            return file_path

        from src.core.office_exporter import render_graph_to_png_bytes

        # White page background
        painter.fillRect(QRectF(0, 0, pt_w, pt_h), QColor(255, 255, 255))

        for p in layout.placeholders:
            rx, ry, rw, rh = p.rect
            target_rect = QRectF(rx * pt_w, ry * pt_h, rw * pt_w, rh * pt_h)

            rendered = False
            if p.graph_id and document is not None:
                graph_sheet = document.get_sheet(p.graph_id)
                if graph_sheet is not None:
                    try:
                        png_data = render_graph_to_png_bytes(
                            graph_sheet,
                            document=document,
                            graph_canvas=graph_canvas,
                            width=int(target_rect.width() * 2),
                            height=int(target_rect.height() * 2),
                        )
                        graph_img = QImage.fromData(png_data)
                        if not graph_img.isNull():
                            painter.save()
                            painter.setClipRect(target_rect)
                            scale = max(0.2, min(5.0, getattr(p, "custom_scale", 1.0)))
                            if abs(scale - 1.0) > 1e-4:
                                cx = target_rect.center().x()
                                cy = target_rect.center().y()
                                sw = target_rect.width() * scale
                                sh = target_rect.height() * scale
                                draw_rect = QRectF(cx - sw / 2.0, cy - sh / 2.0, sw, sh)
                            else:
                                draw_rect = target_rect
                            painter.drawImage(draw_rect, graph_img)
                            painter.restore()
                            rendered = True
                    except Exception:
                        rendered = False

            if not rendered:
                painter.setPen(QPen(QColor(180, 180, 180), 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(248, 249, 250)))
                painter.drawRect(target_rect)
                painter.setPen(QColor(120, 120, 120))
                painter.setFont(QFont("Helvetica", 10))
                label = p.linked_graph_title or f"Slot {p.slot_id}"
                painter.drawText(target_rect, Qt.AlignmentFlag.AlignCenter, label)

        for ann in layout.annotations:
            ann_type = ann.get("type", "text")
            if ann_type == "text":
                text = ann.get("text", "")
                ax = float(ann.get("x", 0.0)) * pt_w
                ay = float(ann.get("y", 0.0)) * pt_h
                aw = float(ann.get("w", 0.2)) * pt_w
                ah = float(ann.get("h", 0.05)) * pt_h
                font = QFont(ann.get("font_family", "Helvetica"), ann.get("font_size", 14))
                font.setBold(bool(ann.get("bold", True)))
                painter.setFont(font)
                painter.setPen(QColor(ann.get("color", "#000000")))
                painter.drawText(QRectF(ax, ay, aw, ah), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            elif ann_type in ("line", "connector"):
                x1 = float(ann.get("x1", 0.0)) * pt_w
                y1 = float(ann.get("y1", 0.0)) * pt_h
                x2 = float(ann.get("x2", 1.0)) * pt_w
                y2 = float(ann.get("y2", 1.0)) * pt_h
                pen_color = QColor(ann.get("color", "#000000"))
                pen_w = max(1.0, float(ann.get("width", 1.5)))
                pen = QPen(pen_color, pen_w)
                if ann.get("style") == "dash":
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    finally:
        if painter.isActive():
            painter.end()

    return file_path
