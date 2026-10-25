"""
Rich Text Formatting Toolbar & Auto-List Editor for GemGraph.
Provides a reusable RichTextEditor widget wrapping a QTextEdit with a compact
formatting toolbar and keyboard auto-formatting (- + Space auto-bullet, Tab/Shift+Tab indent).
"""

from __future__ import annotations
from typing import Optional, Dict, List
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QTextCharFormat, QTextListFormat, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QComboBox,
    QTextEdit,
)


class RichTextEdit(QTextEdit):
    """
    Custom QTextEdit with keyboard auto-formatting:
    - Auto-bullet: typing '- ' at the start of a line converts to a bulleted list.
    - Indentation: Tab increases block indentation, Shift+Tab decreases it.
    """

    def keyPressEvent(self, event) -> None:
        key = event.key()
        modifiers = event.modifiers()

        # Keyboard Auto-Bullet: '- ' + Space at beginning of line
        if key == Qt.Key.Key_Space and not (modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
            cursor = self.textCursor()
            block = cursor.block()
            text = block.text()
            pos_in_block = cursor.positionInBlock()
            text_before = text[:pos_in_block]

            # If user has typed '-' (with optional leading spaces) right before pressing space
            if text_before.strip() == "-":
                cursor.beginEditBlock()
                list_fmt = QTextListFormat()
                list_fmt.setStyle(QTextListFormat.Style.ListDisc)
                cursor.createList(list_fmt)
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                while cursor.positionInBlock() < len(cursor.block().text()) and cursor.block().text()[cursor.positionInBlock()] in (' ', '-'):
                    cursor.deleteChar()
                cursor.endEditBlock()
                self.setTextCursor(cursor)
                event.accept()
                return

        # Keyboard Indentation: Tab increases indent, Shift+Tab / Backtab decreases
        if key == Qt.Key.Key_Tab and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            cursor = self.textCursor()
            cursor.beginEditBlock()
            bf = cursor.blockFormat()
            bf.setIndent(bf.indent() + 1)
            cursor.setBlockFormat(bf)
            cursor.endEditBlock()
            self.setTextCursor(cursor)
            event.accept()
            return

        if key in (Qt.Key.Key_Backtab, Qt.Key.Key_Tab) and (modifiers & Qt.KeyboardModifier.ShiftModifier or key == Qt.Key.Key_Backtab):
            cursor = self.textCursor()
            cursor.beginEditBlock()
            bf = cursor.blockFormat()
            bf.setIndent(max(0, bf.indent() - 1))
            cursor.setBlockFormat(bf)
            cursor.endEditBlock()
            self.setTextCursor(cursor)
            event.accept()
            return

        super().keyPressEvent(event)


class RichTextEditor(QWidget):
    """
    Reusable rich text editor widget combining RichTextEdit with a compact formatting toolbar:
    - Typeface selector (Sans-Serif, Serif, Monospace)
    - Font size selector (Small, Medium, Large)
    - Style toggles: Bold (Ctrl+B), Italic (Ctrl+I), Underline (Ctrl+U), Bullet List
    """

    textChanged = pyqtSignal()

    FONT_FAMILIES: Dict[str, List[str]] = {
        "Sans-Serif": ["Liberation Sans", "DejaVu Sans", "sans-serif"],
        "Serif": ["Liberation Serif", "DejaVu Serif", "serif"],
        "Monospace": ["Liberation Mono", "DejaVu Sans Mono", "monospace"],
    }

    FONT_SIZES: Dict[str, float] = {
        "Small (10pt)": 10.0,
        "Medium (12pt - Default)": 12.0,
        "Large (16pt)": 16.0,
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._updating_toolbar = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar container
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)

        # 1. Typeface selector
        self.combo_font = QComboBox()
        self.combo_font.addItems(["Sans-Serif", "Serif", "Monospace"])
        self.combo_font.setCurrentIndex(0)
        self.combo_font.currentIndexChanged.connect(self._on_font_family_changed)
        toolbar.addWidget(self.combo_font)

        # 2. Font size selector
        self.combo_size = QComboBox()
        self.combo_size.addItems(["Small (10pt)", "Medium (12pt - Default)", "Large (16pt)"])
        self.combo_size.setCurrentIndex(1)  # Default to Medium
        self.combo_size.currentIndexChanged.connect(self._on_font_size_changed)
        toolbar.addWidget(self.combo_size)

        # 3. Bold toggle
        self.btn_bold = QToolButton()
        self.btn_bold.setText("B")
        self.btn_bold.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.btn_bold.setCheckable(True)
        self.btn_bold.setToolTip("Bold (Ctrl+B)")
        self.btn_bold.clicked.connect(self._on_bold_toggled)
        toolbar.addWidget(self.btn_bold)

        # 4. Italic toggle
        self.btn_italic = QToolButton()
        self.btn_italic.setText("I")
        f_italic = QFont("Arial", 10)
        f_italic.setItalic(True)
        self.btn_italic.setFont(f_italic)
        self.btn_italic.setCheckable(True)
        self.btn_italic.setToolTip("Italic (Ctrl+I)")
        self.btn_italic.clicked.connect(self._on_italic_toggled)
        toolbar.addWidget(self.btn_italic)

        # 5. Underline toggle
        self.btn_underline = QToolButton()
        self.btn_underline.setText("U")
        f_under = QFont("Arial", 10)
        f_under.setUnderline(True)
        self.btn_underline.setFont(f_under)
        self.btn_underline.setCheckable(True)
        self.btn_underline.setToolTip("Underline (Ctrl+U)")
        self.btn_underline.clicked.connect(self._on_underline_toggled)
        toolbar.addWidget(self.btn_underline)

        # 6. Bullet List toggle
        self.btn_bullet = QToolButton()
        self.btn_bullet.setText("• List")
        self.btn_bullet.setCheckable(True)
        self.btn_bullet.setToolTip("Bullet List")
        self.btn_bullet.clicked.connect(self._on_bullet_toggled)
        toolbar.addWidget(self.btn_bullet)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Editor area
        self.editor = RichTextEdit()
        self.editor.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #d0d0d0;
                padding: 6px;
                font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
                font-size: 12pt;
            }
        """)
        # Initialize default formatting
        default_fmt = QTextCharFormat()
        default_fmt.setFontFamily("Liberation Sans")
        if hasattr(default_fmt, "setFontFamilies"):
            default_fmt.setFontFamilies(["Liberation Sans", "DejaVu Sans", "sans-serif"])
        default_fmt.setFontPointSize(12.0)
        self.editor.setCurrentCharFormat(default_fmt)

        self.editor.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.editor.textChanged.connect(self.textChanged)
        layout.addWidget(self.editor)

        # Keyboard shortcuts for style toggles
        sc_bold = QShortcut(QKeySequence("Ctrl+B"), self.editor)
        sc_bold.activated.connect(lambda: self.btn_bold.click())
        sc_italic = QShortcut(QKeySequence("Ctrl+I"), self.editor)
        sc_italic.activated.connect(lambda: self.btn_italic.click())
        sc_underline = QShortcut(QKeySequence("Ctrl+U"), self.editor)
        sc_underline.activated.connect(lambda: self.btn_underline.click())

    def _merge_format_on_selection(self, fmt: QTextCharFormat) -> None:
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        cursor.mergeCharFormat(fmt)
        self.editor.mergeCurrentCharFormat(fmt)

    def _on_font_family_changed(self) -> None:
        if self._updating_toolbar:
            return
        name = self.combo_font.currentText()
        families = self.FONT_FAMILIES.get(name, ["sans-serif"])
        fmt = QTextCharFormat()
        fmt.setFontFamily(families[0])
        if hasattr(fmt, "setFontFamilies"):
            fmt.setFontFamilies(families)
        self._merge_format_on_selection(fmt)

    def _on_font_size_changed(self) -> None:
        if self._updating_toolbar:
            return
        name = self.combo_size.currentText()
        size_val = self.FONT_SIZES.get(name, 12.0)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(size_val)
        self._merge_format_on_selection(fmt)

    def _on_bold_toggled(self) -> None:
        if self._updating_toolbar:
            return
        fmt = QTextCharFormat()
        is_bold = self.btn_bold.isChecked()
        fmt.setFontWeight(QFont.Weight.Bold if is_bold else QFont.Weight.Normal)
        self._merge_format_on_selection(fmt)

    def _on_italic_toggled(self) -> None:
        if self._updating_toolbar:
            return
        fmt = QTextCharFormat()
        fmt.setFontItalic(self.btn_italic.isChecked())
        self._merge_format_on_selection(fmt)

    def _on_underline_toggled(self) -> None:
        if self._updating_toolbar:
            return
        fmt = QTextCharFormat()
        fmt.setFontUnderline(self.btn_underline.isChecked())
        self._merge_format_on_selection(fmt)

    def _on_bullet_toggled(self) -> None:
        if self._updating_toolbar:
            return
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        current_list = cursor.currentList()
        if current_list is not None:
            # Remove list format
            bf = cursor.blockFormat()
            bf.setIndent(0)
            bf.setObjectIndex(-1)
            cursor.setBlockFormat(bf)
        else:
            list_fmt = QTextListFormat()
            list_fmt.setStyle(QTextListFormat.Style.ListDisc)
            cursor.createList(list_fmt)
        cursor.endEditBlock()
        self.editor.setFocus()

    def _on_cursor_position_changed(self) -> None:
        if self._updating_toolbar:
            return
        self._updating_toolbar = True
        try:
            fmt = self.editor.currentCharFormat()
            bold = fmt.fontWeight() == QFont.Weight.Bold or (hasattr(fmt, "font") and fmt.font().bold())
            self.btn_bold.setChecked(bool(bold))
            self.btn_italic.setChecked(bool(fmt.fontItalic()))
            self.btn_underline.setChecked(bool(fmt.fontUnderline()))

            cursor = self.editor.textCursor()
            self.btn_bullet.setChecked(cursor.currentList() is not None)

            # Update size selector
            sz = fmt.fontPointSize()
            if sz > 0:
                if sz <= 10.5:
                    self.combo_size.setCurrentIndex(0)
                elif sz >= 15.0:
                    self.combo_size.setCurrentIndex(2)
                else:
                    self.combo_size.setCurrentIndex(1)

            # Update font selector
            family = ""
            if hasattr(fmt, "font"):
                family = fmt.font().family().lower()
            elif hasattr(fmt, "fontFamilies") and fmt.fontFamilies():
                family = str(fmt.fontFamilies()).lower()
            if "mono" in family or "courier" in family:
                self.combo_font.setCurrentIndex(2)
            elif "serif" in family and "sans" not in family:
                self.combo_font.setCurrentIndex(1)
            elif family:
                self.combo_font.setCurrentIndex(0)
        finally:
            self._updating_toolbar = False

    def toHtml(self) -> str:
        return self.editor.toHtml()

    def setHtml(self, html: str) -> None:
        self.editor.setHtml(html)

    def toPlainText(self) -> str:
        return self.editor.toPlainText()

    def setPlainText(self, text: str) -> None:
        self.editor.setPlainText(text)

    def blockSignals(self, b: bool) -> bool:
        res = super().blockSignals(b)
        self.editor.blockSignals(b)
        return res
