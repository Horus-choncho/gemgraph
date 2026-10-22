"""
Undo/Redo Command Architecture for GemGraph.
Derived from BridgeCLR command execution hierarchy (REA-CLR-BRIDGE-091).
"""

from PyQt6.QtGui import QUndoStack, QUndoCommand
from src.core.table import DataTable


class Command(QUndoCommand):
    """Abstract base command for undoable actions, inheriting from QUndoCommand."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)

    def redo(self) -> None:
        self.execute()

    def execute(self) -> None:
        """Executes the command."""
        pass

    def undo(self) -> None:
        """Reverses the command."""
        pass


class CellEditCommand(Command):
    """Command modifying a single cell's numerical value."""

    def __init__(
        self,
        table: DataTable,
        row: int,
        col_type: str,
        group_idx: int,
        subcol_idx: int,
        old_val: float,
        new_val: float,
    ) -> None:
        super().__init__("Edit Cell")
        self.table = table
        self.row = row
        self.col_type = col_type
        self.group_idx = group_idx
        self.subcol_idx = subcol_idx
        self.old_val = old_val
        self.new_val = new_val

    def execute(self) -> None:
        self.table.set_cell_value(
            self.row, self.col_type, self.group_idx, self.subcol_idx, self.new_val
        )

    def undo(self) -> None:
        self.table.set_cell_value(
            self.row, self.col_type, self.group_idx, self.subcol_idx, self.old_val
        )


class CellExcludeCommand(Command):
    """Command toggling cell exclusion mask."""

    def __init__(
        self,
        table: DataTable,
        row: int,
        col_type: str,
        group_idx: int,
        subcol_idx: int,
        old_excluded: bool,
        new_excluded: bool,
    ) -> None:
        super().__init__("Toggle Exclude Cell")
        self.table = table
        self.row = row
        self.col_type = col_type
        self.group_idx = group_idx
        self.subcol_idx = subcol_idx
        self.old_excluded = old_excluded
        self.new_excluded = new_excluded

    def execute(self) -> None:
        self.table.set_excluded(
            self.row, self.col_type, self.group_idx, self.subcol_idx, self.new_excluded
        )

    def undo(self) -> None:
        self.table.set_excluded(
            self.row, self.col_type, self.group_idx, self.subcol_idx, self.old_excluded
        )


class VariableCellEditCommand(Command):
    """Command modifying a single cell in a Multiple Variables table."""

    def __init__(
        self,
        table: DataTable,
        row: int,
        col: int,
        old_val: Any,
        new_val: Any,
    ) -> None:
        super().__init__("Edit Variable Cell")
        self.table = table
        self.row = row
        self.col = col
        self.old_val = old_val
        self.new_val = new_val

    def execute(self) -> None:
        self.table.set_variable_value(self.row, self.col, self.new_val)

    def undo(self) -> None:
        self.table.set_variable_value(self.row, self.col, self.old_val)


class UndoStack(QUndoStack):
    """Manages undo and redo stacks for a document or view, wrapping Qt QUndoStack."""

    def __init__(self, parent=None, limit: int = 100) -> None:
        super().__init__(parent)
        self.setUndoLimit(limit)

    def push(self, command: QUndoCommand) -> None:
        """Executes a command and pushes it onto the undo stack."""
        super().push(command)

    def undo(self) -> bool:
        """Pops and reverses the most recent command."""
        if not self.canUndo():
            return False
        super().undo()
        return True

    def redo(self) -> bool:
        """Re-executes the most recently undone command."""
        if not self.canRedo():
            return False
        super().redo()
        return True

    @property
    def can_undo(self) -> bool:
        return self.canUndo()

    @property
    def can_redo(self) -> bool:
        return self.canRedo()

    def clear(self) -> None:
        super().clear()

    def __bool__(self) -> bool:
        return True
