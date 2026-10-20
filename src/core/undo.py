"""
Undo/Redo Command Architecture for GemGraph.
Derived from BridgeCLR command execution hierarchy (REA-CLR-BRIDGE-091).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.table import DataTable


class Command(ABC):
    """Abstract base command for undoable actions."""

    @abstractmethod
    def execute(self) -> None:
        """Executes the command."""
        pass

    @abstractmethod
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


class UndoStack:
    """Manages undo and redo stacks for a document or view."""

    def __init__(self, limit: int = 100) -> None:
        self.limit = limit
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    def push(self, command: Command) -> None:
        """Executes a command and pushes it onto the undo stack."""
        command.execute()
        self._undo_stack.append(command)
        if len(self._undo_stack) > self.limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        """Pops and reverses the most recent command."""
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return True

    def redo(self) -> bool:
        """Re-executes the most recently undone command."""
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
