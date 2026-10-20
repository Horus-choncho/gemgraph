"""
No-Sudo-Strict-Local Security Gate for GemGraph.
Enforces pure user-space execution, no root/sudo privileges,
and no direct execution of untrusted foreign Windows PE binaries.
"""

import os
import sys
import unittest
import ast
import pathlib


class TestSecurityPolicy(unittest.TestCase):
    """
    Validates operating boundaries and codebase compliance with
    'No-Sudo-Strict-Local' security directives.
    """

    def test_non_root_execution(self) -> None:
        """Process must run strictly as unprivileged user (non-root)."""
        if hasattr(os, "getuid"):
            uid = os.getuid()
            self.assertNotEqual(
                uid,
                0,
                f"FATAL SECURITY VIOLATION: Execution detected as root (UID {uid}). GemGraph must run in local user space.",
            )

    def test_no_sudo_in_codebase(self) -> None:
        """Scans codebase to guarantee no hardcoded 'sudo' invocations exist."""
        project_root = pathlib.Path(__file__).resolve().parent.parent
        src_path = project_root / "src"

        for py_file in src_path.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
                for line_no, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    self.assertNotIn(
                        "sudo ",
                        stripped.lower(),
                        f"Security Violation: Found 'sudo' in {py_file}:{line_no}: {stripped}",
                    )

    def test_no_foreign_binary_execution(self) -> None:
        """
        Scans codebase to verify no wine or direct subprocess execution of
        untrusted Windows PE binaries (.exe, .dll) exists.
        """
        project_root = pathlib.Path(__file__).resolve().parent.parent
        src_path = project_root / "src"

        for py_file in src_path.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            for node in ast.walk(tree):
                # Check for subprocess or wine calls
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr

                    if func_name in ("Popen", "run", "call", "check_call", "check_output"):
                        for subnode in ast.walk(node):
                            if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str):
                                val_lower = subnode.value.lower()
                                self.assertNotIn("wine", val_lower, f"Found 'wine' in {py_file} at line {node.lineno}")
                                self.assertNotIn(".exe", val_lower, f"Found '.exe' in {py_file} at line {node.lineno}")
                                self.assertNotIn(".dll", val_lower, f"Found '.dll' in {py_file} at line {node.lineno}")
                    # Check ctypes loading
                    if func_name in ("CDLL", "WinDLL", "OleDLL", "LoadLibrary"):
                        self.fail(f"Banned foreign library dynamic loading in {py_file} at line {node.lineno}")

    def test_no_arbitrary_shell_execution(self) -> None:
        """
        Ensures src/ code does not make unsanitized os.system calls.
        """
        project_root = pathlib.Path(__file__).resolve().parent.parent
        src_path = project_root / "src"

        for py_file in src_path.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == "system" and getattr(node.func.value, "id", None) == "os":
                            self.fail(f"Banned os.system call found in {py_file} at line {node.lineno}")


if __name__ == "__main__":
    unittest.main()
