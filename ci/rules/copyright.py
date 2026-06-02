#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import re
from datetime import date

import libcst
from fixit import LintRule


class CopyrightRule(LintRule):
    """Ensures that the copyright notice is present at the top of the file."""

    MESSAGE = "Missing or incorrect copyright notice."

    _COPYRIGHT_RE = re.compile(r"^#\s*Copyright\s*©\s*\d{4}\s+Emmi AI GmbH\.\s+All rights reserved\.\s*$")

    @classmethod
    def _has_copyright_notice(cls, node: libcst.Module) -> bool:
        for h in node.header:
            if isinstance(h, libcst.EmptyLine) and h.comment:
                if cls._COPYRIGHT_RE.match(h.comment.value.strip()):
                    return True
        return False

    def visit_Module(self, node: libcst.Module) -> None:
        expected_notice = f"#  Copyright © {date.today().year} Emmi AI GmbH. All rights reserved."

        # If there's already a notice (even with a different year), do nothing.
        if self._has_copyright_notice(node):
            return

        # Otherwise, insert the expected notice at the top.
        if len(node.header) == 0:
            new_node = node.with_changes(
                header=[libcst.EmptyLine(comment=libcst.Comment(expected_notice)), libcst.Newline()]
            )
        else:
            new_node = node.with_changes(
                header=[libcst.EmptyLine(comment=libcst.Comment(expected_notice)), libcst.Newline()] + list(node.header)
            )

        self.report(node, self.MESSAGE, replacement=new_node)
