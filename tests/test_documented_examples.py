# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Run the introductory Python examples as a reader would copy them."""

import re
import unittest
from pathlib import Path


class DocumentationExamplesTests(unittest.TestCase):
    def test_data_free_python_guides(self):
        docs = Path(__file__).resolve().parents[1] / "docs"
        for guide in ("api.md", "api-zh.md", "getting-started.md", "start-zh.md", "matching-api.md"):
            with self.subTest(guide=guide):
                path = docs / guide
                snippets = re.findall(
                    r"^```python\n(.*?)^```$", path.read_text(encoding="utf-8"), re.M | re.S
                )
                self.assertTrue(snippets, f"Missing executable Python example in {guide}")
                for snippet in snippets:
                    exec(compile(snippet, str(path), "exec"), {})
