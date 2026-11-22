# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Launch the local CLI with Mac native-library paths scoped to this process.

LightGBM's macOS wheel requires libomp. Reuse PyTorch's bundled LLVM OpenMP
runtime when present, without changing Homebrew or any global installation.
"""

import importlib.util
import os
import sys
from pathlib import Path

environment = os.environ.copy()
if sys.platform == "darwin":
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        torch_dir = Path(spec.origin).parent
        # macOS wheels have shipped the OpenMP runtime in both layouts.
        for library in (torch_dir / "lib", torch_dir / ".dylibs"):
            if (library / "libomp.dylib").is_file():
                existing = environment.get("DYLD_LIBRARY_PATH", "")
                environment["DYLD_LIBRARY_PATH"] = str(library) + (":" + existing if existing else "")
                break
arguments = sys.argv[1:]
if arguments[:1] == ["--python"]:
    arguments = arguments[1:]
else:
    arguments = ["-m", "gamematch", *arguments]
os.execve(sys.executable, [sys.executable, *arguments], environment)
