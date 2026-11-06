"""测试包：统一把 src/ 加入 sys.path，使 discover 与单模块单跑行为一致。

这样 `python -m unittest discover -s tests -t .` 与
`python -m unittest tests.test_async_embed` 都不需要预先设置 PYTHONPATH。
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
