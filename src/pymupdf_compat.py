"""PyMuPDF 导入兼容层。

PyMuPDF >= 1.24 起，包名从 `fitz` 改为 `pymupdf`，
继续 `import fitz` 会打 deprecation 警告（未来版本会删掉）。
这里统一处理：优先 `pymupdf`，退回 `fitz`，并给出版本信息。
"""

from __future__ import annotations

from typing import Any

fitz: Any = None
IMPORT_ERROR: str = ""

try:                                    # PyMuPDF >= 1.24（推荐）
    import pymupdf as fitz  # type: ignore
except ImportError:
    try:                                # 老版本
        import fitz  # type: ignore
    except ImportError as exc:          # 完全没装
        IMPORT_ERROR = str(exc)


def is_available() -> bool:
    return fitz is not None


def require() -> Any:
    """拿不到就抛一个带安装提示的错误。"""
    if fitz is None:
        raise ImportError(
            "需要 PyMuPDF：pip install PyMuPDF\n"
            f"（原始错误：{IMPORT_ERROR}）"
        )
    return fitz


def version() -> str:
    if fitz is None:
        return "not installed"
    v = getattr(fitz, "__version__", None)
    if v:
        return str(v)
    try:
        return fitz.__doc__.strip().splitlines()[0]
    except Exception:
        return "unknown"
