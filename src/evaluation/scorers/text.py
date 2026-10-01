"""Historical text normalization shared by WER implementations."""
import re

_ZH_PUNCT_MAP = {
    "，": ", ",
    "。": ".",
    "：": ":",
    "；": ";",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ",",
    "—": "-",
    "…": "...",
    "·": ".",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
}


def _is_cjk_char(ch: str) -> bool:
    try:
        return "\u4e00" <= ch <= "\u9fff"
    except Exception:
        return False


def _to_simplified_zh(text: str) -> str:
    """Convert Chinese text to Simplified Chinese when possible.
    Tries OpenCC first, then hanziconv, otherwise returns the input unchanged.
    """
    s = str(text or "")
    try:
        # opencc-python-reimplemented
        from opencc import OpenCC  # type: ignore

        try:
            cc = OpenCC("t2s")
        except Exception:
            # Fallback conversion map commonly available
            cc = OpenCC("hk2s")
        return cc.convert(s)
    except Exception:
        pass

    try:
        # hanziconv
        from hanziconv import HanziConv  # type: ignore

        return HanziConv.toSimplified(s)
    except Exception:
        return s


def _contains_cjk(text: str) -> bool:
    try:
        return any(_is_cjk_char(ch) for ch in str(text or ""))
    except Exception:
        return False


def _normalize_zh_text(text: str) -> str:
    s = str(text or "")
    # Always normalize to Simplified to reduce script variance
    try:
        s = _to_simplified_zh(s)
    except Exception:
        pass
    # Normalize Chinese punctuation to English equivalents
    for zh, en in _ZH_PUNCT_MAP.items():
        s = s.replace(zh, en)
    # Remove ALL punctuation for WER calculation to avoid spurious errors
    # Keep only letters, numbers, Chinese characters, and spaces
    s = re.sub(r'[^\w\s\u4e00-\u9fff]', '', s)
    # Collapse whitespace, remove surrounding spaces
    s = " ".join(s.split())
    # Remove spaces between CJK to get stable char sequence
    # e.g., "摩 洛" -> "摩洛"
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    return s.strip()
