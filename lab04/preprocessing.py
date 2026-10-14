import html
import re
from collections.abc import Iterable

class TextPreprocessor:

    _HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _WHITESPACE_RE = re.compile(r"\s+")
    _TOKEN_WITH_APOSTROPHE_RE = re.compile(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?"
    )
    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

    def __init__(self, *, lowercase=True, remove_html=True, keep_apostrophes=True):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.keep_apostrophes = keep_apostrophes


    def clean_text(self, text):
        if text is None:
            return ""

        normalized = html.unescape(str(text))
        if self.remove_html:
            normalized = self._HTML_BREAK_RE.sub(" ", normalized)
            normalized = self._HTML_TAG_RE.sub(" ", normalized)

        if self.lowercase:
            normalized = normalized.lower()

        return self._WHITESPACE_RE.sub(" ", normalized).strip()


    def tokenize(self, text):
        cleaned = self.clean_text(text)
        token_pattern = (
            self._TOKEN_WITH_APOSTROPHE_RE 
            if self.keep_apostrophes
            else self._TOKEN_RE
        )
        return token_pattern.findall(cleaned)


    def transform(self, texts):
        return [self.clean_text(text) for text in texts]


    def __call__(self, text):
        return self.clean_text(text)
    