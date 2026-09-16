"""Parses help-messages/messages.txt into per-mode, per-pane, per-language
help dialog copy for the "?" buttons next to each pane's (and the header's)
title - see server/main.py's /help-messages route and static/app.js's
help-button handling.

File format (see that file itself for the real content):

    #CATEGORY:digits
    #PANE:draw
    #LANGUAGE:english
    Draw a digit between 0 and 9 in the white box...
    #LANGUAGE:swedish
    Rita en siffra mellan 0 och 9 i den vita rutan...
    #PANE:configure
    ...
    #CATEGORY:drawings
    ...

A #CATEGORY/#PANE/#LANGUAGE line starts a new section; every other line is
appended (newline-joined) to the current language's text, so a message may
span multiple lines. "menu" is a real category with no panes yet (nothing to
explain on the mode-picker screen) - that's expected, not a parse error.
"""

from server.config import HELP_MESSAGES_PATH

# Every pane that actually has a "?" button in the UI (see static/index.html)
# for "digits"/"drawings" - "title" is the header's own title, not a .pane.
# "menu" intentionally has no panes yet (see module docstring).
_REQUIRED_PANES = {"title", "draw", "configure", "classification", "AIstatus"}
_REQUIRED_LANGUAGES = {"english", "swedish"}


def _parse_help_messages(path: str) -> dict[str, dict[str, dict[str, str]]]:
    categories: dict[str, dict[str, dict[str, str]]] = {}
    current_category: str | None = None
    current_pane: str | None = None
    current_language: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_category is not None and current_pane is not None and current_language is not None:
            text = "\n".join(buffer).strip()
            categories[current_category].setdefault(current_pane, {})[current_language] = text
        buffer.clear()

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        if line.startswith("#CATEGORY:"):
            flush()
            current_category = line.split(":", 1)[1].strip()
            categories.setdefault(current_category, {})
            current_pane = None
            current_language = None
        elif line.startswith("#PANE:"):
            flush()
            current_pane = line.split(":", 1)[1].strip()
            current_language = None
        elif line.startswith("#LANGUAGE:"):
            flush()
            current_language = line.split(":", 1)[1].strip()
        elif current_language is not None:
            buffer.append(line)
        # else: blank/stray lines before any #LANGUAGE section (e.g. the
        # gap after "#CATEGORY:menu") carry no text - ignored.
    flush()

    for category in ("digits", "drawings"):
        panes = categories.get(category, {})
        missing_panes = _REQUIRED_PANES - panes.keys()
        if missing_panes:
            raise ValueError(f"{path}: category {category!r} is missing pane(s): {sorted(missing_panes)}")
        for pane, by_language in panes.items():
            missing_languages = _REQUIRED_LANGUAGES - by_language.keys()
            if missing_languages:
                raise ValueError(
                    f"{path}: category {category!r} pane {pane!r} is missing language(s): {sorted(missing_languages)}"
                )
            empty = [lang for lang in _REQUIRED_LANGUAGES if not by_language[lang]]
            if empty:
                raise ValueError(f"{path}: category {category!r} pane {pane!r} has empty text for: {sorted(empty)}")

    return categories


HELP_MESSAGES: dict[str, dict[str, dict[str, str]]] = _parse_help_messages(HELP_MESSAGES_PATH)
