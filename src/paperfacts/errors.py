"""The project's unified exception base classes.

The CLI only catches :class:`PaperFactsError` (plus ``FileNotFoundError``), turning them into a
single red line plus exit code 1; every other exception (programming errors, a third-party
library's ValueError…) propagates with its traceback intact so debugging clues aren't lost.
"""


class PaperFactsError(RuntimeError):
    """Any expected failure that should be reported to the user as a single red line."""


class ConfigError(PaperFactsError):
    """Configuration is missing or invalid (for example, no LLM key)."""
