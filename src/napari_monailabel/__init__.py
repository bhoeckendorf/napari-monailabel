
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from .ui.widgets import MonaiLabelWidget

__all__ = (
    "MonaiLabelWidget",
)
