"""Windows compatibility shim loaded only by scripts/build-html.sh.

The legacy d2lbook release derives tab-specific output paths by splitting on
``/``. Python on Windows creates these paths with ``\\``, causing its output
directory to become empty for the default tab.
"""

from d2lbook.config import Config


def _default_tab_dir_windows_safe(self, dirname):
    dirname = dirname.replace("\\", "/")
    tokens = dirname.split("/")
    if self.tabs and "_" in tokens[-1]:
        tokens[-1] = "_".join(tokens[-1].split("_")[:-1])
        return "/".join(tokens)
    return dirname


Config._default_tab_dir = _default_tab_dir_windows_safe
