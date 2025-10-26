import importlib
import json
from pathlib import Path
from typing import Any

from tabulate import tabulate


def import_from_string(dotted_path: str) -> Any:
    """
    The function taken from github.com/UKPLab/sentence-transformers.
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except ModuleNotFoundError:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path,
            class_name,
        )
        raise ImportError(msg)


def tabulate_dict(d: dict[str, Any], **kwargs: Any) -> str:
    """Transforms dict into pretty table string (keys and values forms a row)."""
    return tabulate(
        [[key, val] for key, val in d.items()],
        headers=["Name", "Value"],
        **kwargs,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Loads json from the file"""
    with open(path) as file:
        return json.load(file)
