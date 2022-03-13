import pathlib

import mankey

PACKAGE_ROOT = pathlib.Path(mankey.__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"

name = "mankey"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
