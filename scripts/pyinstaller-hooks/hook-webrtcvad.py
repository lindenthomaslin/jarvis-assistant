"""PyInstaller hook for the ``webrtcvad-wheels`` distribution.

The package exposes the module as ``webrtcvad`` while its installed metadata
uses the distribution name ``webrtcvad-wheels``.  The contrib hook assumes the
former, which breaks macOS builds on current wheels.
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

binaries = collect_dynamic_libs("webrtcvad")
datas = copy_metadata("webrtcvad-wheels")
