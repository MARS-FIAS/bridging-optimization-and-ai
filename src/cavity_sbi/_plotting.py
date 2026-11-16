"""Load optional visualization libraries only when a plotting function is used."""
import importlib


class LazyModule:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, name):
        return getattr(importlib.import_module(self._name), name)


plt = LazyModule('matplotlib.pyplot')
matplotlib = LazyModule('matplotlib')
