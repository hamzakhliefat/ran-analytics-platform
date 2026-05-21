# -*- coding: utf-8 -*-
import sys

# Dynamic mock class that prevents Streamlit watcher / transformers lazy import crashes
class DummyModule(object):
    def __init__(self, name):
        self.__name__ = name
    def __getattr__(self, name):
        if name in ('__path__', '__file__', '__spec__'):
            raise AttributeError(name)
        return DummyModule(f"{self.__name__}.{name}")
    def __call__(self, *args, **kwargs):
        return DummyModule("dummy")
    def __repr__(self):
        return f"<DummyModule {self.__name__}>"

# Pre-populate sys.modules with mocked submodules so standard package import syntax resolves them instantly
submodules = [
    'torchvision.transforms',
    'torchvision.transforms.functional',
    'torchvision.transforms.v2',
    'torchvision.transforms.v2.functional',
    'torchvision.ops',
    'torchvision.ops.boxes',
    'torchvision.io'
]

for sub in submodules:
    if sub not in sys.modules:
        sys.modules[sub] = DummyModule(sub)
