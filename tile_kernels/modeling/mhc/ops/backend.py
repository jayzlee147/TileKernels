"""Backend selection for MHC ops: 'tilelang' or 'triton'."""

import os

_BACKEND = os.environ.get('MHC_BACKEND', 'tilelang')  # default tilelang for backward compat


def get_backend() -> str:
    return _BACKEND


def set_backend(backend: str):
    global _BACKEND
    assert backend in ('tilelang', 'triton'), f'Unknown backend: {backend}'
    _BACKEND = backend


def use_triton() -> bool:
    return _BACKEND == 'triton'
