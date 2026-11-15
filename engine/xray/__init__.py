"""LLM X-Ray: measure what a local language model does inside.

    import xray
    m = xray.Model("Qwen/Qwen2.5-1.5B")
    m.inspect()

See `xray.sdk` for the full API.
"""

__version__ = "0.9.0"

_PUBLIC = ("Model", "Run", "compare", "load_run", "runs")
__all__ = ["__version__", *_PUBLIC]


def __getattr__(name: str):
    # Imported on first use so `import xray` stays instant.
    if name in _PUBLIC:
        from xray import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'xray' has no attribute '{name}'")
