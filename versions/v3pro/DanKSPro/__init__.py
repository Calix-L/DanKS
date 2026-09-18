"""Inference-only enhancements for the separately installed DanKS V3 package."""

GENERATION = "v3pro"
__all__ = ["GENERATION", "ProPolicy"]


def __getattr__(name):
    if name == "ProPolicy":
        from .policy import ProPolicy
        return ProPolicy
    raise AttributeError(name)
