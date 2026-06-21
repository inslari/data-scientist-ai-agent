"""Data-driven engine shared by every concrete sector pack.

A concrete pack (financial, ecommerce, …) is reduced to *data*: a list of
:class:`~src.domains.base.Expert` personas, their routing keywords, and a few
framing labels. The reusable machinery here turns that data into:

* expert routing (LLM-first + deterministic keyword fallback) — :mod:`router`
* tool-using ReAct expert agents + council synthesis — :mod:`agents`
* the four prompt hooks + ``select_experts`` / ``analyze`` — :class:`pack.GenericDomainPack`

This keeps per-sector code to a single small ``pack.py`` (config + experts).
"""
