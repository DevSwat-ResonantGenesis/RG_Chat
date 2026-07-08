"""Shared pytest setup: make the sibling rg_llm package and heavy optional ML
deps (sentence-transformers/torch) resolvable without requiring a full
multi-GB install, so unit tests can import app modules directly."""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_RG_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../CascadeProjects/RG
_RG_LLM_SRC = os.path.join(_RG_ROOT, "RG_UnifiedLLMClient", "src")
if os.path.isdir(_RG_LLM_SRC) and _RG_LLM_SRC not in sys.path:
    sys.path.insert(0, _RG_LLM_SRC)

if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def encode(self, *a, **k):
            raise RuntimeError("sentence_transformers stub — not available in this test env")

    st_stub.SentenceTransformer = _StubSentenceTransformer
    sys.modules["sentence_transformers"] = st_stub
