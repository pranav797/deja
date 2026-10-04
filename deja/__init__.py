from .cache import Cache, openai_chat_verifier, openai_embedder, openai_verifier


def wrap(client, **cache_kwargs):
    """One-line integration: client = deja.wrap(OpenAI())."""
    return Cache(**cache_kwargs).wrap(client)


__all__ = ["Cache", "openai_chat_verifier", "openai_embedder", "openai_verifier", "wrap"]
