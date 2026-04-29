import os
from typing import Any

from curl_cffi.requests import AsyncSession


DEFAULT_BROWSER_IMPERSONATE = "chrome"


def resolve_browser_impersonate(value: str | None = None) -> str:
    """
    Resolve the browser profile used by curl_cffi.
    """

    return (
        value
        or os.getenv("GEMINI_CURL_IMPERSONATE", "").strip()
        or DEFAULT_BROWSER_IMPERSONATE
    )


def resolve_gemini_proxy(value: str | None = None) -> str | None:
    """
    Resolve the explicit proxy for Gemini web requests.
    """

    proxy = value or os.getenv("GEMINI_PROXY", "").strip()
    return proxy or None


def create_gemini_session(
    *,
    proxy: str | None = None,
    verify: bool = True,
    cookies: Any | None = None,
    **kwargs: Any,
) -> AsyncSession:
    """
    Create a curl_cffi session with one consistent browser/network fingerprint.
    """

    session_kwargs = dict(kwargs)
    impersonate = resolve_browser_impersonate(session_kwargs.pop("impersonate", None))
    explicit_proxy = resolve_gemini_proxy(proxy)
    return AsyncSession(
        impersonate=impersonate,
        proxy=explicit_proxy,
        allow_redirects=session_kwargs.pop("allow_redirects", True),
        verify=verify,
        cookies=cookies,
        **session_kwargs,
    )
