import hashlib
import os
import random
import re
import time

from curl_cffi.requests import AsyncSession, Cookies, Response
import orjson as json

from .load_browser_cookies import HAS_BC3, load_browser_cookies
from .logger import logger
from .http_session import create_gemini_session
from .parsing import extract_json_from_response, get_nested_value
from .rotate_1psidts import (
    _extract_cookie_value,
    _get_cookies_cache_path,
    _get_cookie_cache_dir,
)
from ..constants import AccountStatus, Endpoint, GRPC, Headers
from ..exceptions import AuthError


def _clone_cookies(cookies: dict | Cookies) -> Cookies:
    jar = Cookies()
    if isinstance(cookies, Cookies):
        for cookie in cookies.jar:
            if not cookie.is_expired():
                jar.set(
                    str(cookie.name),
                    str(cookie.value),
                    domain=cookie.domain or ".google.com",
                    path=cookie.path or "/",
                )
    else:
        for name, value in cookies.items():
            if value:
                jar.set(str(name), str(value), domain=".google.com", path="/")
    return jar


def build_cookie_fingerprint(cookies: dict | Cookies) -> str | None:
    entries: list[tuple[str, str, str, str]] = []

    if isinstance(cookies, Cookies):
        for cookie in cookies.jar:
            if cookie.is_expired():
                continue
            entries.append(
                (
                    str(cookie.name),
                    str(cookie.value),
                    str(cookie.domain or ""),
                    str(cookie.path or "/"),
                )
            )
    else:
        for name, value in cookies.items():
            if value:
                entries.append((str(name), str(value), ".google.com", "/"))

    if not entries:
        return None

    entries.sort()
    payload = json.dumps(entries).decode("utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def _probe_account_status(
    client: AsyncSession,
    access_token: str | None,
    build_label: str | None,
    session_id: str | None,
    language: str | None,
    verbose: bool = False,
) -> AccountStatus | None:
    if not access_token:
        return None

    params = {
        "rpcids": str(GRPC.GET_USER_STATUS),
        "hl": language or "en",
        "_reqid": random.randint(10000, 99999),
        "rt": "c",
        "source-path": "/app",
    }
    if build_label:
        params["bl"] = build_label
    if session_id:
        params["f.sid"] = session_id

    response = await client.post(
        Endpoint.BATCH_EXEC,
        params=params,
        headers={
            **Headers.GEMINI.value,
            **Headers.BATCH_EXEC.value,
            **Headers.SAME_DOMAIN.value,
        },
        data={
            "at": access_token,
            "f.req": json.dumps([[[str(GRPC.GET_USER_STATUS), "[]", None, "generic"]]]).decode("utf-8"),
        },
    )
    if verbose:
        logger.debug(f"HTTP Request: POST {Endpoint.BATCH_EXEC} [{response.status_code}]")

    if response.status_code != 200:
        return None

    response_json = extract_json_from_response(response.text)
    for part in response_json:
        part_body_str = get_nested_value(part, [2])
        if not part_body_str:
            continue

        try:
            part_body = json.loads(part_body_str)
        except json.JSONDecodeError:
            continue

        return AccountStatus.from_status_code(get_nested_value(part_body, [14]))

    return None


async def _send_request(
    client: AsyncSession, cookies: dict | Cookies, verbose: bool = False
) -> Response:
    """
    Send http request with provided cookies using a shared session.
    """

    client.cookies.clear()
    if isinstance(cookies, Cookies):
        client.cookies.update(cookies)
    else:
        for k, v in cookies.items():
            client.cookies.set(k, v, domain=".google.com")

    response = await client.get(Endpoint.INIT, headers=Headers.GEMINI.value)
    if verbose:
        logger.debug(f"HTTP Request: GET {Endpoint.INIT} [{response.status_code}]")
    response.raise_for_status()
    return response


async def get_access_token(
    base_cookies: dict | Cookies,
    proxy: str | None = None,
    verbose: bool = False,
    verify: bool = True,
    **session_kwargs,
) -> tuple[str | None, str | None, str | None, str | None, str | None, AsyncSession]:
    """
    Send a get request to gemini.google.com for each group of available cookies and return
    the value of "SNlM0e" as access token on the first successful request.

    Returns the **live** AsyncSession that succeeded so the caller can reuse
    the same TLS connection for subsequent requests.

    Parameters
    ----------
    base_cookies: `dict | curl_cffi.requests.Cookies`
        Initial cookies to try. Can be a dictionary or a Cookies object.
    proxy: `str`, optional
        Proxy URL.
    verbose: `bool`, optional
        If True, log more details.
    verify: `bool`, optional
        Whether to verify SSL certificates.

    Returns
    -------
    `tuple[str | None, str | None, str | None, str | None, str | None, AsyncSession]`
        By order: access token; build label; session id; language; file push id; live AsyncSession of the successful request.

    Raises
    ------
    `gemini_webapi.AuthError`
        If all requests failed.
    """

    client = create_gemini_session(
        proxy=proxy,
        verify=verify,
        **session_kwargs,
    )

    try:
        response = await client.get(Endpoint.GOOGLE)
        if verbose:
            logger.debug(
                f"HTTP Request: GET {Endpoint.GOOGLE} [{response.status_code}]"
            )
        preflight_cookies = Cookies(client.cookies)
    except Exception:
        await client.close()
        raise

    extra_cookies = Cookies()
    if response.status_code == 200:
        extra_cookies = preflight_cookies

    cookie_jars_to_test: list[tuple[Cookies, str, str, str | None]] = []
    seen_fingerprints: set[str] = set()

    if isinstance(base_cookies, Cookies):
        base_psid = _extract_cookie_value(base_cookies, "__Secure-1PSID")
    else:
        base_psid = base_cookies.get("__Secure-1PSID")
    def add_candidate(jar: Cookies, source: str, group_name: str) -> None:
        jar.update(extra_cookies)
        fingerprint = build_cookie_fingerprint(jar)
        if not fingerprint:
            return
        if fingerprint in seen_fingerprints:
            return
        seen_fingerprints.add(fingerprint)
        cookie_jars_to_test.append((jar, source, group_name, fingerprint))

    # Phase 1: Explicit Cookies
    if base_psid:
        add_candidate(_clone_cookies(base_cookies), "explicit", "Explicit Cookies")
    elif verbose:
        logger.debug("Skipping explicit cookies. __Secure-1PSID is not provided.")

    # Phase 2: Cache
    if base_psid:
        cache_seed = Cookies()
        cache_seed.set("__Secure-1PSID", base_psid, domain=".google.com")
        cache_file = _get_cookies_cache_path(cache_seed)

        if cache_file and cache_file.is_file():
            content = cache_file.read_text().strip()
            if content:
                try:
                    cache_data = json.loads(content)
                    jar = Cookies()
                    for cookie in cache_data:
                        expires = cookie.get("expires")
                        if expires and expires < time.time():
                            continue
                        jar.set(
                            cookie["name"],
                            cookie["value"],
                            domain=cookie.get("domain", ".google.com"),
                            path=cookie.get("path", "/"),
                        )
                    add_candidate(jar, "cache", "Cache")
                except Exception as e:
                    logger.warning(f"Failed to parse cached cookies as JSON: {e}")
            elif verbose:
                logger.debug("Skipping loading cached cookies. Cache file is empty.")
        elif verbose:
            logger.debug("Skipping loading cached cookies. Cache file not found.")
    else:
        cache_files = list(_get_cookie_cache_dir().glob(".cached_cookies_*.json"))
        if cache_files:
            cache_file = max(cache_files, key=lambda p: p.stat().st_mtime)
            content = cache_file.read_text().strip()
            if content:
                try:
                    cache_data = json.loads(content)
                    jar = Cookies()
                    for cookie in cache_data:
                        expires = cookie.get("expires")
                        if expires and expires < time.time():
                            continue
                        jar.set(
                            cookie["name"],
                            cookie["value"],
                            domain=cookie.get("domain", ".google.com"),
                            path=cookie.get("path", "/"),
                        )
                    add_candidate(jar, "cache:latest", "Cache (Latest)")
                except Exception as e:
                    logger.warning(f"Failed to parse cached cookies as JSON: {e}")

    # Phase 3: Browser Cookies
    try:
        browser_cookies = load_browser_cookies(
            domain_name="google.com", verbose=verbose
        )
        if browser_cookies:
            for browser, cookie_list in browser_cookies.items():
                temp_cookies = {c["name"]: c["value"] for c in cookie_list}
                secure_1psid = temp_cookies.get("__Secure-1PSID")
                secure_1psidts = temp_cookies.get("__Secure-1PSIDTS", "")

                if secure_1psid:
                    if base_psid and base_psid != secure_1psid:
                        if verbose:
                            logger.debug(
                                f"Skipping loading local browser cookies from {browser}. "
                                "__Secure-1PSID does not match the one provided."
                            )
                        continue

                    jar = Cookies()
                    for cookie in cookie_list:
                        name = cookie["name"]
                        # Load only __Secure-1PSID and __Secure-1PSIDTS to prevent HTTP 401 errors when rotating cookies.
                        if name not in ["__Secure-1PSID", "__Secure-1PSIDTS"]:
                            continue

                        jar.set(
                            cookie["name"],
                            cookie["value"],
                            domain=cookie["domain"],
                            path=cookie["path"],
                        )

                    add_candidate(jar, f"browser:{browser}", f"Browser ({browser})")
                    if verbose:
                        logger.debug(
                            f"Prepared essential browser cookies from {browser}."
                        )

        if (
            HAS_BC3
            and not any(group.startswith("Browser") for _, _, group, _ in cookie_jars_to_test)
            and verbose
        ):
            logger.debug(
                "Skipping loading local browser cookies. Login to gemini.google.com in your browser first."
            )
    except Exception:
        if verbose:
            logger.debug(
                "Skipping loading local browser cookies (Not available or no permission)."
            )

    prefer_browser_cookies = os.getenv(
        "GEMINI_PREFER_BROWSER_COOKIES",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if prefer_browser_cookies:
        def auth_priority(candidate: tuple[Cookies, str, str, str | None]) -> int:
            auth_source = candidate[1]
            if auth_source.startswith("browser:"):
                return 0
            if auth_source == "explicit":
                return 1
            return 2

        cookie_jars_to_test.sort(key=auth_priority)
        if verbose and any(
            auth_source.startswith("browser:")
            for _, auth_source, _, _ in cookie_jars_to_test
        ):
            logger.debug(
                "Prioritizing matching browser cookies for Gemini auth to keep media sessions fresh."
            )

    current_attempt = 0
    first_success: dict[str, object] | None = None
    for jar, auth_source, group_name, fingerprint in cookie_jars_to_test:
        current_attempt += 1
        try:
            response = await _send_request(client, jar, verbose=verbose)
            access_token = re.search(r'"SNlM0e":\s*"(.*?)"', response.text)
            build_label = re.search(r'"cfb2h":\s*"(.*?)"', response.text)
            session_id = re.search(r'"FdrFJe":\s*"(.*?)"', response.text)
            language = re.search(r'"TuX5cc":\s*"(.*?)"', response.text)
            push_id = re.search(r'"qKIAYe":\s*"(.*?)"', response.text)
            if access_token or build_label or session_id or language or push_id:
                account_status = await _probe_account_status(
                    client=client,
                    access_token=access_token.group(1) if access_token else None,
                    build_label=build_label.group(1) if build_label else None,
                    session_id=session_id.group(1) if session_id else None,
                    language=language.group(1) if language else None,
                    verbose=verbose,
                )
                if verbose:
                    logger.debug(
                        f"Init attempt ({current_attempt}) from {group_name} succeeded."
                    )
                    if account_status is not None:
                        logger.debug(
                            f"Init attempt ({current_attempt}) from {group_name} reached account status {account_status.name}."
                        )

                candidate_result = {
                    "access_token": access_token.group(1) if access_token else None,
                    "build_label": build_label.group(1) if build_label else None,
                    "session_id": session_id.group(1) if session_id else None,
                    "language": language.group(1) if language else None,
                    "push_id": push_id.group(1) if push_id else None,
                    "auth_source": auth_source,
                    "auth_cookie_fingerprint": fingerprint,
                    "account_status": account_status,
                    "live_cookies": Cookies(client.cookies),
                }
                if first_success is None:
                    first_success = candidate_result

                if account_status == AccountStatus.AVAILABLE:
                    client.cookies.clear()
                    client.cookies.update(candidate_result["live_cookies"])
                    client._gemini_auth_source = auth_source
                    client._gemini_auth_cookie_fingerprint = fingerprint
                    client._gemini_auth_account_status = account_status
                    return (
                        candidate_result["access_token"],
                        candidate_result["build_label"],
                        candidate_result["session_id"],
                        candidate_result["language"],
                        candidate_result["push_id"],
                        client,
                    )
        except Exception:
            if verbose:
                logger.debug(
                    f"Init attempt ({current_attempt}) from {group_name} failed."
                )

    if first_success is not None:
        client.cookies.clear()
        client.cookies.update(first_success["live_cookies"])
        client._gemini_auth_source = first_success["auth_source"]
        client._gemini_auth_cookie_fingerprint = first_success["auth_cookie_fingerprint"]
        client._gemini_auth_account_status = first_success["account_status"]
        return (
            first_success["access_token"],
            first_success["build_label"],
            first_success["session_id"],
            first_success["language"],
            first_success["push_id"],
            client,
        )

    await client.close()
    raise AuthError(
        f"Failed to initialize client after {current_attempt} attempts. SECURE_1PSIDTS "
        "could get expired frequently, please make sure cookie values are up to date."
    )
