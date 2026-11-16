"""Deterministic, side-effect-free normalization and tokenization.

Nothing here executes, decodes, or fetches input content.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from alert_disposition.features.config import Limits

R_EMAIL = "redacted_email"
R_URL = "redacted_url"
R_IP = "redacted_ip"
R_GUID = "redacted_guid"
R_SID = "redacted_sid"
R_HASH = "redacted_hash"
R_HEX = "redacted_hex"
R_BLOB = "redacted_blob"
R_HOST = "redacted_host"
R_USER = "redacted_user"
REDACTED_PREFIX = "redacted_"

_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_GUID = re.compile(
    r"\{?\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b\}?", re.IGNORECASE
)
_SID = re.compile(r"\bS-1-\d+(?:-\d+){1,}\b", re.IGNORECASE)
_IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")
_HASH = re.compile(r"\b(?:[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64}|[0-9a-f]{128})\b", re.IGNORECASE)
_LONG_HEX = re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{16,}\b", re.IGNORECASE)
_BLOB = re.compile(r"[A-Za-z0-9+/_-]{40,}={0,2}")
_UNC_HOST = re.compile(r"^(\\\\|//)([^\\/]+)")
_FQDN = re.compile(r"\b(?:[a-z0-9][a-z0-9-]*\.){2,}[a-z]{2,6}\b", re.IGNORECASE)
_WIN_USER = re.compile(r"(?i)\b([a-z]:[\\/]users[\\/])[^\\/\s\"']+")
_UNIX_USER = re.compile(r"(?<![\w.])(/(?:home|Users)/)[^/\s\"']+")

_FILE_EXTENSIONS = frozenset(
    [
        "exe",
        "dll",
        "sys",
        "bat",
        "cmd",
        "ps1",
        "vbs",
        "js",
        "jar",
        "py",
        "sh",
        "msi",
        "scr",
        "com",
        "lnk",
        "tmp",
        "log",
        "txt",
        "dat",
        "zip",
        "rar",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "pdf",
    ]
)

_ENV_VARS: dict[str, str] = {
    "temp": r"c:\users\redacted_user\appdata\local\temp",
    "tmp": r"c:\users\redacted_user\appdata\local\temp",
    "appdata": r"c:\users\redacted_user\appdata\roaming",
    "localappdata": r"c:\users\redacted_user\appdata\local",
    "userprofile": r"c:\users\redacted_user",
    "programdata": r"c:\programdata",
    "systemroot": r"c:\windows",
    "windir": r"c:\windows",
    "programfiles": r"c:\program files",
    "programfiles(x86)": r"c:\program files (x86)",
    "systemdrive": "c:",
}
_ENV_REF = re.compile(r"%([A-Za-z()0-9_]+)%")


def redact(text: str) -> str:
    """Replace unstable or sensitive values with fixed placeholders."""
    text = _URL.sub(R_URL, text)
    text = _EMAIL.sub(R_EMAIL, text)
    text = _GUID.sub(R_GUID, text)
    text = _SID.sub(R_SID, text)
    text = _IPV4.sub(R_IP, text)
    text = _HASH.sub(R_HASH, text)
    text = _LONG_HEX.sub(R_HEX, text)
    text = _BLOB.sub(R_BLOB, text)
    text = _WIN_USER.sub(lambda m: m.group(1) + R_USER, text)
    text = _UNIX_USER.sub(lambda m: m.group(1) + R_USER, text)
    return _FQDN.sub(_fqdn_repl, text)


def _fqdn_repl(match: re.Match[str]) -> str:
    last = match.group(0).rsplit(".", 1)[-1].lower()
    return match.group(0) if last in _FILE_EXTENSIONS - {"com"} else R_HOST


def normalize_text(text: str | None, limits: Limits) -> tuple[str, bool]:
    """Bound, redact, lowercase, and collapse whitespace.

    Returns:
        Normalized text and whether it was truncated.
    """
    if text is None:
        return "", False
    truncated = len(text) > limits.max_text_chars
    cleaned = redact(text[: limits.max_text_chars])
    return " ".join(cleaned.lower().split()), truncated


def is_unstable_component(part: str) -> bool:
    """Whether a path component is likely a high-cardinality identifier."""
    if part == R_USER:
        return False
    if REDACTED_PREFIX in part or len(part) > 64:
        return True
    if re.fullmatch(r"[0-9a-f]{8,}", part) and re.search(r"\d", part):
        return True
    if re.fullmatch(r"\d{4,}", part):
        return True
    return bool(
        re.fullmatch(r"[\w-]{20,}", part) and re.search(r"\d", part) and re.search(r"[a-z]", part)
    )


@dataclass(frozen=True)
class PathInfo:
    """Tokens and structure derived from one path or file name.

    Attributes:
        tokens: Atomic tokens (``dir:``, ``ext:``, ``basename:``, ``loc:``).
        depth: Directory depth, None when no full path was observed.
        has_full_path: True only when a directory was actually present.
    """

    tokens: tuple[str, ...] = ()
    depth: int | None = None
    has_full_path: bool = False


def _expand_env(path: str) -> str:
    return _ENV_REF.sub(lambda m: _ENV_VARS.get(m.group(1).lower(), m.group(0)), path)


def _is_windows_style(path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:|^\\\\|^\\SystemRoot", path)) or ("\\" in path)


def _basename_tokens(name: str) -> tuple[list[str], str]:
    name = name.strip().strip("\"'").lower()
    name = redact(name)
    tokens: list[str] = []
    if name == "" or REDACTED_PREFIX in name or is_unstable_component(name):
        return tokens, name
    tokens.append(f"basename:{name}")
    suffix = name.rsplit(".", 1)[-1] if "." in name.strip(".") else ""
    if re.fullmatch(r"[a-z0-9]{1,10}", suffix):
        tokens.append(f"ext:{suffix}")
    return tokens, name


def path_info(path: str | None, *, is_full_path: bool, limits: Limits) -> PathInfo:
    """Tokenize a path.

    Args:
        path: Path or bare file name.
        is_full_path: Whether ``path`` came from a validated full-path field.
            When False only the basename is used and no directory signal or
            ``has_full_path`` is produced.
        limits: Work bounds.
    """
    if path is None or path.strip() == "":
        return PathInfo()
    path = path[: limits.max_path_chars]
    if not is_full_path:
        base = re.split(r"[\\/]", path.strip())[-1]
        early_tokens, _ = _basename_tokens(base)
        return PathInfo(tokens=tuple(early_tokens))

    expanded = _expand_env(path.strip().strip("\"'"))
    windows = _is_windows_style(expanded)
    tokens: list[str] = []
    if windows:
        unc = _UNC_HOST.match(expanded)
        if unc is not None:
            tokens.append("path:unc")
            expanded = expanded[unc.end() :]
        pure = PureWindowsPath(redact(expanded).lower())
        parts = list(pure.parts[1:] if pure.anchor != "" else pure.parts)
    else:
        pure_posix = PurePosixPath(redact(expanded))
        parts = [p for p in pure_posix.parts if p != "/"]
    if len(parts) == 0:
        return PathInfo(tokens=tuple(tokens))
    *dirs, name = parts
    dirs = [d.strip("\\/") for d in dirs if d.strip("\\/") != ""]
    if len(dirs) > limits.max_path_parts:
        dirs = dirs[: limits.max_path_parts]
    for d in dirs:
        if d == R_USER:
            tokens.append("loc:user_profile")
        elif not is_unstable_component(d):
            tokens.append(f"dir:{d if windows else d.lower()}")
    base_tokens, _ = _basename_tokens(name)
    tokens.extend(base_tokens)
    has_dir = len(dirs) > 0 or "path:unc" in tokens
    return PathInfo(tokens=tuple(dict.fromkeys(tokens)), depth=len(dirs), has_full_path=has_dir)


@dataclass(frozen=True)
class CommandInfo:
    """Tokens and scalars derived from one command line.

    Attributes:
        tokens: Atomic tokens (``exe:``, ``flag:``, ``cmd_dir:``, ``cmd_ext:``, ``has:``).
        length: Raw command length (after bounding).
        arg_count: Arguments after the executable.
        flag_count: Switch/option count.
        truncated: Whether length or argument bounds were hit.
    """

    tokens: tuple[str, ...] = ()
    length: int = 0
    arg_count: int = 0
    flag_count: int = 0
    truncated: bool = False


def split_command(cmd: str, *, windows: bool, max_args: int) -> tuple[list[str], bool]:
    """Split a command line into arguments without raising.

    Windows mode: only double quotes quote; backslashes are literal.
    POSIX mode: single and double quotes, backslash escapes.
    Unbalanced quotes consume the rest of the input.

    Returns:
        Arguments (bounded) and whether the argument bound was hit.
    """
    args: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    in_token = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if quote is not None:
            if ch == quote:
                quote = None
            elif (
                not windows and ch == "\\" and quote == '"' and i + 1 < n and cmd[i + 1] in '"\\$`'
            ):
                i += 1
                buf.append(cmd[i])
            else:
                buf.append(ch)
        elif ch == '"' or (not windows and ch == "'"):
            quote = ch
            in_token = True
        elif not windows and ch == "\\" and i + 1 < n:
            i += 1
            buf.append(cmd[i])
            in_token = True
        elif ch.isspace():
            if in_token:
                args.append("".join(buf))
                buf, in_token = [], False
                if len(args) >= max_args:
                    return args, True
        else:
            buf.append(ch)
            in_token = True
        i += 1
    if in_token:
        args.append("".join(buf))
    return args[:max_args], len(args) > max_args


_FLAG = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")
_WIN_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|%[A-Za-z()0-9_]+%[\\/])")
_POSIX_ABS = re.compile(r"^/[^/\s]+/[^/\s]")


def _is_opaque(arg: str) -> bool:
    return len(arg) >= 40 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", arg) is not None


def command_info(cmd: str | None, limits: Limits) -> CommandInfo:
    """Extract atomic command tokens and a few scalars.

    The command is only tokenized and pattern-matched; opaque arguments are
    reduced to ``has:opaque_arg`` and never preserved.
    """
    if cmd is None or cmd.strip() == "":
        return CommandInfo()
    truncated = len(cmd) > limits.max_cmd_chars
    cmd = cmd[: limits.max_cmd_chars]
    head = cmd.lstrip()
    windows = "\\" in head[:260] or (
        re.match(r'^"?[^"\s]*\.(exe|com|bat|cmd)\b', head, re.IGNORECASE) is not None
    )
    args, arg_truncated = split_command(cmd, windows=windows, max_args=limits.max_cmd_args)
    if len(args) == 0:
        return CommandInfo(length=len(cmd), truncated=truncated)

    tokens: list[str] = []
    exe_tokens, _ = _basename_tokens(re.split(r"[\\/]", args[0])[-1])
    exe = next((t.removeprefix("basename:") for t in exe_tokens if t.startswith("basename:")), None)
    if exe is not None:
        tokens.append(f"exe:{exe}")

    flag_count = 0
    for arg in args[1:]:
        if _is_opaque(arg):
            tokens.append("has:opaque_arg")
            continue
        if "://" in arg:
            tokens.append("has:url")
            continue
        flag = _flag_name(arg, windows=windows)
        if flag is not None:
            flag_count += 1
            tokens.append(f"flag:{flag}")
            continue
        if _WIN_ABS.match(arg) or _POSIX_ABS.match(arg):
            info = path_info(arg, is_full_path=True, limits=limits)
            tokens.extend(
                "cmd_" + t for t in info.tokens if t.startswith(("dir:", "ext:", "loc:", "path:"))
            )
    return CommandInfo(
        tokens=tuple(dict.fromkeys(tokens)),
        length=len(cmd),
        arg_count=len(args) - 1,
        flag_count=flag_count,
        truncated=truncated or arg_truncated,
    )


def _flag_name(arg: str, *, windows: bool) -> str | None:
    if arg.startswith("--") or arg.startswith("-"):
        name = arg.lstrip("-")
    elif windows and arg.startswith("/") and not re.search(r"[\\/]", arg[1:]):
        name = arg[1:]
    else:
        return None
    name = re.split(r"[=:]", name, maxsplit=1)[0].lower()
    return name if _FLAG.fullmatch(name) is not None else None
