"""Shared helpers for HTTP parser tests: a MockTransport client and multipart body parsing.

Both HTTP parsers' test cases need to answer the same question — "what did we actually send to
the server?" We factor "record the request" and "turn a multipart body back into a field table"
out here, so assertions can compare the **whole** parsed request body instead of slicing raw bytes
with string windows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser

import httpx

Handler = Callable[[httpx.Request], httpx.Response]


@dataclass(frozen=True)
class FilePart:
    """One file part inside a multipart body."""

    filename: str | None
    content_type: str
    content: bytes


def make_client(handler: Handler) -> httpx.Client:
    """An httpx client that hands every request to ``handler``; never makes a real network call."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def recording_client(response_factory: Callable[[httpx.Request], httpx.Response]) -> tuple[httpx.Client, list]:
    """Return ``(client, requests)``: every request is appended to ``requests`` for later assertions."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response_factory(request)

    return make_client(handler), requests


def _parts(request: httpx.Request) -> list[EmailMessage]:
    """Fully parse the multipart request body using the boundary from its Content-Type header."""
    content_type = request.headers["content-type"]
    raw = b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if not message.is_multipart():
        raise AssertionError(f"request body is not multipart: content-type={content_type!r}")
    return list(message.iter_parts())


def multipart_fields(request: httpx.Request) -> dict[str, str]:
    """All **non-file** fields in the multipart body, as ``{name: value}``."""
    fields: dict[str, str] = {}
    for part in _parts(request):
        if part.get_filename() is not None:
            continue
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        fields[str(name)] = payload.decode("utf-8")
    return fields


def multipart_files(request: httpx.Request) -> dict[str, FilePart]:
    """All **file** parts in the multipart body, as ``{field name: FilePart}``."""
    files: dict[str, FilePart] = {}
    for part in _parts(request):
        filename = part.get_filename()
        if filename is None:
            continue
        name = part.get_param("name", header="content-disposition")
        files[str(name)] = FilePart(
            filename=filename,
            content_type=part.get_content_type(),
            content=part.get_payload(decode=True) or b"",
        )
    return files
