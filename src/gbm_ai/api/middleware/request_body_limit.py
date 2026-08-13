from __future__ import annotations

import json
from collections.abc import Awaitable, Callable


class RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """
    ASGI-level request body ceiling for study upload routes.

    It checks Content-Length when present and also counts streamed body chunks,
    so chunked transfer does not bypass the application-level ceiling.
    """

    def __init__(
        self,
        app,
        *,
        max_bytes: int,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)
        self.api_prefix = api_prefix.rstrip("/")

    def _is_upload_request(self, scope: dict) -> bool:
        if scope.get("type") != "http":
            return False
        if scope.get("method", "").upper() != "POST":
            return False

        path = scope.get("path", "")
        return (
            path.startswith(f"{self.api_prefix}/studies/")
            and path.endswith("/upload")
        )

    async def _send_413(self, send) -> None:
        payload = json.dumps(
            {
                "detail": "upload request exceeds configured size limit"
            }
        ).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (
                        b"content-length",
                        str(len(payload)).encode("ascii"),
                    ),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": payload,
                "more_body": False,
            }
        )

    async def __call__(self, scope, receive, send):
        if not self._is_upload_request(scope):
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_413(send)
                    return
            except ValueError:
                pass

        consumed = 0
        response_started = False

        async def limited_receive():
            nonlocal consumed
            message = await receive()

            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_bytes:
                    raise RequestBodyTooLarge

            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._send_413(send)
