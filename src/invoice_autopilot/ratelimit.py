"""Per-client rate limiting for public demos (protects the LLM budget).

Disabled unless RATE_LIMIT_PER_HOUR is set. In-memory, per instance: fine for a
single Cloud Run instance; use Redis or an API gateway for a real multi-instance service.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_hour: int, methods: tuple[str, ...] = ("POST",)):
        super().__init__(app)
        self.limit, self.methods = limit_per_hour, methods
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    @staticmethod
    def client_id(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method not in self.methods:
            return await call_next(request)
        now, window = time.monotonic(), self.hits[self.client_id(request)]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= self.limit:
            return JSONResponse({"detail": f"Rate limit: {self.limit} requests per hour"}, status_code=429)
        window.append(now)
        return await call_next(request)


def rate_limit_from_env() -> int:
    try:
        return max(0, int(os.environ.get("RATE_LIMIT_PER_HOUR", "0")))
    except ValueError:
        return 0
