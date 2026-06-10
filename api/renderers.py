from __future__ import annotations

import html
import json
from typing import Any

from rest_framework.renderers import BaseRenderer


class SvgRenderer(BaseRenderer):
    media_type = 'image/svg+xml'
    format = 'svg'
    charset = 'utf-8'

    def render(self, data: Any, accepted_media_type: str | None = None, renderer_context: dict[str, Any] | None = None) -> bytes:
        if data is None:
            return b''
        if isinstance(data, bytes):
            return data
        if isinstance(data, str) and data.lstrip().startswith('<svg'):
            return data.encode(self.charset)

        status_code = 500
        if renderer_context and renderer_context.get('response') is not None:
            status_code = int(getattr(renderer_context['response'], 'status_code', status_code))

        if isinstance(data, str):
            message = data
        else:
            message = json.dumps(data, ensure_ascii=False)

        return self._error_svg(status_code, message).encode(self.charset)

    @staticmethod
    def _error_svg(status_code: int, message: str) -> str:
        escaped_message = html.escape(message[:360])
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="240" viewBox="0 0 900 240" role="img" aria-label="GitStarter SVG error">
  <rect width="900" height="240" rx="0" fill="#f8f3e8"/>
  <rect x="32" y="32" width="836" height="176" rx="22" fill="#fffdf7" stroke="#d05a2f" stroke-width="2"/>
  <text x="60" y="82" font-family="ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="800" fill="#19231d">GitStarter SVG error</text>
  <text x="60" y="120" font-family="ui-sans-serif, system-ui, sans-serif" font-size="16" font-weight="700" fill="#d05a2f">HTTP {status_code}</text>
  <text x="60" y="158" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" fill="#667161">{escaped_message}</text>
</svg>'''


class SseRenderer(BaseRenderer):
    media_type = 'text/event-stream'
    format = 'event-stream'
    charset = 'utf-8'

    def render(self, data: Any, accepted_media_type: str | None = None, renderer_context: dict[str, Any] | None = None) -> bytes:
        if data is None:
            return b''

        status_code = 500
        if renderer_context and renderer_context.get('response') is not None:
            status_code = int(getattr(renderer_context['response'], 'status_code', status_code))

        event_name = 'error' if status_code >= 400 else 'message'
        payload = data if isinstance(data, dict) else {'data': data}
        text = json.dumps(payload, ensure_ascii=False)
        return f'event: {event_name}\ndata: {text}\n\n'.encode(self.charset)
