from django.conf import settings
from rest_framework.throttling import SimpleRateThrottle


class ShareCreateRateThrottle(SimpleRateThrottle):
    scope = 'share_create'

    def get_rate(self):
        return getattr(settings, 'SHARE_CREATE_THROTTLE_RATE', None) or super().get_rate()

    def get_cache_key(self, request, view):
        if request.method != 'POST':
            return None
        return self.cache_format % {
            'scope': self.scope,
            'ident': self.get_ident(request),
        }
