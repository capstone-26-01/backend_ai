from typing import cast

from django.http import HttpResponse
from django.test import TestCase
import yaml


class DocsEndpointsTests(TestCase):
    def test_schema_endpoint_returns_openapi_document(self):
        response = cast(HttpResponse, self.client.get('/api/schema/'))

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'application/vnd.oai.openapi',
            response.headers.get('Content-Type', ''),
        )

        payload = cast(dict[str, object], yaml.safe_load(response.content))
        paths = cast(dict[str, object], payload['paths'])

        self.assertIn('openapi', payload)
        self.assertIn('paths', payload)
        self.assertIn('/api/repo/', paths)
        self.assertIn('/api/qa/', paths)

    def test_swagger_docs_endpoint_renders_ui(self):
        response = cast(HttpResponse, self.client.get('/api/docs/'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'SwaggerUIBundle')
        self.assertContains(response, '/api/schema/')
