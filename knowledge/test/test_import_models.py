import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from knowledge.api.main import create_app
from knowledge.schema.import_models import ImportSettings, UploadParameters


class ImportModelTests(unittest.TestCase):
    def test_document_id_normalization_and_byte_limit(self):
        params = UploadParameters(filename="中文说明书.MD", document_id="  doc-1  ")
        self.assertEqual(params.document_id, "doc-1")
        self.assertEqual(UploadParameters(filename="a.pdf").document_id, "")
        UploadParameters(filename="a.pdf", document_id="中" * 42)
        for value in ("中" * 43, "a" * 129, "a\0b", 123):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                UploadParameters(filename="a.pdf", document_id=value)

    def test_invalid_filenames(self):
        for name in ("", "../x.md", "a\\x.md", "a\0.md", "CON.md", "LPT1.pdf",
                     "a.md ", "a.md.", "x" * 161, None):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                UploadParameters(filename=name)
        with self.assertRaises(ValidationError) as error:
            UploadParameters(filename="program.exe")
        self.assertEqual(error.exception.errors()[0]["type"], "unsupported_file")

    def test_positive_settings_and_environment_defaults(self):
        with patch.dict(os.environ, {"IMPORT_QUEUE_CAPACITY": "3", "IMPORT_RETENTION_SECONDS": "60"}):
            config = ImportSettings()
        self.assertEqual(config.queue_capacity, 3)
        self.assertEqual(config.retention_seconds, 60)
        for field in ("queue_capacity", "retention_seconds", "max_file_bytes", "max_disk_bytes"):
            for value in (0, -1, True, "invalid"):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    ImportSettings(**{field: value})
        with patch.dict(os.environ, {"IMPORT_QUEUE_CAPACITY": "0"}), self.assertRaises(ValidationError):
            ImportSettings()

    def test_explicit_zero_does_not_fall_back_to_default(self):
        with self.assertRaises(ValidationError):
            create_app(max_file_bytes=0)
