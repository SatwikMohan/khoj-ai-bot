import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import config
import train_engine


class IndexResumeTests(unittest.TestCase):
    def test_completed_candidate_resumes_without_reindexing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_data_dir = root / "raw_data_files"
            raw_data_dir.mkdir()
            source_file = raw_data_dir / "example.txt"
            source_file.write_text("Already indexed", encoding="utf-8")
            vector_db_dir = root / "vector_db"
            vector_db_dir.mkdir()
            source = train_engine.raw_relative_source(source_file, raw_data_dir)
            settings = {"test": "settings"}
            candidate = {
                "schema_version": 2,
                "settings": settings,
                "active_collection": "test_collection",
                "files": {source: {"hash": train_engine.file_hash(source_file), "ids": ["chunk-1"]}},
                "ocr_available_at_build": True,
            }
            train_engine.save_manifest(vector_db_dir, {"settings": {}, "files": {}, "candidate": candidate})

            output = io.StringIO()
            with (
                patch.object(config, "VECTOR_DB_DIR", str(vector_db_dir)),
                patch.object(train_engine, "load_environment"),
                patch.object(train_engine, "chunk_settings", return_value=settings),
                patch.object(train_engine, "ocr_engine_available", return_value=True),
                patch.object(train_engine, "create_vector_store", return_value=(object(), vector_db_dir, "test_collection")) as create_store,
                patch.object(train_engine, "iter_prepared_files") as prepared_files,
                redirect_stdout(output),
            ):
                train_engine.train(raw_data_dir)

            create_store.assert_called_once_with(collection_name="test_collection", reset_collection=False)
            prepared_files.assert_called_once_with([], raw_data_dir.resolve())
            self.assertIn('"event": "index_run_completed"', output.getvalue())
            self.assertIn('"files_unchanged": 1', output.getvalue())
            self.assertIn("Initial index test_collection activated", output.getvalue())
            activated = json.loads((vector_db_dir / train_engine.MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(activated["active_collection"], "test_collection")
            self.assertNotIn("candidate", activated)


if __name__ == "__main__":
    unittest.main()
