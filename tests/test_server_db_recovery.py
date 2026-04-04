import os
import tempfile
import unittest
import importlib
import sys


sys.modules.pop("server.db", None)
TaskDB = importlib.import_module("server.db").TaskDB


class ServerDbRecoveryTests(unittest.TestCase):
    def test_requeue_running_tasks_recovers_stale_work_after_restart(self) -> None:
        fd, path = tempfile.mkstemp(prefix="qwentts_db_recovery_", suffix=".sqlite3")
        os.close(fd)
        try:
            db = TaskDB(path)
            task_id = db.enqueue("phrase", {"phrase_id": "phrase-1"})
            db.mark_running(task_id)

            recovered = db.requeue_running_tasks()
            next_task = db.get_next_due()

            self.assertEqual(recovered, 1)
            self.assertIsNotNone(next_task)
            assert next_task is not None
            self.assertEqual(next_task.task_id, task_id)
            self.assertEqual(next_task.status, "queued")
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
