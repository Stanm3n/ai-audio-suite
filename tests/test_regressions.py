"""API access and worker regression tests; no model downloads or GPU needed.

Run: python -m unittest discover -s tests -v
Install the small test dependencies listed in README in a separate environment.
Only the heavyweight inference libraries are replaced during module imports.
"""
import importlib
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

torch_stub = MagicMock()
torch_stub.cuda.is_available.return_value = False
with patch.dict(sys.modules, {
    "torch": torch_stub,
    "whisperx": MagicMock(),
    "transformers": MagicMock(),
}):
    state = importlib.import_module("state")
    api = importlib.import_module("api.jobs")
    worker = importlib.import_module("core.transcription")
    from models import Job, JobResult, JobStatus


def make_job(job_id="owned", token="owned-token", status=JobStatus.COMPLETE):
    return Job(
        job_id=job_id, access_token=token, status=status, created_at=1,
        filename="private.wav", model="medium", language="en", device="cpu",
        result=JobResult(segments=[{"start": 0, "end": 1, "text": "Private transcript"}]),
    )


class JobAccessTests(unittest.TestCase):
    def setUp(self):
        state.jobs.clear()
        state.cancelled_jobs.clear()
        state.jobs.update({
            "owned": make_job(),
            "foreign": make_job("foreign", "foreign-token"),
            "queued": make_job("queued", "queued-token", JobStatus.QUEUED),
        })
        app = FastAPI()
        app.include_router(api.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        state.jobs.clear()
        state.cancelled_jobs.clear()

    def test_all_job_endpoints_reject_missing_wrong_and_unicode_tokens(self):
        endpoints = [
            ("GET", "/api/v1/results/owned"),
            ("GET", "/api/v1/results/owned/download"),
            ("GET", "/api/v1/jobs"),
            ("POST", "/api/v1/jobs/queued/cancel"),
        ]
        for method, url in endpoints:
            for token in [None, "wrong", "fremd-ä"]:
                with self.subTest(url=url, token=token):
                    params = {} if token is None else {"access_token": token}
                    response = self.client.request(method, url, params=params)
                    self.assertEqual(response.status_code, 403)
                    self.assertNotIn("Private transcript", response.text)
        self.assertEqual(state.jobs["queued"].status, JobStatus.QUEUED)

    def test_result_requires_its_own_token_and_hides_internal_fields(self):
        response = self.client.get("/api/v1/results/owned", headers={"Authorization": "Bearer owned-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["segments"][0]["text"], "Private transcript")
        self.assertNotIn("access_token", response.json())
        self.assertNotIn("temp_audio_path", response.json())
        response = self.client.get("/api/v1/results/foreign", headers={"Authorization": "Bearer owned-token"})
        self.assertEqual(response.status_code, 403)

    def test_listing_is_scoped_to_the_supplied_token(self):
        response = self.client.get("/api/v1/jobs", headers={"Authorization": "Bearer owned-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([j["job_id"] for j in response.json()], ["owned"])
        self.assertNotIn("access_token", response.json()[0])
        self.assertNotIn("temp_audio_path", response.json()[0])

    def test_download_with_bearer_token(self):
        response = self.client.get("/api/v1/results/owned/download", headers={"Authorization": "Bearer owned-token"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Private transcript", response.text)

    def test_cancel_with_bearer_token(self):
        response = self.client.post("/api/v1/jobs/queued/cancel", headers={"Authorization": "Bearer queued-token"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["cancelled"])
        self.assertEqual(state.jobs["queued"].status, JobStatus.CANCELLED)

    def test_legacy_query_tokens_work_for_every_job_endpoint(self):
        for method, url, token in [
            ("GET", "/api/v1/results/owned", "owned-token"),
            ("GET", "/api/v1/results/owned/download", "owned-token"),
            ("GET", "/api/v1/jobs", "owned-token"),
            ("POST", "/api/v1/jobs/queued/cancel", "queued-token"),
        ]:
            with self.subTest(url=url):
                self.assertEqual(self.client.request(method, url, params={"access_token": token}).status_code, 200)

    def test_invalid_header_never_falls_back_to_valid_query_token(self):
        for header in ["Basic owned-token", "Bearer ", "Bearer wrong"]:
            with self.subTest(header=header):
                response = self.client.get("/api/v1/results/owned", params={"access_token": "owned-token"}, headers={"Authorization": header})
                self.assertEqual(response.status_code, 403)

    def test_authorized_missing_job_returns_404(self):
        response = self.client.get("/api/v1/results/missing", headers={"Authorization": "Bearer owned-token"})
        self.assertEqual(response.status_code, 404)


class StopWorker(BaseException):
    """Exit the infinite worker loop after a real queue is drained."""


class DrainQueue(queue.Queue):
    def get(self, *args, **kwargs):
        if self.empty():
            raise StopWorker()
        return super().get(*args, **kwargs)


class WorkerQueueTests(unittest.TestCase):
    def setUp(self):
        state.jobs.clear()
        state.cancelled_jobs.clear()

    def tearDown(self):
        state.jobs.clear()
        state.cancelled_jobs.clear()

    def drain(self, items, process):
        q = DrainQueue()
        for item in items:
            q.put(item)
        with patch.object(worker, "_process_job", side_effect=process) as mocked:
            with self.assertRaises(StopWorker):
                worker.run_worker_loop("cpu", q)
        self.assertEqual(q.unfinished_tasks, 0)
        return mocked

    def test_missing_job_does_not_break_next_job(self):
        state.jobs["next"] = make_job("next", status=JobStatus.QUEUED)
        mocked = self.drain(["missing", "next"], lambda **kwargs: None)
        self.assertEqual(mocked.call_args.kwargs["job_id"], "next")
        self.assertEqual(mocked.call_count, 1)

    def test_cancelled_job_cleans_file_and_processes_next_job(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cancelled.wav"
            path.write_bytes(b"test")
            cancelled = make_job("cancelled", status=JobStatus.CANCELLED)
            cancelled.temp_audio_path = str(path)
            state.jobs.update({"cancelled": cancelled, "next": make_job("next", status=JobStatus.QUEUED)})
            state.cancelled_jobs.add("cancelled")
            mocked = self.drain(["cancelled", "next"], lambda **kwargs: None)
            self.assertEqual(mocked.call_count, 1)
            self.assertFalse(path.exists())
            self.assertNotIn("cancelled", state.cancelled_jobs)

    def test_cancel_flag_skips_queued_job(self):
        state.jobs["cancelled"] = make_job("cancelled", status=JobStatus.QUEUED)
        state.cancelled_jobs.add("cancelled")
        mocked = self.drain(["cancelled"], lambda **kwargs: None)
        self.assertEqual(mocked.call_count, 0)
        self.assertNotIn("cancelled", state.cancelled_jobs)

    def test_processing_error_does_not_break_next_job(self):
        state.jobs.update({"broken": make_job("broken", status=JobStatus.QUEUED), "next": make_job("next", status=JobStatus.QUEUED)})
        def process(job_id, job_data):
            if job_id == "broken":
                raise RuntimeError("test failure")
        mocked = self.drain(["broken", "next"], process)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(state.jobs["broken"].status, JobStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
