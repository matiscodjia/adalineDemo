"""The runner and the HTTP API."""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from adalineDemo import chip
from adalineDemo.demo import runner, server


def test_run_returns_a_well_formed_result(tmp_path):
    r = runner.run(16, 8000, 0.004, 1.0, str(tmp_path))
    assert r["samples"] == 8000
    assert r["late"]["corr"] > r["early"]["corr"]
    assert r["host"]["us_per_sample"] > 0
    assert 0 < r["chip_margin"]
    assert all((tmp_path / Path(u).name).is_file() for u in r["audio"].values())
    assert len(r["waveform"]["d"]) <= 800


def test_run_audio_files_are_valid_wav_at_the_right_rate(tmp_path):
    r = runner.run(16, 8000, 0.004, 1.0, str(tmp_path))
    with wave.open(str(tmp_path / Path(r["audio"]["cancelled"]).name), "rb") as f:
        assert f.getframerate() == 8000 and f.getnchannels() == 1 and f.getnframes() == r["samples"]


def test_run_reports_progress():
    stages = []
    runner.run(16, 8000, 0.004, 1.0, "/tmp", progress=stages.append)
    assert any("Training" in s for s in stages) and any("Encoding" in s for s in stages)


def test_chip_verified_flag_follows_taps(tmp_path):
    assert runner.run(16, 8000, 0.004, 0.5, str(tmp_path))["chip_projection"]["verified"] is True
    assert runner.run(32, 8000, 0.004, 0.5, str(tmp_path))["chip_projection"]["verified"] is False


def test_validation():
    with pytest.raises(ValueError, match="taps"):
        runner.validate(17, 8000, 0.004, 1.0)
    with pytest.raises(ValueError, match="fs"):
        runner.validate(16, 12345, 0.004, 1.0)
    with pytest.raises(ValueError, match="lr"):
        runner.validate(16, 8000, 10.0, 1.0)
    with pytest.raises(ValueError, match="duration"):
        runner.validate(16, 8000, 0.004, 999)


@pytest.fixture()
def api(tmp_path):
    server.Handler.media_dir = tmp_path
    server.Handler.jobs = server.Jobs()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _post(port, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/run", json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def _wait(port, job):
    for _ in range(300):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/job/{job}") as response:
            data = json.load(response)
        if data["state"] != "running":
            return data
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_api_config(api):
    with urllib.request.urlopen(f"http://127.0.0.1:{api}/api/config") as r:
        cfg = json.load(r)
    assert cfg["taps_ladder"] == chip.TAPS_LADDER
    assert cfg["chip"]["verified_cycles"] == 575
    assert len(cfg["signal"]["f_signal"]) == 3


def test_api_run_lifecycle_and_single_job_at_a_time(api):
    body = {"taps": 16, "fs": 8000, "lr": 0.004, "duration_s": 1.0}
    status, data = _post(api, body)
    assert status == 202
    assert _post(api, body)[0] == 409
    result = _wait(api, data["job"])
    assert result["state"] == "done" and result["result"]["samples"] == 8000


def test_api_serves_audio_with_range(api):
    body = {"taps": 16, "fs": 8000, "lr": 0.004, "duration_s": 1.0}
    result = _wait(api, _post(api, body)[1]["job"])["result"]
    url = f"http://127.0.0.1:{api}" + result["audio"]["cancelled"]
    with urllib.request.urlopen(url) as r:
        full = r.read()
    request = urllib.request.Request(url, headers={"Range": "bytes=0-99"})
    with urllib.request.urlopen(request) as r:
        assert r.status == 206 and len(r.read()) == 100
    assert len(full) > 100


def test_api_rejects_bad_requests(api):
    good = {"taps": 16, "fs": 8000, "lr": 0.004, "duration_s": 1.0}
    assert _post(api, {**good, "taps": 17})[0] == 400
    assert _post(api, {**good, "fs": 1})[0] == 400
    conn = http.client.HTTPConnection("127.0.0.1", api)
    conn.request("GET", "/api/config", headers={"Host": "evil.example"})
    assert conn.getresponse().status == 403


def test_an_ip_address_as_host_is_refused_unless_lan_is_on(api):
    def status(host):
        conn = http.client.HTTPConnection("127.0.0.1", api)
        conn.request("GET", "/api/config", headers={"Host": host})
        return conn.getresponse().status

    assert status("192.168.1.27:8000") == 403 and status("localhost:8000") == 200
    server.Handler.lan = True
    try:
        assert status("192.168.1.27:8000") == 200
        assert status("evil.example") == 403
    finally:
        server.Handler.lan = False


def test_the_page_and_its_assets_are_served(api):
    for path in ("/", "/static/style.css", "/static/app.js"):
        with urllib.request.urlopen(f"http://127.0.0.1:{api}{path}") as r:
            assert r.status == 200 and len(r.read()) > 100
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"http://127.0.0.1:{api}/static/../server.py")
    assert err.value.code == 404
