"""Browser test of the viewer (Play mode): the frame number it shows must be the frame on screen, and the overlay
must draw that frame's boxes.

Drives headless Chrome (DevTools protocol) against the app, on a generated H.264 test video whose frames show their
own index as a row of black / white squares (binary), so the picture on screen can be read back exactly. Each frame
has one box whose x position depends on the frame index, so the overlay can be read back too.

Also: changing a setting in the settings panel re-runs filtering and shows the new results.

Skipped without Chrome (or set CHROME_PATH). Leave it out with: pytest -m "not browser"
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from swim.app.server import create_app
from swim.pipeline.config import PipelineConfig
from swim.stages.detection.types import BBox, Detection, FrameDetections, VideoDetections
from swim.utils.video import video_writer

websocket = pytest.importorskip("websocket", reason="needs websocket-client")
uvicorn = pytest.importorskip("uvicorn")

pytestmark = pytest.mark.browser

FPS, WIDTH, HEIGHT, FRAMES = 30.0, 640, 360, 90
BITS, SQUARE, SQUARE_X0, SQUARE_STEP, SQUARE_Y = 7, 64, 24, 84, 40  # frame index as squares, bit k at x0 + k * step
BOX_X0, BOX_STEP, BOX_Y1, BOX_Y2 = 20, 6, 200, 330  # frame i's box starts at x = BOX_X0 + i * BOX_STEP
VIDEO_NAME = "test_video"
SETTINGS_VIDEO = "test_video_settings"  # a separate results folder: its test re-runs filtering

# Read back, in the page: the frame index shown in the video, the frame whose box the overlay draws, the frame label
PAGE_HELPERS = f"""
window.__test = {{
  screenFrame() {{
    const v = document.getElementById('video');
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext('2d', {{ willReadFrequently: true }});
    ctx.drawImage(v, 0, 0);
    let n = 0;
    for (let k = 0; k < {BITS}; k++) {{
      const x = {SQUARE_X0} + k * {SQUARE_STEP} + {SQUARE // 2};
      if (ctx.getImageData(x, {SQUARE_Y + SQUARE // 2}, 1, 1).data[0] > 128) n |= 1 << k;
    }}
    return n;
  }},
  overlayFrame() {{
    // First red pixel on a row through the middle of the box = the box's left edge
    const c = document.getElementById('overlay');
    const s = c.width / {WIDTH};
    const row = c.getContext('2d').getImageData(0, Math.round({(BOX_Y1 + BOX_Y2) // 2} * s), c.width, 1).data;
    for (let x = 0; x < c.width; x++) {{
      const [r, g, b, a] = row.slice(4 * x, 4 * x + 4);
      if (a > 200 && r > 200 && g < 80 && b < 80) return Math.round((x / s - {BOX_X0}) / {BOX_STEP});
    }}
    return null;
  }},
  label() {{
    const m = (document.getElementById('position')?.textContent || '').match(/frame (\\d+)/);
    return m ? Number(m[1]) : null;
  }},
}};
"""


def write_test_video(path: Path) -> None:
    with video_writer(path, FPS, WIDTH, HEIGHT, codec="avc1") as writer:
        for i in range(FRAMES):
            frame = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
            for k in range(BITS):
                x = SQUARE_X0 + k * SQUARE_STEP
                frame[SQUARE_Y:SQUARE_Y + SQUARE, x:x + SQUARE] = 255 if i >> k & 1 else 0
            writer.write(frame)


def write_test_results(folder: Path, video: Path) -> None:
    frames = [
        FrameDetections(i, i / FPS, [Detection(BBox(BOX_X0 + i * BOX_STEP, BOX_Y1, BOX_X0 + i * BOX_STEP + 60, BOX_Y2), 0.9)])
        for i in range(FRAMES)
    ]
    VideoDetections(str(video), FPS, WIDTH, HEIGHT, "test", frames).to_json(folder / "1_detections.json")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_chrome() -> str | None:
    candidates = [
        os.environ.get("CHROME_PATH"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("viewer")
    write_test_video(root / "input" / "test.mp4")
    write_test_results(root / "output" / VIDEO_NAME, root / "input" / "test.mp4")
    write_test_results(root / "output" / SETTINGS_VIDEO, root / "input" / "test.mp4")
    settings = PipelineConfig().with_changes({"detection": {"threshold": 0.5}})
    settings.to_json(root / "output" / SETTINGS_VIDEO / "config.json")

    port = free_port()
    app_server = uvicorn.Server(uvicorn.Config(create_app(root / "output"), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=app_server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not app_server.started:
        assert time.time() < deadline, "app server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    app_server.should_exit = True
    thread.join(timeout=5)


class Chrome:
    """Minimal DevTools protocol client: send commands, keep the page's exceptions."""

    def __init__(self, ws):
        self.ws = ws
        self.last_id = 0
        self.exceptions = []

    def cdp(self, method: str, **params):
        self.last_id += 1
        self.ws.send(json.dumps({"id": self.last_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                details = message["params"]["exceptionDetails"]
                self.exceptions.append(details.get("exception", {}).get("description") or details.get("text"))
            if message.get("id") == self.last_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message["result"]

    def js(self, expression: str):
        result = self.cdp("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")


@pytest.fixture(scope="module")
def chrome(tmp_path_factory):
    path = find_chrome()
    if path is None:
        pytest.skip("Chrome not found (set CHROME_PATH)")
    profile = tmp_path_factory.mktemp("chrome-profile")
    process = subprocess.Popen(
        [path, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}", "--window-size=1440,900",
         "--autoplay-policy=no-user-gesture-required", "--mute-audio", "--hide-scrollbars", "--no-first-run",
         "--no-default-browser-check", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # With port 0 Chrome picks a free port and writes it to DevToolsActivePort
        port_file = profile / "DevToolsActivePort"
        deadline = time.time() + 20
        while not (port_file.exists() and port_file.read_text().strip()):
            assert time.time() < deadline, "Chrome did not start"
            time.sleep(0.1)
        port = port_file.read_text().split()[0]
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
        page = next(t for t in targets if t["type"] == "page")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
        browser = Chrome(ws)
        browser.cdp("Runtime.enable")
        browser.cdp("Page.enable")
        browser.cdp("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
        browser.cdp("Page.addScriptToEvaluateOnNewDocument", source=PAGE_HELPERS)
        yield browser
        ws.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


class Viewer:
    def __init__(self, chrome: Chrome, base_url: str):
        self.chrome = chrome
        self.base_url = base_url

    def label(self) -> int | None:
        try:
            return self.chrome.js("window.__test ? window.__test.label() : null")
        except RuntimeError:  # page still loading
            return None

    def wait_label(self, expected: int, timeout: float = 10) -> int | None:
        """Wait until the viewer shows frame `expected` and the video is not seeking; return the last label seen."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            label = self.label()
            seeking = self.chrome.js("document.getElementById('video').seeking")
            if label == expected and not seeking:
                return label
            time.sleep(0.05)
        return self.label()

    def wait_for(self, expression: str, done, timeout: float = 15):
        """Evaluate `expression` until `done(value)`; return the last value."""
        deadline = time.time() + timeout
        value = self.chrome.js(expression)
        while not done(value) and time.time() < deadline:
            time.sleep(0.05)
            value = self.chrome.js(expression)
        return value

    def open(self, frame: int, video: str = VIDEO_NAME) -> None:
        self.chrome.cdp("Page.navigate", url=f"{self.base_url}/?video={video}&frame={frame}")
        if self.wait_label(frame) != frame:
            error = self.chrome.js("document.getElementById('video').error?.code ?? null")
            if error == 4:
                pytest.skip("this Chrome can't play H.264 (Chromium without proprietary codecs?)")

    def shown(self) -> tuple[int | None, int, int | None]:
        """(frame label, frame index in the picture on screen, frame whose box the overlay draws)"""
        return tuple(self.chrome.js("[__test.label(), __test.screenFrame(), __test.overlayFrame()]"))

    def key(self, key: str, code: int, shift: bool = False) -> None:
        for event in ("rawKeyDown", "keyUp"):
            self.chrome.cdp("Input.dispatchKeyEvent", type=event, key=key, code=key, windowsVirtualKeyCode=code,
                            modifiers=8 if shift else 0)

    def click_timeline(self, frame: int) -> None:
        left, top, width, height = self.chrome.js(
            "(() => { const r = document.getElementById('timeline').getBoundingClientRect();"
            " return [r.left, r.top, r.width, r.height]; })()"
        )
        x, y = left + width * (frame + 0.5) / FRAMES, top + height / 2
        for event in ("mousePressed", "mouseReleased"):
            self.chrome.cdp("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)

    def play_and_sample(self, count: int) -> list[list]:
        """Play; at each new frame on screen (requestVideoFrameCallback, after the viewer's own) record `shown()`."""
        return self.chrome.js(f"""new Promise((resolve) => {{
          const v = document.getElementById('video');
          const out = [];
          const sample = () => {{
            out.push([__test.label(), __test.screenFrame(), __test.overlayFrame()]);
            if (out.length < {count} && !v.ended) v.requestVideoFrameCallback(sample);
            else {{ v.pause(); resolve(out); }}
          }};
          v.requestVideoFrameCallback(sample);
          v.play();
        }})""")


@pytest.fixture
def viewer(chrome, server):
    chrome.exceptions.clear()
    yield Viewer(chrome, server)
    assert chrome.exceptions == [], "exceptions in the page"


@pytest.mark.parametrize("frame", [0, 1, 37, FRAMES - 1])
def test_opening_at_a_frame_shows_that_frame(viewer, frame):
    viewer.open(frame)

    assert viewer.shown() == (frame, frame, frame)


def test_stepping_with_keys(viewer):
    viewer.open(40)
    steps = [("ArrowRight", 39, False, 41), ("ArrowRight", 39, False, 42), ("ArrowLeft", 37, True, 32),
             ("ArrowLeft", 37, False, 31), ("Home", 36, False, 0), ("End", 35, False, FRAMES - 1)]

    for key, code, shift, expected in steps:
        viewer.key(key, code, shift)
        viewer.wait_label(expected)
        assert viewer.shown() == (expected, expected, expected), f"after {'Shift+' if shift else ''}{key}"


def test_timeline_click_jumps_to_that_frame(viewer):
    viewer.open(0)

    viewer.click_timeline(63)
    viewer.wait_label(63)

    assert viewer.shown() == (63, 63, 63)


def test_label_and_overlay_follow_the_video_while_playing(viewer):
    viewer.open(10)

    samples = viewer.play_and_sample(20)

    assert len(samples) >= 10
    assert all(label == screen == overlay for label, screen, overlay in samples), samples
    assert samples[-1][1] > samples[0][1]  # the video advanced


def test_changing_a_setting_reruns_and_shows_the_new_results(viewer):
    js = viewer.chrome.js
    viewer.open(5, video=SETTINGS_VIDEO)
    assert viewer.shown() == (5, 5, 5)  # detections layer: the frame's red box
    assert js("!document.getElementById('settings-hint').hidden")  # min_score 0.1 < detection threshold 0.5

    js("""(() => {
      const input = document.getElementById('setting-filtering-min_score');
      input.value = '0.95';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      document.getElementById('apply-settings').click();
    })()""")
    status = viewer.wait_for("document.getElementById('settings-status').textContent",
                             lambda text: text.startswith(("Done", "Re-run failed")))

    assert status.startswith("Done"), status
    assert js("document.getElementById('settings-hint').hidden")
    assert js("document.getElementById('frame-title').textContent") == "Frame 5 · filtered"  # new layer shown
    assert viewer.shown() == (5, 5, None)  # every box scores 0.9 < 0.95: rejected, drawn gray, no red box
    assert "min_score" in js("document.getElementById('frame-boxes').textContent")
