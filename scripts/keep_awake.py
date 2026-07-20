"""Keep Streamlit Community Cloud apps awake.

Streamlit Community Cloud puts an app to sleep after ~12h with no real
browser sessions (a websocket connection from an actual page load) -- plain
HTTP pings (e.g. UptimeRobot) don't count, since they never open a session.

This script opens each target app in headless Chromium, clicks the "wake up"
button if the app is asleep, and polls for the app's own content to render
inside its iframe before declaring success. Confirming a real heading
element (not just a 200 status) is what proves a genuine session was
established.

Architecture note (found by inspecting the live apps): when an app is
awake, Streamlit Community Cloud serves it inside a same-origin iframe whose
src ends in "/~/+/". A separate, unrelated statuspage.io status-embed iframe
is also present on every load and must be excluded when searching for the
app's heading. When an app is asleep, the wake button lives directly on the
top-level page (no iframe).
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

WAKE_BUTTON_PATTERN = re.compile(r"get this app back up", re.IGNORECASE)
EXCLUDED_FRAME_HOST = "statuspage.io"

NAV_TIMEOUT_MS = 60_000  # generous: a cold app's initial static page can be slow
WAKE_CLICK_SETTLE_MS = 3_000
RENDER_POLL_TIMEOUT_S = 240  # observed empirically: a cold wake can take several minutes
RENDER_POLL_INTERVAL_S = 4
SESSION_HOLD_S = 12  # keep the tab open so the session isn't torn down instantly


@dataclass
class Target:
    name: str
    url: str
    heading: str  # exact text of the real app's own <h1>, proof it rendered


TARGETS = [
    Target(
        name="course-recommender-system",
        url="https://course-recommender-system-dlrwwyqrh9vvstfxaf79wn.streamlit.app/",
        heading="Course Recommender",
    ),
    Target(
        name="awc-operations-dashboard",
        url="https://awc-operations-dashboard.streamlit.app",
        heading="AWC Operations Dashboard",
    ),
    Target(
        name="cnn-vit-land-classification",
        url="https://cnn-vit-land-classification.streamlit.app/",
        heading="Satellite Land Classification (CNN, ONNX)",
    ),
]


def app_frame_heading(page: Page, expected_heading: str):
    """Return the matching heading text from the real app's iframe, or None."""
    for frame in page.frames:
        if EXCLUDED_FRAME_HOST in frame.url:
            continue
        try:
            h1 = frame.locator("h1", has_text=expected_heading)
            if h1.count() > 0 and h1.first.is_visible():
                return h1.first.inner_text(timeout=2000)
        except Exception:
            continue
    return None


def wait_for_app_render(page: Page, expected_heading: str, timeout_s: int) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = app_frame_heading(page, expected_heading)
        if text:
            return text
        page.wait_for_timeout(RENDER_POLL_INTERVAL_S * 1000)
    raise TimeoutError(f"'{expected_heading}' never rendered within {timeout_s}s")


def process_target(browser, target: Target, screenshot_dir: Path) -> bool:
    """Wake/verify one target. Never raises -- failures are caught and logged."""
    print(f"\n=== {target.name} ({target.url}) ===", flush=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1200})
    try:
        try:
            page.goto(target.url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            # Streamlit keeps some connections open even on a healthy load, so
            # networkidle can time out on its own. Fall back to "load" and let
            # the render-poll below do the real verification.
            print("  networkidle timed out, falling back to load-state")
            page.wait_for_load_state("load", timeout=NAV_TIMEOUT_MS)

        wake_button = page.get_by_role("button", name=WAKE_BUTTON_PATTERN)
        was_asleep = wake_button.count() > 0
        if was_asleep:
            print("  status: asleep -- clicking wake button")
            wake_button.first.click()
            page.wait_for_timeout(WAKE_CLICK_SETTLE_MS)
        else:
            print("  status: already awake (no wake button found)")

        heading_text = wait_for_app_render(page, target.heading, RENDER_POLL_TIMEOUT_S)
        print(f"  rendered: found heading '{heading_text}'")
        print(f"  holding session open for {SESSION_HOLD_S}s so it registers")
        page.wait_for_timeout(SESSION_HOLD_S * 1000)

        verb = "woken and confirmed alive" if was_asleep else "confirmed already alive"
        print(f"  RESULT: {verb}")
        return True

    except Exception as exc:
        print(f"  FAILURE: {exc}")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshot_dir / f"{target.name}-failure.png"
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  saved failure screenshot: {screenshot_path}")
        except Exception as shot_exc:
            print(f"  could not save screenshot: {shot_exc}")
        return False
    finally:
        page.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot-dir", default="keep_awake_screenshots")
    args = parser.parse_args()
    screenshot_dir = Path(args.screenshot_dir)

    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for target in TARGETS:
                results[target.name] = process_target(browser, target, screenshot_dir)
        finally:
            browser.close()

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
