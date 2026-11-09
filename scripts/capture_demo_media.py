"""Regenerate docs/media from the demo as it really runs.

The viewer captures drive the actual viewer/index.html in headless Chromium (WebGL via SwiftShader) with the files
from runs/sphere and runs/sphere_rl, so `make demo` must have been run first. The terminal GIF is rendered from a
recorded transcript of `make demo`; every line is the real output, only the typing and the pauses are staged.

Needs playwright (not a project dependency) and ffmpeg:
    uv venv /tmp/pw && VIRTUAL_ENV=/tmp/pw uv pip install playwright && /tmp/pw/bin/playwright install chromium
    script -q /tmp/make_demo.txt make demo          # record the real run
    /tmp/pw/bin/python scripts/capture_demo_media.py /tmp/make_demo.txt /tmp/media   # writes docs/media/*
Optional third argument: `terminal` or `viewer` to redo one half.
"""
from __future__ import annotations

import asyncio
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "media"
VIEWER = f"file://{ROOT}/viewer/index.html"
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
VIEWPORT = {"width": 1280, "height": 800}
PANEL_W = 300  # width of the viewer's side panel; the canvas is to its right
GIF_FILTER = ("fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4")


class Frames:
    """PNG frames plus a concat list with per-frame durations, the input ffmpeg needs for a paced GIF."""

    def __init__(self, folder: Path):
        self.folder = folder
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        self.durations: list[float] = []

    async def shot(self, page, seconds: float) -> None:
        await page.screenshot(path=self.folder / f"f{len(self.durations):04d}.png")
        self.durations.append(seconds)

    def write_list(self) -> None:
        body = "".join(f"file 'f{j:04d}.png'\nduration {d}\n" for j, d in enumerate(self.durations))
        last = len(self.durations) - 1
        (self.folder / "list.txt").write_text(body + f"file 'f{last:04d}.png'\n")

    def to_gif(self, out: Path) -> None:
        self.write_list()
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(self.folder / "list.txt"), "-vf", GIF_FILTER, str(out)], check=True)
        print("wrote", out, out.stat().st_size // 1024, "KB,", len(self.durations), "frames")


# ---------------------------------------------------------------- viewer

async def load_run(page, run: str) -> None:
    await page.goto(VIEWER)
    await page.wait_for_timeout(2500)
    for input_id, name in (("#buildFile", "build.json"), ("#voxelFile", "voxels.json"), ("#stabFile", "stability.json")):
        await page.set_input_files(input_id, str(ROOT / "runs" / run / name))
    await page.wait_for_timeout(1200)


async def orbit(page, dx: float, dy: float, steps: int = 12) -> None:
    """Drag on the canvas the way a user would; OrbitControls does the rest."""
    cx, cy = PANEL_W + (VIEWPORT["width"] - PANEL_W) / 2, VIEWPORT["height"] / 2
    await page.mouse.move(cx, cy)
    await page.mouse.down()
    for i in range(1, steps + 1):
        await page.mouse.move(cx + dx * i / steps, cy + dy * i / steps)
        await page.wait_for_timeout(20)
    await page.mouse.up()
    await page.wait_for_timeout(150)


async def set_step(page, k: int) -> None:
    await page.evaluate("k => { const s = document.getElementById('step'); s.value = k; s.dispatchEvent(new Event('input')); }", k)
    await page.wait_for_timeout(60)


async def capture_viewer(p, scratch: Path) -> None:
    browser = await p.chromium.launch(args=CHROMIUM_ARGS)
    page = await browser.new_page(viewport=VIEWPORT, color_scheme="light")
    for run, name in (("sphere", "viewer_sphere_greedy"), ("sphere_rl", "viewer_sphere_rl")):
        await load_run(page, run)
        await orbit(page, -120, 60)
        await page.screenshot(path=OUT / f"{name}.png")
    await load_run(page, "sphere")
    await orbit(page, -120, 60)
    frames = Frames(scratch / "frames_viewer")
    n = int(await page.evaluate("document.getElementById('step').max"))
    await set_step(page, 0)
    await frames.shot(page, 0.8)
    for k in range(1, n + 1):
        await set_step(page, k)
        await frames.shot(page, 0.09 if k < n else 1.2)
    for _ in range(24):
        await orbit(page, -40, 0, steps=3)
        await frames.shot(page, 0.08)
    await frames.shot(page, 1.0)
    await browser.close()
    frames.to_gif(OUT / "viewer_sphere_steps.gif")


# ---------------------------------------------------------------- terminal

ROWS = 44  # lines visible in the rendered terminal window; older lines scroll off like a real terminal
TERM_CSS = """
html,body{margin:0;background:#0f1117}
.win{width:1200px;margin:16px;background:#1b1e27;border-radius:10px;box-shadow:0 8px 30px #0008;overflow:hidden}
.bar{height:34px;background:#2a2e3a;display:flex;align-items:center;padding:0 12px;gap:8px;color:#9aa;font:13px -apple-system,system-ui}
.dot{width:12px;height:12px;border-radius:6px}
pre{margin:0;padding:14px 18px;color:#e6e6e6;font:14px/1.42 Menlo,"SF Mono",Consolas,monospace;white-space:pre-wrap;HEIGHT}
.p{color:#7fd1ff}.c{color:#fff;font-weight:600}.k{color:#9ecbff}.s{color:#7ee787}.e{color:#ff8a80}.m{color:#8b949e}
.cur{display:inline-block;width:8px;height:16px;background:#e6e6e6;vertical-align:-3px}
"""
PROMPT = '<span class="p">~/brickforge-test $</span>'


def clean_transcript(txt: str) -> list[str]:
    """script(1) records carriage returns, the terminal's ^D echo and the backspaces that erase it."""
    out: list[str] = []
    for ch in txt.replace("\r", ""):
        if ch == "\x08":
            if out:
                out.pop()
        elif ch == "\n" or ch >= " ":
            out.append(ch)
    lines = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", ln) for ln in "".join(out).split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def colour(line: str) -> str:
    h = html.escape(line)
    if line.startswith("uv run"):
        return f'<span class="m">{h}</span>'
    if line.startswith("---"):
        return f'<span class="k">{h}</span>'
    if '"status": "validated_candidate"' in line or '"label": "stable' in line:
        return f'<span class="s">{h}</span>'
    if '"status": "failed"' in line or "Error 1 (ignored)" in line or '"error_code"' in line:
        return f'<span class="e">{h}</span>'
    return h


def term_html(cmd_typed: str, lines: list[str], cursor: bool = True, rows: int = ROWS, fixed_height: bool = True) -> str:
    visible = lines[-rows:] if len(lines) > rows else lines
    body = f'{PROMPT} <span class="c">{html.escape(cmd_typed)}</span>'
    if visible:
        head = body + "\n" if len(lines) < rows else ""
        body = head + "\n".join(colour(ln) for ln in visible)
    if cursor:
        body += ' <span class="cur"></span>' if not visible else f'\n{PROMPT} <span class="cur"></span>'
    css = TERM_CSS.replace("HEIGHT", "height:880px;overflow:hidden" if fixed_height else "min-height:880px")
    bar = ('<div class="bar"><span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span>'
           '<span class="dot" style="background:#28c840"></span>&nbsp; Terminal — make demo</div>')
    return f'<html><head><meta charset="utf-8"><style>{css}</style></head><body><div class="win">{bar}<pre>{body}</pre></div></body></html>'


# where the real run paused, and for how long the GIF holds before revealing the next block
STAGES = [
    (lambda ln: ln.startswith("uv run brickforge generate") and ln.endswith("runs/sphere"), 0.9),
    (lambda ln: ln.startswith("--- intentional failure"), 2.6),
    (lambda ln: ln.startswith("--- the same sphere"), 1.4),
    (lambda ln: ln.startswith("--- done"), 2.4),
]


async def capture_terminal(p, transcript: Path, scratch: Path) -> None:
    lines = clean_transcript(transcript.read_text())
    browser = await p.chromium.launch()
    page = await browser.new_page(viewport={"width": 1232, "height": 960}, color_scheme="dark")
    frames = Frames(scratch / "frames_term")

    async def shot(markup: str, seconds: float) -> None:
        await page.set_content(markup)
        await frames.shot(page, seconds)

    cmd = "make demo"
    await shot(term_html("", []), 0.7)
    for n in range(1, len(cmd) + 1):
        await shot(term_html(cmd[:n], []), 0.12)
    await shot(term_html(cmd, [], cursor=False), 0.5)
    shown: list[str] = []
    for pred, pause in STAGES:
        end = next((i + 1 for i in range(len(shown), len(lines)) if pred(lines[i])), len(lines))
        await shot(term_html(cmd, shown, cursor=False), pause)
        shown = lines[:end]
    await shot(term_html(cmd, lines), 3.0)
    await page.set_content(term_html(cmd, lines, rows=10_000, fixed_height=False))
    await page.screenshot(path=OUT / "terminal_make_demo.png", full_page=True)
    await browser.close()
    frames.to_gif(OUT / "terminal_make_demo.gif")


async def main() -> None:
    transcript, scratch = Path(sys.argv[1]), Path(sys.argv[2])
    only = sys.argv[3] if len(sys.argv) > 3 else "all"
    scratch.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        if only in ("all", "terminal"):
            await capture_terminal(p, transcript, scratch)
        if only in ("all", "viewer"):
            await capture_viewer(p, scratch)


if __name__ == "__main__":
    asyncio.run(main())
