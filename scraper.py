#!/usr/bin/env python3
"""
==============================================================================
  IMAGE DATASET SCRAPER - domain-agnostic image collector for CV datasets
==============================================================================

One input, a full image dataset out. You type what you want (e.g. "walk
compactor"); an LLM (Groq / Llama 3.3 70B) expands it into many diverse search
queries; those queries are run across multiple free image sources; results are
quality-filtered, de-duplicated (exact + perceptual hash), then verified by an
AI vision model (Llama-4 Scout via the Groq API) that keeps only images matching
your description. A markdown report is generated at the end.

Sources (all free):
    - Pixabay            (API key)
    - Pexels             (API key; curated real photos)
    - Unsplash           (API key; curated real photos)
    - DuckDuckGo Images  (no key)
    - Openverse          (no key, optional token; 800M+ Creative Commons images)
    - Wikimedia Commons  (no key)

Usage:
    python scraper.py                      # interactive
    python scraper.py "golden retriever"   # one-shot
    python scraper.py "golden retriever" --target 500 --per-query 40

Keys live in the .env file next to this script. Only GROQ + PIXABAY are
required; everything else degrades gracefully.
==============================================================================
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import errno
import hashlib
import io
import json
import os
import shutil
import random
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Windows console: force UTF-8 so the pretty output never crashes on encoding.
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Third-party deps (all installed via requirements.txt) ----------------
try:
    import requests
    from dotenv import load_dotenv
    from PIL import Image
    import imagehash
    from rich.console import Console
    from rich.table import Table
    from rich.columns import Columns
    from rich import box
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}\nRun:  pip install -r requirements.txt")
    sys.exit(1)

# Silence Pillow's chatty "Palette images with Transparency..." UserWarning;
# we handle mode conversion ourselves when saving.
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Pillow safety ceiling: the MAXIMUM pixel count we will even attempt to decode,
# to defend against "decompression bomb" attacks (a tiny file that expands to a
# gigantic image and exhausts memory). This is an UPPER limit, not a target -
# normal photos are far below it: 4K = ~8 MP, 8K = ~33 MP, a 100 MP phone shot
# is still under this. 178 MP ~= a 13350 x 13350 image. Anything bigger is
# almost certainly malicious or broken, so we refuse it.
Image.MAX_IMAGE_PIXELS = 178_000_000

# Cap the width so the UI looks the same on a full-screen monitor as on a small
# window: dividers and bars never sprawl across the whole screen, and the live
# progress region never miscalculates wrapped lines (which caused duplicated
# output on wide terminals).
_TERM_WIDTH = shutil.get_terminal_size((90, 25)).columns
console = Console(width=min(_TERM_WIDTH, 90))
HERE = Path(__file__).resolve().parent
APP_NAME = "SCRAPR"
APP_TAGLINE = "Image Dataset Builder for Computer Vision"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

WIKI_UA = "ImageDatasetScraper/1.0 (CV research; contact: local user)"

# --- AI vision verification (Groq multimodal) ------------------------------
# A real, working vision model on the user's Groq key. It looks at every
# downloaded image and decides whether it actually matches what the user asked
# for (right subject, and real-photo-vs-illustration if they asked).
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MAXDIM = 384          # downscale before sending (~470 tokens/image, well under the cap)
VISION_HARD_CAP = 300        # never verify more than this many images in one run
VISION_MIN_TOKENS = 1500     # if fewer tokens than this remain this minute, wait for refill


# Recommended defaults + the safe (min, max) range for each tunable setting.
# Used by the CLI flags, the interactive prompts, and the report - one source
# of truth so they can never disagree.
DEFAULTS = {"target": 400, "queries": 18, "per_query": 30, "min_dim": 300, "format": "jpg"}
LIMITS = {"target": (1, 2000), "queries": (1, 30), "per_query": (3, 50), "min_dim": (50, 1024)}

SETTING_NOTES = {
    "target": "stop once this many unique images are saved",
    "queries": "different search phrasings the AI generates",
    "per_query": "images requested from each source per search",
    "min_dim": "smallest allowed width and height, in pixels",
    "format": "jpg = uniform & viewable, png = lossless, original = as downloaded",
}


@dataclass
class Config:
    description: str                         # the user's full description of what they need
    subject: str = ""                        # short label (folder name / display); set by the LLM
    target: int = DEFAULTS["target"]        # desired number of unique images
    per_query: int = DEFAULTS["per_query"]  # images requested per source per query
    num_queries: int = DEFAULTS["queries"]  # how many query variations the LLM makes
    min_dim: int = DEFAULTS["min_dim"]      # minimum width AND height in px
    min_bytes: int = 5 * 1024               # discard files smaller than this
    hash_cutoff: int = 4                    # perceptual-hash distance for "duplicate"
    workers: int = 12                       # parallel download threads
    image_format: str = DEFAULTS["format"]  # "jpg" | "png" | "original"
    sources: tuple | None = None            # subset of source keys, or None = all
    verify: bool = True                      # AI-vision check each image against the description
    out_dir: Path = field(default_factory=lambda: HERE / "scraped_images")

    @property
    def label(self) -> str:
        """Short human label for folders/headlines."""
        return self.subject or self.description


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip BOM, zero-width and control characters from user/terminal input."""
    for junk in ("﻿", "​", "‌", "‍"):
        text = text.replace(junk, "")
    text = "".join(ch for ch in text if ch == "\t" or ord(ch) >= 0x20)
    return text.strip()


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", clean_text(text).lower()).strip()
    s = re.sub(r"[\s_-]+", "_", s)
    return s or "dataset"


def ua() -> str:
    return random.choice(USER_AGENTS)


def polite_sleep(a: float = 0.4, b: float = 1.3) -> None:
    time.sleep(random.uniform(a, b))


# ---------------------------------------------------------------------------
# One consistent colour scheme + symbols used everywhere in the UI.
# ---------------------------------------------------------------------------
class S:
    OK = "green"
    BAD = "red"
    INFO = "cyan"
    MUTE = "grey58"
    HEAD = "bold cyan"
    ACCENT = "bold white"
    TICK = "[green]✓[/]"
    CROSS = "[red]✗[/]"
    DOT = "[grey58]•[/]"


# ---------------------------------------------------------------------------
# Simple, terminal-safe progress: a single line updated in place with carriage
# return. Deliberately NOT rich's Live/Progress, which redraws multiple lines
# and duplicates output on some Windows consoles.
# ---------------------------------------------------------------------------

def progress_line(text: str) -> None:
    width = max(20, console.width - 1)
    sys.stdout.write("\r  " + text[: width - 2].ljust(width - 2))
    sys.stdout.flush()


def progress_done() -> None:
    """Erase the in-place progress line so the next print starts clean."""
    sys.stdout.write("\r" + " " * console.width + "\r")
    sys.stdout.flush()


def http_get(url, *, params=None, headers=None, timeout=25, retries=2, backoff=1.5):
    """GET with automatic retry on transient failures (429 rate-limit, 5xx).

    Returns the Response (any status) or raises after exhausting retries on
    network errors. Callers still check status_code themselves.
    """
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise
    raise RuntimeError("http_get: retries exhausted")  # pragma: no cover


# ---------------------------------------------------------------------------
# A discovered image candidate (a URL we may download later)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    url: str
    source: str
    query: str


# ---------------------------------------------------------------------------
# Layer 1 - LLM query expansion (Groq / Llama 3.3 70B)
# ---------------------------------------------------------------------------

# Models tried in order. If the first fails (down, rate-limited, deprecated),
# the next is used automatically. Both are current Groq production models.
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

SYSTEM_PROMPT = """You turn a user's description of the images they need into web image-search queries for building a computer-vision dataset.

You MUST reply with ONE JSON object of EXACTLY this shape and nothing else:
{"subject": "<2-4 word label>", "queries": ["query one", "query two", ...]}

"subject" is a short label for the core thing being collected (used as a folder name).

Rules for "queries":
- EVERY query must be about the EXACT subject the user described. NEVER drift to related-but-different things. (If the subject is "golden retriever", never output other breeds like "labrador" or generic "dog". If it is "solar panel installation", never output unrelated electrical or roofing topics.)
- Cover the specific aspects, stages, angles and contexts the user mentions. If they mention a process with stages, make a query for each stage.
- Carry the user's constraints into the keywords (e.g. if they want real photos at a construction site, add words like "site", "construction", "real", "photo"; if they want close-ups, add "close up"). These bias search engines toward the right results.
- 2 to 7 words each. Concrete search keywords, NOT full sentences.
- No duplicates, no numbering, no explanations.

Example description: "solar panel installation on residential rooftops, all stages, real photographs only, no diagrams"
Example output: {"subject": "solar panel installation", "queries": ["solar panel rooftop installation photo","residential solar panel mounting","solar panel roof racking system","solar panel wiring connection real","rooftop solar array completed","solar panel installer working roof","residential solar panel before after","solar panel roof mounting close up","home solar panel installation progress","solar panels residential neighborhood aerial"]}
"""


def generate_queries(cfg: Config):
    """Ask Groq to turn the user's description into a subject + tight queries.

    Returns (subject, queries, notes). notes is a list of (message, style) tuples
    the caller prints. On any failure it falls back to the raw description.
    """
    from groq import Groq

    notes: list[tuple[str, str]] = []
    api_key = os.getenv("GROQ_API_KEY")
    fallback_subject = " ".join(cfg.description.split()[:5])
    fallback = (fallback_subject, _dedupe_keep([cfg.description]))

    if not api_key:
        notes.append(("No GROQ_API_KEY found - searching your exact words (no AI expansion).", "yellow"))
        return (*fallback, notes)

    client = Groq(api_key=api_key)
    user_msg = (
        f'The user needs images described as: "{cfg.description}".\n'
        f'Produce a short "subject" label and about {cfg.num_queries} precise, on-subject queries. '
        f'Reply with the JSON object only.'
    )

    last_err = None
    for i, model in enumerate(GROQ_MODELS):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.6,
                max_tokens=900,
                response_format={"type": "json_object"},  # forces valid JSON
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            subject, queries = _extract_subject_and_queries(raw)
            queries = _validate_queries(queries)

            # Guardrail: too few usable queries -> let the next model try.
            if len(queries) < 3:
                raise ValueError(f"only {len(queries)} valid queries returned")

            subject = (subject or fallback_subject).strip()
            queries = _dedupe_keep(queries)[: cfg.num_queries + 2]

            if i > 0:
                notes.append((f"Primary model unavailable - used fallback model '{model}'.", "yellow"))
            return subject, queries, notes
        except Exception as exc:
            last_err = exc
            notes.append((f"  AI model '{model}' failed: {str(exc)[:90]}", "yellow"))
            continue

    notes.append((f"All AI models failed ({str(last_err)[:90]}) - searching your exact words.", "red"))
    return (*fallback, notes)


def _extract_subject_and_queries(raw: str) -> tuple[str, list[str]]:
    """Pull (subject, queries) out of an LLM JSON response, robustly."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    subject = ""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            subject = str(data.get("subject", "") or "")
            for key in ("queries", "query", "items", "list", "results"):
                if isinstance(data.get(key), list):
                    return subject, [str(x) for x in data[key]]
            for v in data.values():
                if isinstance(v, list):
                    return subject, [str(x) for x in v]
        if isinstance(data, list):
            return subject, [str(x) for x in data]
    except (json.JSONDecodeError, TypeError):
        pass

    # Salvage: a [...] array anywhere in the text.
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return subject, [str(x) for x in data]
        except json.JSONDecodeError:
            pass

    lines = [re.sub(r'^[\s\-\*\d\.\)"\']+|["\']+$', "", ln).strip() for ln in raw.splitlines()]
    return subject, [ln for ln in lines if ln]


def _validate_queries(queries: list[str]) -> list[str]:
    """Keep only sane, keyword-style queries (drop junk / sentences / empties)."""
    out = []
    for q in queries:
        q = re.sub(r"\s+", " ", str(q)).strip().strip("\"'")
        if not q:
            continue
        words = q.split()
        if len(words) > 8 or len(q) > 80:   # a sentence, not a keyword query
            continue
        out.append(q)
    return out


def _dedupe_keep(items: list[str]) -> list[str]:
    """Case-insensitive de-dupe that preserves order."""
    seen, out = set(), []
    for q in items:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(q.strip())
    return out


# ---------------------------------------------------------------------------
# Layer 2 - Source collectors  (each returns a list[Candidate], never raises)
# ---------------------------------------------------------------------------

# Each collector returns (candidates, error). error is None on success (even if
# it found zero images) or a short string describing why it failed. Collectors
# NEVER raise - a broken source can't take down the run.

def _http_reason(code: int) -> str:
    """Turn an HTTP status code into plain English for the user."""
    return {
        400: "bad request",
        401: "access denied (bad key)",
        403: "access denied",
        404: "not found",
        429: "rate-limited (too many requests)",
        500: "server error",
        502: "server error",
        503: "service unavailable",
        504: "timed out",
    }.get(code, f"HTTP {code}")


def collect_pixabay(query: str, cfg: Config):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return [], "no PIXABAY_API_KEY"
    try:
        r = http_get(
            "https://pixabay.com/api/",
            params={
                "key": key,
                "q": query,
                "image_type": "photo",
                "per_page": min(max(cfg.per_query, 3), 200),
                "safesearch": "false",
            },
        )
        if r.status_code != 200:
            return [], _http_reason(r.status_code)
        hits = r.json().get("hits", [])
        return [
            Candidate(h.get("largeImageURL") or h.get("webformatURL"), "pixabay", query)
            for h in hits
            if h.get("largeImageURL") or h.get("webformatURL")
        ], None
    except Exception as exc:
        return [], str(exc)[:90]


def collect_duckduckgo(query: str, cfg: Config):
    try:
        from ddgs import DDGS
    except Exception:
        return [], "ddgs library not installed"
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=cfg.per_query)
        out = [Candidate(item["image"], "duckduckgo", query)
               for item in results if item.get("image")]
        return out, None
    except Exception as exc:
        return [], str(exc)[:90]


def collect_wikimedia(query: str, cfg: Config):
    try:
        r = http_get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": 6,  # File: namespace
                "gsrsearch": query,
                "gsrlimit": min(cfg.per_query, 50),
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": 1024,
            },
            headers={"User-Agent": WIKI_UA},
            retries=3, backoff=2.0,  # Wikimedia rate-limits easily; be patient
        )
        if r.status_code != 200:
            return [], _http_reason(r.status_code)
        pages = r.json().get("query", {}).get("pages", {})
        out = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if mime and not mime.startswith("image/"):
                continue
            if mime in ("image/svg+xml",):
                continue
            url = info.get("thumburl") or info.get("url")
            if url:
                out.append(Candidate(url, "wikimedia", query))
        return out, None
    except Exception as exc:
        return [], str(exc)[:90]


def collect_pexels(query: str, cfg: Config):
    # Pexels: free key, 200 requests/hour + 20,000/month, curated real photos.
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return [], "no PEXELS_API_KEY"
    try:
        r = http_get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": min(max(cfg.per_query, 3), 80)},
            headers={"Authorization": key},
        )
        if r.status_code != 200:
            return [], _http_reason(r.status_code)
        out = []
        for p in r.json().get("photos", []):
            src = p.get("src", {})
            url = src.get("large2x") or src.get("large") or src.get("original")
            if url:
                out.append(Candidate(url, "pexels", query))
        return out, None
    except Exception as exc:
        return [], str(exc)[:90]


def collect_unsplash(query: str, cfg: Config):
    # Unsplash: free key (50 requests/hour demo tier), curated real photos.
    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not key:
        return [], "no UNSPLASH_ACCESS_KEY"
    try:
        r = http_get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": min(max(cfg.per_query, 3), 30)},
            headers={"Authorization": f"Client-ID {key}", "Accept-Version": "v1"},
        )
        if r.status_code != 200:
            return [], _http_reason(r.status_code)
        out = []
        for p in r.json().get("results", []):
            urls = p.get("urls", {})
            url = urls.get("regular") or urls.get("full") or urls.get("raw")
            if url:
                out.append(Candidate(url, "unsplash", query))
        return out, None
    except Exception as exc:
        return [], str(exc)[:90]


# Openverse token cache: with a free client id/secret the page size jumps from 20
# (anonymous, Cloudflare-throttled) to 50 and the source becomes reliable.
_OPENVERSE_TOKEN = {"value": None, "tried": False}


def _openverse_token() -> str | None:
    cid, csec = os.getenv("OPENVERSE_CLIENT_ID"), os.getenv("OPENVERSE_CLIENT_SECRET")
    if not (cid and csec):
        return None
    if _OPENVERSE_TOKEN["tried"]:
        return _OPENVERSE_TOKEN["value"]
    _OPENVERSE_TOKEN["tried"] = True
    try:
        r = requests.post(
            "https://api.openverse.org/v1/auth_tokens/token/",
            data={"client_id": cid, "client_secret": csec, "grant_type": "client_credentials"},
            headers={"User-Agent": "SCRAPR/1.0"}, timeout=20,
        )
        if r.status_code == 200:
            _OPENVERSE_TOKEN["value"] = r.json().get("access_token")
    except requests.RequestException:
        pass
    return _OPENVERSE_TOKEN["value"]


def collect_openverse(query: str, cfg: Config):
    # Openverse (WordPress/Creative Commons): aggregates 800M+ CC images. With a
    # free token it allows 50/page and is reliable; without, 20/page (anonymous).
    token = _openverse_token()
    headers = {"User-Agent": "SCRAPR/1.0 (CV dataset research tool)"}
    cap = 20
    if token:
        headers["Authorization"] = f"Bearer {token}"
        cap = 50
    try:
        r = http_get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": min(max(cfg.per_query, 3), cap)},
            headers=headers,
            retries=0,  # anonymous Openverse rate-limits hard; fail fast and move on
        )
        if r.status_code != 200:
            return [], _http_reason(r.status_code)
        out = []
        for it in r.json().get("results", []):
            url = it.get("url")
            if url:
                out.append(Candidate(url, "openverse", query))
        return out, None
    except Exception as exc:
        return [], str(exc)[:90]


# (display name, internal key used on Candidate.source, collector function)
SOURCES = [
    ("Pixabay", "pixabay", collect_pixabay),
    ("Pexels", "pexels", collect_pexels),
    ("Unsplash", "unsplash", collect_unsplash),
    ("DuckDuckGo", "duckduckgo", collect_duckduckgo),
    ("Openverse", "openverse", collect_openverse),
    ("Wikimedia", "wikimedia", collect_wikimedia),
]


# ---------------------------------------------------------------------------
# Layer 3 - Download + quality filter + perceptual dedup
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    saved: int = 0
    checked: int = 0
    by_source: dict = field(default_factory=dict)
    rejected_small: int = 0
    rejected_dupe: int = 0
    rejected_error: int = 0
    bytes_total: int = 0
    disk_full: bool = False


class Deduper:
    """Thread-safe exact (md5) + perceptual (phash) duplicate detector."""

    def __init__(self, cutoff: int):
        import threading

        self.cutoff = cutoff
        self.md5s: set[str] = set()
        self.phashes: list = []
        self._lock = threading.Lock()

    def check_and_add(self, raw: bytes, img: Image.Image):
        md5 = hashlib.md5(raw).hexdigest()
        try:
            ph = imagehash.phash(img)
        except Exception:
            ph = None
        with self._lock:
            if md5 in self.md5s:
                return False
            if ph is not None:
                for existing in self.phashes:
                    if (ph - existing) <= self.cutoff:
                        return False
            self.md5s.add(md5)
            if ph is not None:
                self.phashes.append(ph)
        return True


def _to_output_image(img: Image.Image, target_fmt: str) -> Image.Image:
    """Prepare a PIL image for saving in the chosen format.

    For JPG we must drop alpha (JPEG has no transparency). Transparent areas
    are flattened onto a white background instead of turning black. We also
    honour any EXIF rotation so saved images aren't sideways.
    """
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if target_fmt == "jpg":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
    else:  # png - keep transparency, but normalise exotic modes
        if img.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            img = img.convert("RGBA")
    return img


def _download_one(cand: Candidate, cfg: Config, deduper: Deduper, dest: Path, idx: int):
    """Returns ('saved'|'small'|'dupe'|'error', source, nbytes)."""
    try:
        # Use the image's own host as Referer. Many CDNs allow same-origin
        # hotlinking but 403 a blank/foreign referer.
        from urllib.parse import urlparse

        parsed = urlparse(cand.url)
        referer = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme else None
        headers = {
            "User-Agent": ua(),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if referer:
            headers["Referer"] = referer
        r = requests.get(cand.url, headers=headers, timeout=25, stream=True)
        if r.status_code != 200:
            return ("error", cand.source, 0)
        raw = r.content
        if len(raw) < cfg.min_bytes:
            return ("small", cand.source, 0)

        try:
            img = Image.open(io.BytesIO(raw))
            img.verify()  # integrity check
        except Exception:
            return ("error", cand.source, 0)

        # Reopen after verify() (verify leaves the object unusable).
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if w < cfg.min_dim or h < cfg.min_dim:
            return ("small", cand.source, 0)

        fmt = (img.format or "").lower()
        if fmt in ("svg", "gif"):
            return ("small", cand.source, 0)

        if not deduper.check_and_add(raw, img):
            return ("dupe", cand.source, 0)
    except Exception:
        return ("error", cand.source, 0)

    # --- Save in a consistent, viewable format (disk errors handled apart) ---
    try:
        if cfg.image_format == "original":
            ext = {"jpeg": "jpg", "jpg": "jpg", "png": "png",
                   "webp": "webp", "bmp": "bmp"}.get(fmt, "jpg")
            path = dest / f"{cand.source}_{idx:05d}.{ext}"
            path.write_bytes(raw)
            nbytes = len(raw)
        else:
            # Re-encode every image into one standard format so the whole
            # dataset is uniform and opens anywhere (no surprise .webp files).
            target_fmt = cfg.image_format  # "jpg" or "png"
            ext = target_fmt
            path = dest / f"{cand.source}_{idx:05d}.{ext}"
            out = _to_output_image(img, target_fmt)
            if target_fmt == "jpg":
                out.save(path, format="JPEG", quality=95, optimize=True)
            else:  # png
                out.save(path, format="PNG", optimize=True)
            nbytes = path.stat().st_size
        return ("saved", cand.source, nbytes)
    except OSError as exc:
        # Out of disk space (ENOSPC) is fatal - signal the caller to stop.
        if getattr(exc, "errno", None) == errno.ENOSPC:
            return ("diskfull", cand.source, 0)
        return ("error", cand.source, 0)
    except Exception:
        return ("error", cand.source, 0)


def download_all(cands: list[Candidate], cfg: Config, dest: Path,
                 stop_at: int | None = None, on_progress=None) -> DownloadResult:
    """Download/clean/dedupe candidates into `dest`, stopping at `stop_at` saved.

    `on_progress(saved, checked, limit)` is called for live status. When vision
    verification is on, `stop_at` is a larger buffer than the final target (so the
    vision step has spares to filter), and `dest` is a staging dir.
    """
    res = DownloadResult()
    deduper = Deduper(cfg.hash_cutoff)
    counter = {"i": 0}
    limit = stop_at if stop_at is not None else cfg.target

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = []
        for cand in cands:
            counter["i"] += 1
            futures.append(pool.submit(_download_one, cand, cfg, deduper, dest, counter["i"]))

        for fut in as_completed(futures):
            status, source, nbytes = fut.result()
            res.checked += 1
            if status == "saved":
                res.saved += 1
                res.by_source[source] = res.by_source.get(source, 0) + 1
                res.bytes_total += nbytes
            elif status == "small":
                res.rejected_small += 1
            elif status == "dupe":
                res.rejected_dupe += 1
            elif status == "diskfull":
                res.disk_full = True
                for f in futures:
                    f.cancel()
                break
            else:
                res.rejected_error += 1
            if on_progress:
                on_progress(res.saved, res.checked, limit)
            if res.saved >= limit:
                for f in futures:
                    f.cancel()
                break
    return res


# ---------------------------------------------------------------------------
# Layer 4 - AI vision verification (Groq multimodal)
# ---------------------------------------------------------------------------

def _thumb_data_url(path: Path, maxdim: int = VISION_MAXDIM) -> str:
    """Downscaled JPEG data-URL of an image, to keep vision token cost low."""
    im = Image.open(path)
    try:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    im = im.convert("RGB")
    im.thumbnail((maxdim, maxdim))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse_groq_duration(s: str) -> float:
    """Parse a Groq reset string like '4.9s', '1m2s', '2h11m2.4s' into seconds."""
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", s or ""):
        try:
            v = float(num)
        except ValueError:
            continue
        total += {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit] * v
    return total


class VisionDailyLimit(Exception):
    """Raised when the per-day vision token budget is exhausted. We stop the
    vision step cleanly rather than thrashing for the rest of the run."""


class VisionVerifier:
    """Looks at each image and decides if it matches the user's description.

    Single-threaded and self-regulating: it reads Groq's rate-limit headers after
    every call and waits for the token bucket to refill when it runs low, so it
    never thrashes against the free-tier limit. If an image truly cannot be
    checked, it returns None. The caller then keeps it OUT of the clean dataset.
    """

    def __init__(self, description: str):
        self.description = description
        self.key = os.getenv("GROQ_API_KEY")
        self.available = bool(self.key)
        self.unavailable_reason = "" if self.key else "no GROQ_API_KEY"
        self._prompt = (
            "You are checking whether an image belongs in a computer-vision dataset.\n"
            f'The user needs: "{description}".\n'
            "Decide if the image clearly satisfies that need: the correct subject, and the "
            "correct kind of image if the user specified one (for example a real photograph "
            "versus a drawing, sketch, diagram, 3D render or illustration).\n"
            'Reply with ONLY this JSON: {"keep": true, "reason": "<max 8 words>"} '
            'or {"keep": false, "reason": "<max 8 words>"}.\n'
            "Be strict: if it is only loosely related, the wrong type, or you are unsure, use false."
        )

    def probe(self) -> None:
        """One cheap call up front to learn if vision is usable right now. Sets
        available=False (with a reason) if the daily budget is spent or the model
        is not accessible, so we can run without it instead of failing mid-way."""
        if not self.available:
            return
        # Use a realistically-sized image (same as a real check) so the probe
        # consumes a comparable number of tokens. Otherwise a tiny probe can
        # squeak under a nearly-exhausted daily budget that real checks can't.
        probe_img = Image.new("RGB", (VISION_MAXDIM, VISION_MAXDIM), (127, 127, 127))
        buf = io.BytesIO()
        probe_img.save(buf, format="JPEG", quality=80)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": VISION_MODEL, "max_tokens": 3, "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "reply ok"},
                    {"type": "image_url", "image_url": {"url": data_url}}]}]}
        try:
            r = requests.post(VISION_URL, headers={"Authorization": f"Bearer {self.key}"},
                              json=body, timeout=30)
        except requests.RequestException:
            self.available = False
            self.unavailable_reason = "could not reach the vision service"
            return
        if r.status_code == 200:
            return
        if r.status_code == 429 and self._is_daily_limit(r):
            self.available = False
            self.unavailable_reason = "daily AI-vision limit reached (free tier; resets within a day)"
            return
        if r.status_code in (401, 403):
            self.available = False
            self.unavailable_reason = "vision model not accessible on this key"
            return
        # A transient/per-minute hiccup during probe - assume it's usable.

    def _is_daily_limit(self, r) -> bool:
        """True if a 429 is the per-DAY token cap (vs a brief per-minute one)."""
        body = (r.text or "").lower()
        if "per day" in body or "tpd" in body or "tokens per day" in body:
            return True
        ra = r.headers.get("retry-after")
        try:
            return float(ra) > 90  # per-minute resets are seconds; per-day is long
        except (TypeError, ValueError):
            return False

    def verify(self, path: Path):
        """Return (keep, reason). keep is True/False, or None if it couldn't be
        checked. Raises VisionDailyLimit if the per-day budget is exhausted."""
        if not self.available:
            return None, "vision unavailable"
        try:
            data_url = _thumb_data_url(path, VISION_MAXDIM)
        except Exception:
            return None, "could not read image"

        body = {
            "model": VISION_MODEL,
            "temperature": 0,
            "max_tokens": 60,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }
        headers = {"Authorization": f"Bearer {self.key}"}

        for attempt in range(5):
            try:
                r = requests.post(VISION_URL, headers=headers, json=body, timeout=60)
            except requests.RequestException:
                time.sleep(2.0)
                continue

            if r.status_code == 200:
                try:
                    content = r.json()["choices"][0]["message"]["content"]
                except Exception:
                    return None, "bad response"
                self._respect_headers(r.headers)
                return _parse_keep(content or "")

            if r.status_code == 429:
                if self._is_daily_limit(r):
                    raise VisionDailyLimit()
                # brief per-minute limit - wait for refill and retry
                time.sleep(self._wait_for(r.headers))
                continue

            # Some other error (auth, server) - not recoverable per-image.
            return None, f"vision http {r.status_code}"

        return None, "rate limited"

    def _wait_for(self, headers) -> float:
        """How long to sleep after a brief 429, from the response headers."""
        ra = headers.get("retry-after")
        if ra:
            try:
                return min(float(ra) + 0.5, 35)
            except ValueError:
                pass
        reset = _parse_groq_duration(headers.get("x-ratelimit-reset-tokens", ""))
        return min(max(reset + 0.3, 1.0), 35)

    def _respect_headers(self, headers) -> None:
        """After a success, if the per-minute token budget is nearly spent, wait
        for it to refill so the next call does not 429."""
        try:
            remaining = int(headers.get("x-ratelimit-remaining-tokens", "999999"))
        except ValueError:
            remaining = 999999
        if remaining < VISION_MIN_TOKENS:
            reset = _parse_groq_duration(headers.get("x-ratelimit-reset-tokens", ""))
            time.sleep(min(max(reset + 0.3, 0.5), 35))


def _parse_keep(raw: str):
    """Parse {"keep": bool, "reason": str} out of a model reply, robustly."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            keep = d.get("keep")
            reason = str(d.get("reason", "")).strip()[:60] or ("match" if keep else "no match")
            if isinstance(keep, bool):
                return keep, reason
            if isinstance(keep, str):
                return keep.strip().lower() in ("true", "yes", "1"), reason
        except json.JSONDecodeError:
            pass
    low = raw.lower()
    if '"keep": true' in low or '"keep":true' in low:
        return True, "match"
    if '"keep": false' in low or '"keep":false' in low:
        return False, "no match"
    return None, "unclear response"


@dataclass
class VerifyResult:
    accepted: int = 0       # matched -> the clean dataset
    rejected: int = 0       # judged "no match" (or a per-image error) -> rejected/
    discarded: int = 0      # surplus past the target -> deleted
    unchecked: int = 0      # daily limit hit before checking -> unverified/
    limit_hit: bool = False
    rejects: list = field(default_factory=list)  # (filename, reason) listed in REJECTED.md


def verify_images(staged: list[Path], verifier: "VisionVerifier", cfg: Config,
                  dest: Path, on_progress=None) -> VerifyResult:
    """Vision-check each staged image, one at a time (the daily token budget makes
    parallelism pointless and risky).

    - matches your description       -> the clean dataset folder (dest)
    - does NOT match / per-image error -> rejected/ (with reason)
    - surplus once the target is hit  -> discarded (not needed)
    - daily vision budget runs out    -> remaining images go to unverified/, and we
                                         stop cleanly (the dataset stays verified-only)
    """
    res = VerifyResult()
    rej_dir = dest / "rejected"
    unv_dir = dest / "unverified"

    def place(path: Path, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            path.replace(folder / path.name)
        except OSError:
            shutil.move(str(path), str(folder / path.name))

    total = len(staged)
    for i, path in enumerate(staged, 1):
        if res.accepted >= cfg.target:
            try:
                path.unlink()
            except OSError:
                pass
            res.discarded += 1
            if on_progress:
                on_progress(i, total, res)
            continue

        try:
            keep, reason = verifier.verify(path)
        except VisionDailyLimit:
            # Out of daily budget: move this image and all remaining to unverified/
            # and stop. Nothing unverified ever enters the clean dataset.
            for rem in staged[i - 1:]:
                if rem.exists():
                    place(rem, unv_dir)
                    res.unchecked += 1
            res.limit_hit = True
            break

        if keep is True:
            place(path, dest)
            res.accepted += 1
        elif keep is False:
            place(path, rej_dir)
            res.rejected += 1
            res.rejects.append((path.name, reason))
        else:
            # A one-off per-image failure - keep it OUT of the clean dataset.
            place(path, rej_dir)
            res.rejected += 1
            res.rejects.append((path.name, reason or "could not verify"))
        if on_progress:
            on_progress(i, total, res)
    return res


def write_rejected_report(dest: Path, res: VerifyResult, cfg: Config) -> None:
    """List every removed image and why, inside the rejected/ folder."""
    if not res.rejects:
        return
    rej_dir = dest / "rejected"
    rej_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Removed images",
        "",
        f"These were collected for **{cfg.label}** but kept out of the dataset by the "
        f"AI vision check. They either did not match your description or could not be "
        f"checked. Your description was:",
        "",
        f"> {cfg.description}",
        "",
        "Nothing here was deleted. Review them, and move any back into the parent "
        "folder if the check got one wrong.",
        "",
        "| File | Why it was removed |",
        "|---|---|",
    ]
    for name, reason in sorted(res.rejects):
        lines.append(f"| {name} | {reason} |")
    lines += ["", f"_{len(res.rejects)} images removed._", ""]
    (rej_dir / "REJECTED.md").write_text("\n".join(lines), encoding="utf-8")


def write_unverified_note(dest: Path, res: VerifyResult) -> None:
    """Explain the unverified/ folder when the daily vision budget ran out."""
    if not res.limit_hit or res.unchecked == 0:
        return
    unv = dest / "unverified"
    unv.mkdir(parents=True, exist_ok=True)
    (unv / "UNVERIFIED.md").write_text(
        "# Not yet verified\n\n"
        f"These {res.unchecked} images were downloaded but **not** checked by the AI "
        "vision model, because the free daily vision budget ran out during this run.\n\n"
        "- The budget resets within a day. Re-run the same request later to verify them.\n"
        "- Or, if you trust them, move them up into the dataset folder yourself.\n"
        "- They are kept here (not in the dataset) so your dataset stays verified-only.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run_status(cfg: Config, final_count: int, dl: DownloadResult) -> str:
    """One plain-English line describing how the run ended."""
    if dl.disk_full:
        return "Stopped early (the disk ran out of space)"
    if final_count >= cfg.target:
        return "Completed (target reached)"
    return "Completed (collected everything available for this description)"


def write_report(cfg: Config, queries: list[str], source_rows: list[dict],
                 dl: DownloadResult, vr: "VerifyResult | None", final_count: int,
                 dest: Path, elapsed: float) -> Path:
    """Write a structured, consistent Markdown report (same template every run)."""
    report = dest / "REPORT.md"
    mb = dl.bytes_total / (1024 * 1024)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_found = sum(r["found"] for r in source_rows)

    L: list[str] = []
    L += [
        f"# {APP_NAME}: Image Collection Report",
        "",
        f"{APP_TAGLINE}.",
        "",
        "| | |",
        "|---|---|",
        f"| **Subject** | {cfg.label} |",
        f"| **Your description** | {cfg.description} |",
        f"| **Date** | {now} |",
        f"| **Duration** | {elapsed:.1f} seconds |",
        f"| **Status** | {run_status(cfg, final_count, dl)} |",
        f"| **Output folder** | `{dest}` |",
        "",
        "## 1. Settings used",
        "",
        "You choose the target; the search settings are tuned automatically to reach it.",
        "",
        "| Setting | Value | Set by | Meaning |",
        "|---|---|---|---|",
        f"| Target images | {cfg.target} | you | {SETTING_NOTES['target']} |",
        f"| Minimum image size | {cfg.min_dim} px | you | {SETTING_NOTES['min_dim']} |",
        f"| Output format | {cfg.image_format} | you | {SETTING_NOTES['format']} |",
        f"| AI vision check | {'on' if cfg.verify else 'off'} | you | removes images that do not match your description |",
        f"| Search variations | {cfg.num_queries} | auto | {SETTING_NOTES['queries']} |",
        f"| Results per source / search | {cfg.per_query} | auto | {SETTING_NOTES['per_query']} |",
        "",
        "## 2. Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Images in final dataset | **{final_count}** |",
        f"| Dataset size | {mb:.1f} MB |",
        f"| Potential images found (all sources) | {total_found} |",
        f"| Images downloaded & cleaned | {dl.saved} |",
        f"| Duplicates removed | {dl.rejected_dupe} |",
        f"| Too small / wrong type | {dl.rejected_small} |",
        f"| Unreachable / broken | {dl.rejected_error} |",
    ]

    if vr is not None:
        L += [
            "",
            "## 3. AI vision check",
            "",
            f"Each downloaded image was checked against your description with the vision model "
            f"`{VISION_MODEL}`. Rejected images are kept in `rejected/` with reasons.",
            "",
            "| Outcome | Count |",
            "|---|---|",
            f"| Matched your description (kept in dataset) | {vr.accepted} |",
            f"| Did not match (moved to rejected/) | {vr.rejected} |",
            f"| Not checked - daily limit (moved to unverified/) | {vr.unchecked} |",
            f"| Extra downloads discarded (target reached) | {vr.discarded} |",
        ]
        if vr.limit_hit:
            L += [
                "",
                "> **The free daily AI-vision budget ran out during this run.** The images in "
                "`unverified/` were downloaded but not checked. Run again later (the budget "
                "resets within a day) to verify them, or use them as-is.",
            ]
        sec_sources, sec_queries, sec_notes = 4, 5, 6
    else:
        sec_sources, sec_queries, sec_notes = 3, 4, 5

    L += [
        "",
        f"## {sec_sources}. Sources",
        "",
        "| Source | Status | Potential images | Downloaded |",
        "|---|---|---|---|",
    ]
    for r in source_rows:
        L.append(f"| {r['name']} | {r['status']} | {r['found']} | {r['saved']} |")

    L += [
        "",
        f"## {sec_queries}. Search queries ({len(queries)})",
        "",
        "| # | Query |",
        "|---|---|",
    ]
    L += [f"| {i} | {q} |" for i, q in enumerate(queries, 1)]

    L += [
        "",
        f"## {sec_notes}. Notes",
        "",
        "- Duplicates are removed with both exact (md5) and visual (perceptual hash) matching, "
        "so the same photo from different sites is only kept once.",
    ]
    if vr is not None:
        L.append("- The AI vision check reads your description and removes images that do not match "
                 "(wrong subject, or an illustration when you wanted a real photo). Removed images are "
                 "in `rejected/` with a `REJECTED.md` listing why. Nothing is deleted.")
    if final_count < cfg.target and not dl.disk_full:
        L.append(f"- Fewer than your target of {cfg.target}: this is everything available online that "
                 f"matches your description. Rare subjects genuinely have limited images on the web.")
    if dl.disk_full:
        L.append("- **The run stopped because the disk filled up.** Free space and run again.")
    if cfg.image_format != "original":
        L.append(f"- Every image was re-encoded to `.{cfg.image_format}` so the dataset is uniform and opens anywhere.")

    L += ["", "---", f"_Report generated by scraper.py on {now}._", ""]
    report.write_text("\n".join(L), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI presentation
# ---------------------------------------------------------------------------

# A compact wordmark for SCRAPR (kept narrow so it fits small terminals).
_LOGO = [
    " ___  ___ ___    _    ___  ___ ",
    "/ __|/ __| _ \\  /_\\  | _ \\| _ \\",
    "\\__ \\ (__|   / / _ \\ |  _/|   /",
    "|___/\\___|_|_\\/_/ \\_\\|_|  |_|_\\",
]


def banner(animate: bool = True) -> None:
    """Branded intro: the logo, what it is, and what it does."""
    console.print()
    for line in _LOGO:
        # Pass style as an argument (not inline markup) so the backslashes in the
        # ASCII logo are treated as literal text, not markup escapes.
        console.print("  " + line, style="bold cyan")
        if animate:
            time.sleep(0.04)
    console.print(f"  [grey58]{APP_TAGLINE}[/]")
    console.print()

    bullets = [
        "Describe the images you need. The AI turns it into precise searches.",
        "Search several free image sources at the same time.",
        "An AI vision model checks every image and removes the ones that do not match.",
        "Save a clean, verified, ready-to-use dataset, even for rare subjects.",
    ]
    console.print("  [bold]What it does[/]")
    for b in bullets:
        console.print(f"    [cyan]•[/] {b}")
        if animate:
            time.sleep(0.10)
    console.print()
    console.print("  [grey58]Tip: run [/][cyan]scraper.py --help[/][grey58] for the full guide.[/]")


# Total number of steps in the run (5 with vision check, 4 without). Set in main.
STEP_TOTAL = [5]


def step(n: int, title: str, blank_after: bool = True) -> None:
    """A clear, left-aligned divider between the stages.

    Pass blank_after=False when the next thing is a transient progress bar,
    which leaves its own blank line behind when it clears (avoids double gaps).
    """
    console.print()
    console.rule(f"[bold cyan]STEP {n} OF {STEP_TOTAL[0]}  ·  {title}[/]", style="cyan", align="left")
    if blank_after:
        console.print()


@contextlib.contextmanager
def thinking(label: str):
    """Print a 'working…' line, then erase it when done. No Live region (which
    duplicates output on some Windows consoles). Just a plain in-place line."""
    progress_line(label + " …")
    try:
        yield
    finally:
        progress_done()


# ---- interactive prompts (shown only when no subject is passed on the CLI) --

class UserQuit(Exception):
    """Raised when the user types 'exit'/'quit' at any prompt."""


HELP_WORDS = {"help", "?", "h", "--help", "-h"}
QUIT_WORDS = {"exit", "quit", "q", ":q", "cancel"}


def show_prompt_help() -> None:
    """A short, in-app guide shown when the user types 'help' at a prompt."""
    console.print()
    console.print("  [bold cyan]Help[/]")
    console.print("  [grey58]You are answering a few quick questions. Then it collects the images.[/]")
    console.print()
    console.print("  [bold]At any question you can:[/]")
    console.print("    [cyan]•[/] Type your answer and press Enter.")
    console.print("    [cyan]•[/] Press Enter alone to accept the recommended value in brackets.")
    console.print("    [cyan]•[/] Type [bold]exit[/] to quit.")
    console.print()
    console.print("  [bold]What the questions mean:[/]")
    console.print("    [cyan]•[/] [bold]Description[/]  describe exactly what you need: subject, style, context.")
    console.print("                 e.g. \"golden retriever dogs, real photographs,")
    console.print("                 various angles, outdoors and indoors\".")
    console.print("    [cyan]•[/] [bold]How many[/]     total images you want in the final dataset.")
    console.print("    [cyan]•[/] [bold]Smallest[/]     minimum image size in pixels (skips tiny icons).")
    console.print("    [cyan]•[/] [bold]Format[/]       jpg (default), png, or original.")
    console.print()
    console.print("  [grey58]The more detail in your description, the better the AI search and the")
    console.print("  AI vision check work. For the full guide, quit and run:[/] [cyan]scraper.py --help[/]")
    console.print()


def _prompt(label: str, hint: str = "") -> str:
    """Ask one question. Understands 'help' and 'exit' at every prompt."""
    while True:
        if label:
            console.print(f"  [bold]{label}[/]")
        if hint:
            console.print(f"  [grey58]{hint}[/]")
        raw = clean_text(console.input("  [cyan]>[/] "))
        low = raw.lower()
        if low in QUIT_WORDS:
            raise UserQuit
        if low in HELP_WORDS:
            show_prompt_help()
            continue  # re-ask the same question
        return raw


def ask_text(label: str) -> str:
    while True:
        v = _prompt(label)
        if v:
            return v
        console.print("  [red]Please type something.[/]\n")


def ask_int(label: str, note: str, default: int, lo: int, hi: int) -> int:
    hint = f"{note}   (min {lo}, max {hi}, recommended {default})"
    while True:
        raw = _prompt(label, hint)
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            console.print(f"  [red]Enter a whole number, or press Enter for {default}.[/]\n")
            continue
        if not (lo <= v <= hi):
            console.print(f"  [red]Please choose a value between {lo} and {hi}.[/]\n")
            continue
        return v


def ask_choice(label: str, note: str, default: str, choices: list[str]) -> str:
    hint = f"{note}   (options: {', '.join(choices)}; recommended {default})"
    while True:
        raw = _prompt(label, hint).lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        console.print(f"  [red]Please choose one of: {', '.join(choices)}.[/]\n")


# ---------------------------------------------------------------------------
# Source health tracking
# ---------------------------------------------------------------------------

class Notifier:
    """Per-source health tracking. If a source fails `disable_after` times in a
    row it is dropped for the rest of the run (and the user is told, once, the
    moment it happens). Every source is treated identically.
    """

    def __init__(self, disable_after: int = 2):
        self.disable_after = disable_after
        self.fail_streak: dict[str, int] = {}
        self.last_reason: dict[str, str] = {}
        self.disabled: set[str] = set()
        self.events: list[str] = []   # collected, printed after the live region

    def source_ok(self, source: str) -> None:
        self.fail_streak[source] = 0

    def source_failed(self, source: str, reason: str) -> None:
        self.last_reason[source] = reason
        self.fail_streak[source] = self.fail_streak.get(source, 0) + 1
        if self.fail_streak[source] >= self.disable_after and source not in self.disabled:
            self.disabled.add(source)
            # Recorded, not printed live. Printing inside a live progress bar
            # corrupts the display on some terminals.
            self.events.append(
                f"  {S.CROSS} [red]{source}[/] stopped responding "
                f"([grey58]{reason}[/]) - skipped, continuing with the rest"
            )

    def is_disabled(self, source: str) -> bool:
        return source in self.disabled


def source_available(name: str) -> tuple[bool, str]:
    """Pre-flight key check so missing-key sources are skipped once, up front."""
    if name == "Pixabay":
        return bool(os.getenv("PIXABAY_API_KEY")), "no PIXABAY_API_KEY in .env"
    if name == "Pexels":
        return bool(os.getenv("PEXELS_API_KEY")), "no PEXELS_API_KEY in .env"
    if name == "Unsplash":
        return bool(os.getenv("UNSPLASH_ACCESS_KEY")), "no UNSPLASH_ACCESS_KEY in .env"
    return True, ""  # DuckDuckGo, Openverse, Wikimedia need no key


# ---------------------------------------------------------------------------
# Argument parsing & config building
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    description = f"""\
{APP_NAME} - {APP_TAGLINE}

{APP_NAME} builds image datasets for computer vision. You DESCRIBE the images you
need; an AI turns that into precise searches, runs them across several free image
sources, removes duplicates, then an AI VISION model looks at every image and
keeps only the ones that actually match your description. Built to find even rare,
specific images (e.g. one stage of a construction process), not just common ones.
"""
    epilog = """\
GETTING STARTED
  1. Install the requirements:   pip install -r requirements.txt
  2. Put your keys in the .env file next to this script:
        GROQ_API_KEY=...        (required - AI search queries AND the vision check)
        PIXABAY_API_KEY=...     (required - main image source)
        PEXELS_API_KEY=...      (recommended - curated real photos)
        UNSPLASH_ACCESS_KEY=... (recommended - curated real photos)
     DuckDuckGo, Openverse and Wikimedia need no key. Missing sources are skipped.
     See the README for a step-by-step guide to each free key.
  3. Run it.

TWO WAYS TO RUN
  Interactive (recommended) - it asks you everything:
      python scraper.py

  One-shot - pass a DESCRIPTION (use quotes) and any options:
      python scraper.py "golden retriever dogs, real photos, various angles"
      python scraper.py "solar panels on rooftops, real photographs" --target 200
      python scraper.py "coral reef fish, underwater photography" --min-dim 512
      python scraper.py "vintage cars, real photos" --sources duckduckgo,pixabay --no-verify

DESCRIBE, DON'T JUST NAME
  The more specific your description, the better. Say the subject, the style
  (real photo vs illustration), the context, and any stages or angles you want.
  The AI uses it twice: to write the searches, and to judge each downloaded image.

THE SETTINGS (numbers have a safe min/max; out-of-range values are clamped)
  --target      how many images you want in the final dataset (the main control)
  --min-dim     smallest acceptable image, in pixels (width and height)
  --format      jpg (default), png, or original
  --sources     limit which sources are used
  --no-verify   skip the AI vision check (faster, but keeps off-topic images)
  --queries     override: how many search phrasings the AI invents (else auto)
  --per-query   override: results pulled from each source per search (else auto)
  --out         where to save the dataset

WHAT YOU GET
  A folder under ./scraped_images/<subject>/ with the matching images, a REPORT.md,
  and a rejected/ subfolder holding images the vision check removed (with reasons),
  so you can review them and recover any it got wrong. Nothing is deleted.

GOOD TO KNOW
  - The vision check is paced for the free Groq limit, so a run of a few hundred
    images takes a few minutes. The free tier also allows ~1,000 vision checks
    per day; if that runs out mid-run, unchecked images go to unverified/ and the
    tool tells you. Use --no-verify to skip the check entirely.
  - For rare subjects the web genuinely has limited images; the tool collects all
    it can find that match and tells you honestly when that is below your target.
  - No images are invented - this only collects what already exists online.
"""
    p = argparse.ArgumentParser(
        prog="scraper.py",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", metavar="DESCRIPTION",
                   help='describe the images you need (use quotes). Omit it to run interactively.')
    p.add_argument("--target", type=int, default=DEFAULTS["target"], metavar="N",
                   help=f"how many images you want (min {LIMITS['target'][0]}, max {LIMITS['target'][1]}, default {DEFAULTS['target']}). This is the main control.")
    p.add_argument("--queries", type=int, default=None, metavar="N",
                   help=f"override: how many search variations the AI generates (min {LIMITS['queries'][0]}, max {LIMITS['queries'][1]}). Default: chosen automatically from --target.")
    p.add_argument("--per-query", type=int, default=None, metavar="N",
                   help=f"override: images requested per source per search (min {LIMITS['per_query'][0]}, max {LIMITS['per_query'][1]}). Default: chosen automatically from --target.")
    p.add_argument("--min-dim", type=int, default=DEFAULTS["min_dim"], metavar="PX",
                   help=f"minimum width AND height in pixels (min {LIMITS['min_dim'][0]}, max {LIMITS['min_dim'][1]}, default {DEFAULTS['min_dim']}).")
    p.add_argument("--format", choices=["jpg", "png", "original"], default=DEFAULTS["format"],
                   help="output format: jpg (uniform & viewable, default), png (lossless), original (keep source).")
    p.add_argument("--sources", type=str, default=None, metavar="LIST",
                   help="comma-separated subset to use: pixabay,pexels,unsplash,duckduckgo,openverse,wikimedia (default: all).")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the AI vision check (faster, but keeps images that do not match).")
    p.add_argument("--out", type=str, default=None, metavar="PATH",
                   help="base output folder (default: ./scraped_images).")
    return p


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def auto_search_params(target: int) -> tuple[int, int]:
    """Pick (num_queries, per_query) automatically from the desired image count.

    These three numbers are related: the images you can collect are bounded by
    queries x sources x per_query (minus duplicates and dead links). Rather than
    make the user reconcile them, we derive generous search settings from the one
    number they care about - the target, so the target behaves like a real goal.
    """
    per_query = 40  # pull plenty from each source (still within every source cap)
    num_queries = _clamp(round(target / 15) + 4, 8, LIMITS["queries"][1])
    return num_queries, per_query


def build_config(args) -> Config | None:
    """Build a Config either from CLI flags (one-shot) or interactive prompts."""
    if args.query:
        # One-shot. target drives everything; queries/per-query auto-derive unless
        # the user explicitly overrode them on the command line.
        target = _clamp(args.target, *LIMITS["target"])
        auto_q, auto_pq = auto_search_params(target)
        cfg = Config(
            description=clean_text(args.query),
            target=target,
            num_queries=_clamp(args.queries, *LIMITS["queries"]) if args.queries is not None else auto_q,
            per_query=_clamp(args.per_query, *LIMITS["per_query"]) if args.per_query is not None else auto_pq,
            min_dim=_clamp(args.min_dim, *LIMITS["min_dim"]),
            image_format=args.format,
            verify=not args.no_verify,
        )
    else:
        console.print()
        console.print("  [grey58]A few quick questions. Press Enter to accept the value in brackets.[/]")
        console.print("  [grey58]Type [/][bold]help[/][grey58] for guidance or [/][bold]exit[/][grey58] to quit, at any question.[/]\n")
        try:
            console.print("  [bold]Describe the images you need[/] [grey58](be specific: subject, "
                          "style, context)[/]")
            console.print("  [grey58]e.g. \"golden retriever dogs, real photographs, various angles,"
                          " outdoors and indoors\"[/]")
            description = ask_text("")
            console.print()
            target = ask_int("How many images?", SETTING_NOTES["target"],
                             DEFAULTS["target"], *LIMITS["target"])
            console.print()
            min_dim = ask_int("Smallest image size in pixels?", SETTING_NOTES["min_dim"],
                              DEFAULTS["min_dim"], *LIMITS["min_dim"])
            console.print()
            fmt = ask_choice("Output image format?", SETTING_NOTES["format"],
                             DEFAULTS["format"], ["jpg", "png", "original"])
        except UserQuit:
            console.print("\n  [grey58]Exiting. Run again any time, or use[/] [cyan]scraper.py --help[/][grey58].[/]")
            return None
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [grey58]Cancelled.[/]")
            return None
        # Search settings are derived from the target so they never conflict.
        num_queries, per_query = auto_search_params(target)
        cfg = Config(description=description, target=target, num_queries=num_queries,
                     per_query=per_query, min_dim=min_dim, image_format=fmt,
                     verify=not args.no_verify)

    if args.out:
        cfg.out_dir = Path(args.out)
    if args.sources:
        wanted = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
        valid = {key for _, key, _ in SOURCES}
        cfg.sources = tuple(k for k in valid if k in wanted) or None
    return cfg


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    load_dotenv(HERE / ".env")
    banner()

    cfg = build_config(args)
    if cfg is None:
        return

    # Decide whether the AI vision check will run (needs the key, the model, AND
    # the daily budget). probe() makes one cheap call up front so we find out now
    # rather than after downloading everything.
    verifier = VisionVerifier(cfg.description) if cfg.verify else None
    if verifier:
        with thinking("Checking AI vision availability…"):
            verifier.probe()
    do_verify = bool(verifier and verifier.available)
    STEP_TOTAL[0] = 5 if do_verify else 4

    start = time.time()
    notifier = Notifier()

    # ---- STEP 1 · search queries -------------------------------------------
    step(1, "Understanding your request")
    with thinking("Reading your description and writing precise searches…"):
        subject, queries, qnotes = generate_queries(cfg)
    cfg.subject = subject
    for msg, style in qnotes:
        sym = S.CROSS if style == "red" else S.DOT
        console.print(f"  {sym} [grey58]{msg.strip()}[/]")

    console.print(f"  {S.TICK} subject: [bold]{cfg.label}[/]")
    if len(queries) == 1:
        console.print(f"  {S.TICK} searching your exact words [grey58](AI expansion unavailable)[/]")
    else:
        console.print(f"  {S.TICK} [bold]{len(queries)}[/] precise search queries ready")
    if cfg.verify and not do_verify:
        console.print(f"  {S.CROSS} [yellow]AI vision check off:[/] [grey58]{verifier.unavailable_reason}. "
                      f"Running without it (like --no-verify).[/]")
    console.print()
    items = [f"[grey58]{i:>2}[/]  {q}" for i, q in enumerate(queries, 1)]
    console.print(Columns(items, padding=(0, 4), equal=True, column_first=True))

    # Now that we have a short subject, set up the output folder.
    dest = cfg.out_dir / slugify(cfg.label)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        console.print(f"\n  {S.CROSS} [red]Could not create the output folder[/] [cyan]{dest}[/]: {exc}")
        return

    # ---- STEP 2 · search sources -------------------------------------------
    step(2, "Searching image sources", blank_after=False)

    selected = [(d, k, fn) for d, k, fn in SOURCES if cfg.sources is None or k in cfg.sources]
    active, skipped = [], {}
    for disp, key, fn in selected:
        ok, reason = source_available(disp)
        if ok:
            active.append((disp, key, fn))
        else:
            skipped[disp] = reason
    if not active:
        console.print(f"  {S.CROSS} [red]No usable sources. Add keys to .env (need GROQ + PIXABAY).[/]")
        return

    candidates: list[Candidate] = []
    seen_urls: set[str] = set()
    total_steps = len(queries) * len(active)
    done = 0
    for q in queries:
        for disp, key, fn in active:
            done += 1
            if notifier.is_disabled(disp):
                continue
            progress_line(f"Searching {disp}…  ({done}/{total_steps})  found {len(candidates)}")
            found, err = fn(q, cfg)
            if err is not None:
                notifier.source_failed(disp, err)
            else:
                notifier.source_ok(disp)
                for c in found:
                    if c.url not in seen_urls:
                        seen_urls.add(c.url)
                        candidates.append(c)
            polite_sleep()
    progress_done()

    by_src: dict[str, int] = {}
    for c in candidates:
        by_src[c.source] = by_src.get(c.source, 0) + 1

    # Build the per-source result rows (used for the table now + the report).
    source_rows = []
    active_names = {d for d, _, _ in active}
    for disp, key, fn in selected:
        found = by_src.get(key, 0)
        if disp in skipped:
            status, sym, color = "skipped (no API key)", S.CROSS, S.MUTE
        elif notifier.is_disabled(disp):
            status, sym, color = f"stopped ({notifier.last_reason.get(disp, 'failed')})", S.CROSS, S.BAD
        elif found > 0:
            status, sym, color = "ok", S.TICK, S.OK
        elif disp in active_names:
            status, sym, color = "no results", S.DOT, S.MUTE
        else:
            status, sym, color = "not used", S.DOT, S.MUTE
        source_rows.append({"name": disp, "key": key, "status": status,
                            "sym": sym, "color": color, "found": found, "saved": 0})

    t = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, padding=(0, 3))
    t.add_column("  Source")
    t.add_column("Status")
    t.add_column("Found", justify="right")
    for r in source_rows:
        t.add_row(f"  {r['name']}", f"[{r['color']}]{r['sym']} {r['status']}[/]", str(r["found"]))
    console.print(t)
    console.print()
    console.print(f"  {S.TICK} [bold]{len(candidates)}[/] potential images found "
                  f"[grey58]across all sources[/]")

    if not candidates:
        console.print(f"\n  {S.CROSS} [red]No images found.[/] Try a broader topic, "
                      f"or raise [cyan]--target[/] for wider searches.")
        return

    # Shuffle so we interleave sources rather than draining one first.
    random.shuffle(candidates)

    # ---- STEP 3 · download & clean -----------------------------------------
    step(3, "Downloading & cleaning images")
    # With vision on, download a buffer into a staging folder so the vision step
    # has spares to filter; without vision, download straight to the target.
    if do_verify:
        download_budget = min(len(candidates), VISION_HARD_CAP, max(cfg.target * 4, 60))
        staging = dest / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
        download_dir = staging
    else:
        download_budget = cfg.target
        download_dir = dest

    console.print(f"  [grey58]Minimum size {cfg.min_dim}px. Saving as .{cfg.image_format}. "
                  f"{'Downloading a buffer for the vision check.' if do_verify else f'Target {cfg.target} images.'}[/]")

    def dl_progress(saved, checked, limit):
        progress_line(f"Downloading…  {saved}/{limit} saved  ({checked} links checked)")

    res = download_all(candidates, cfg, download_dir, stop_at=download_budget, on_progress=dl_progress)
    progress_done()

    for r in source_rows:
        r["saved"] = res.by_source.get(r["key"], 0)

    console.print(f"  {S.TICK} [bold green]{res.saved}[/] images downloaded & cleaned")
    console.print(f"  {S.DOT} [grey58]from {res.checked} checked: "
                  f"{res.rejected_dupe} duplicates, {res.rejected_small} too small, "
                  f"{res.rejected_error} unreachable[/]")
    if res.disk_full:
        console.print(f"  {S.CROSS} [red]The disk ran out of space, so the run stopped early "
                      f"and kept what already fit.[/]")

    # ---- STEP 4 · AI vision check ------------------------------------------
    vr = None
    if do_verify:
        step(4, "Checking each image with AI vision")
        staged = [p for p in download_dir.iterdir() if p.is_file()]
        est = max(1, round(min(len(staged), VISION_HARD_CAP) / 35))
        console.print(f"  [grey58]Showing each of the {len(staged)} images to the vision model and "
                      f"keeping only real matches.[/]")
        console.print(f"  [grey58]Paced for the free rate limit, about {est} minute(s).[/]")
        if staged:
            def vis_progress(i, total, r):
                progress_line(f"Checking image {i}/{total}…  kept {r.accepted}  removed {r.rejected}")
            vr = verify_images(staged, verifier, cfg, dest, on_progress=vis_progress)
            progress_done()
            write_rejected_report(dest, vr, cfg)
            write_unverified_note(dest, vr)
        else:
            vr = VerifyResult()
        # Clean up the now-empty staging folder.
        try:
            (dest / "_staging").rmdir()
        except OSError:
            pass

        console.print(f"  {S.TICK} [bold green]{vr.accepted}[/] images match your description (kept)")
        if vr.rejected:
            console.print(f"  {S.DOT} [grey58]{vr.rejected} did not match (off-topic or wrong type) "
                          f"→ rejected/ with reasons[/]")
        if vr.discarded:
            console.print(f"  {S.DOT} [grey58]{vr.discarded} extra downloads discarded "
                          f"(target already reached)[/]")
        if vr.limit_hit:
            console.print(f"  {S.CROSS} [yellow]Daily AI-vision budget reached.[/] [grey58]{vr.unchecked} "
                          f"images left unchecked → unverified/ (re-run later to check them).[/]")

    # ---- FINAL · save results ----------------------------------------------
    # The dataset is exactly the verified-accepted images (or all downloaded if
    # the vision check was off).
    final_count = vr.accepted if vr is not None else res.saved
    step(STEP_TOTAL[0], "Saving results")
    elapsed = time.time() - start
    report = write_report(cfg, queries, source_rows, res, vr, final_count, dest, elapsed)
    mb = res.bytes_total / (1024 * 1024)
    console.print(f"  {S.TICK} Images   [cyan]{dest}[/]")
    console.print(f"  {S.TICK} Report   [cyan]{report}[/]")
    if vr is not None and vr.rejects:
        console.print(f"  {S.TICK} Removed  [cyan]{dest / 'rejected'}[/] [grey58]({len(vr.rejects)} images + REJECTED.md)[/]")
    if vr is not None and vr.unchecked:
        console.print(f"  {S.TICK} Unchecked [cyan]{dest / 'unverified'}[/] [grey58]({vr.unchecked} images, daily limit)[/]")
    console.print()
    console.print(f"  [grey58]{final_count} images · {mb:.1f} MB · {elapsed:.0f} seconds[/]")
    if final_count < cfg.target and not res.disk_full and not (vr and vr.limit_hit):
        console.print(f"  [grey58]Fewer than your target of {cfg.target}: this is everything online "
                      f"that matches your description. Rare subjects are genuinely limited.[/]")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]  Interrupted by user.[/]")
        sys.exit(130)
