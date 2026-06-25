# SCRAPR - Image Dataset Builder for Computer Vision

A free, domain-agnostic image-dataset collector for computer-vision projects.
You **describe** the images you need; an AI turns that into precise searches,
scrapes several free sources, removes duplicates, then an **AI vision model looks
at every image and keeps only the ones that actually match your description**.
Built to find even rare, specific images, not just common ones.

```
You describe:  "golden retriever dogs, real photographs, various angles,
                outdoors and indoors"
   |
   +-- AI reads it -> a short subject ("golden retriever") + precise, on-subject queries
   |
   +-- Each query runs across 6 free sources:
   |     Pixabay, Pexels, Unsplash, DuckDuckGo, Openverse, Wikimedia
   |
   +-- Download + clean:  min size, no tiny icons, exact-dup (md5), near-dup (pHash),
   |                      re-encoded to uniform .jpg
   |
   +-- AI VISION CHECK:  every image is shown to a vision model and judged against
   |                     your description. Matches are kept; the rest go to rejected/
   |
   +-- Output:  scraped_images/<subject>/  +  REPORT.md  +  rejected/ (with reasons)
```

**Why the vision check matters:** for a rare or specific subject, a plain search
returns mostly junk: diagrams, sketches, unrelated photos. SCRAPR throws those
out automatically, so you get a clean, relevant dataset instead of noise.

---

## 1. Setup

**a) Install dependencies** (run once):

```bash
pip install -r requirements.txt
```

**b) API keys** - copy `.env.example` to `.env` and fill in the keys:

```bash
cp .env.example .env
```

```
# Required
GROQ_API_KEY=...          # AI search queries and the vision check
PIXABAY_API_KEY=...       # primary image source

# Recommended (extra free sources for higher volume)
PEXELS_API_KEY=...        # curated real photos
UNSPLASH_ACCESS_KEY=...   # curated real photos

# Optional (raises Openverse from 20 to 50 images per search)
OPENVERSE_CLIENT_ID=...
OPENVERSE_CLIENT_SECRET=...
```

Only **Groq** and **Pixabay** are required. DuckDuckGo, Openverse and Wikimedia
work with no key at all. Any missing key or unavailable source is skipped
automatically, and the run continues with whatever sources are available.

> Adding the Pexels and Unsplash keys roughly doubles the candidate images per
> run, which matters when the target is 1000+.

> **Security:** `.env` is git-ignored so keys are never committed. Never share or
> push real keys.

### Obtaining the free keys

All keys are free and require no credit card.

| # | Service | Where | Steps |
|---|---------|-------|-------|
| 1 | **Groq** (required) | [console.groq.com](https://console.groq.com) | Sign in -> **API Keys** -> **Create API Key**. |
| 2 | **Pixabay** (required) | [pixabay.com/api/docs](https://pixabay.com/api/docs/) | Sign in; the key is shown in the **"Your API key"** box. |
| 3 | **Pexels** | [pexels.com/api](https://www.pexels.com/api/) | Sign in -> **Get Started** -> copy the key. |
| 4 | **Unsplash** | [unsplash.com/developers](https://unsplash.com/developers) | **Your apps** -> **New Application** -> copy the **Access Key**. |
| 5 | **Openverse** (optional) | [api.openverse.org](https://api.openverse.org/v1/#tag/auth) | Register a free application to get a client id/secret (lifts the per-search limit to 50). |

After filling in `.env`, run `python scraper.py`; the configured sources appear
in the Step 2 table when the run starts.

### Sources at a glance

| Source | Key | Free limit | Best for |
|--------|-----|-----------|----------|
| **Pixabay** | required | 5,000 req/hour | clean stock photos |
| **Pexels** | optional | 200 req/hour, 20k/month | curated real photos |
| **Unsplash** | optional | 50 req/hour (demo) | curated real photos |
| **DuckDuckGo** | none | generous | broad web image results |
| **Openverse** | optional token | 800M+ images; 50/search with token | Creative Commons images |
| **Wikimedia** | none | generous | documentary / reference images |

With no optional keys the tool still runs on Pixabay, DuckDuckGo, Openverse and
Wikimedia. Adding Pexels and Unsplash is recommended for large datasets.

---

## 2. Usage

### Interactive mode (recommended)

```bash
python scraper.py
```

It asks four short questions (a **description**, **how many images**, **smallest
image size**, and **format**) then does everything else automatically. Images
land in `./scraped_images/<subject>/`.

**Describe, don't just name.** The more specific your description, the better.
It is used twice: to write the searches *and* to judge each image. Compare:

- weak: `coffee cup`
- strong: `coffee cups on cafe tables, real photographs, various angles, natural light`
- strong: `solar panels on residential rooftops, aerial and ground-level, real photos only`

At any question you can type **`help`** for an in-app guide or **`exit`** to
quit. Press Enter alone to accept the recommended value shown in brackets.

### One-shot mode (pass the description directly, in quotes)

```bash
python scraper.py "golden retriever dogs, real photos, various angles"
```

### With options

```bash
python scraper.py "solar panels on rooftops, real photographs" --target 200
python scraper.py "coral reef fish, underwater photography" --min-dim 512 --format png
python scraper.py "vintage cars, real photos" --sources duckduckgo,pixabay --no-verify
```

You can combine any options. Order does not matter. Always quote the description.
Run `python scraper.py --help` for the full guide at any time.

### All options

`--target` is the only number you normally set. Everything else has a sensible
default. The two search knobs are **auto-tuned from `--target`**; you only touch
them if you want manual control.

| Flag          | Default | What it does |
|---------------|---------|--------------|
| `--target N`    | 400   | **The main control** - how many images you want in the final dataset. |
| `--min-dim N`   | 300   | Minimum width **and** height in pixels. |
| `--format X`    | jpg   | Output format: `jpg` (uniform, viewable, smallest), `png` (lossless), or `original` (keep source bytes). |
| `--sources L`   | all   | Comma-separated subset: `pixabay,pexels,unsplash,duckduckgo,openverse,wikimedia`. |
| `--no-verify`   | off   | Skip the AI vision check (faster, but keeps off-topic images). |
| `--out PATH`    | `./scraped_images` | Where to create the dataset folder. |
| `--queries N`   | auto  | Override: how many search-phrase variations the AI generates. |
| `--per-query N` | auto  | Override: images requested **per source, per search**. |

Out-of-range numbers are automatically clamped to the safe min/max in
**section 5**, so you cannot accidentally break a run with a bad value.

---

## 3. What you will see while it runs

The CLI runs in **5 clearly divided steps** (4 with `--no-verify`) and tells you
immediately if anything goes wrong:

```
STEP 1 OF 5 - Understanding your request
  > subject: golden retriever
  > 8 precise search queries ready       (then lists them)

STEP 2 OF 5 - Searching image sources
  Source        Status                      Found
  Pixabay       > ok                          320
  Pexels        > ok                          162
  Unsplash      > ok                           30
  DuckDuckGo    > ok                          255
  Openverse     > ok                          130
  Wikimedia     > ok                          100

  > 997 potential images found across all sources

STEP 3 OF 5 - Downloading & cleaning images
  > 60 images downloaded & cleaned
  - from 149 checked: 2 duplicates, 4 too small, 83 unreachable

STEP 4 OF 5 - Checking each image with AI vision
  > 40 images match your description
  - 12 removed (off-topic or wrong type) -> moved to rejected/ with reasons
  - 8 extra downloads discarded (target already reached)

STEP 5 OF 5 - Saving results
  > Images   ...\scraped_images\golden_retriever
  > Report   ...\scraped_images\golden_retriever\REPORT.md
  > Removed  ...\golden_retriever\rejected  (12 images + REJECTED.md)

  40 images - 16.3 MB - 234 seconds
```

**Reading it**
- **Green check** = working. **Red x** = a problem. The run never stops for one
  bad source; it continues with the others.
- **"Potential images found"** are the links discovered. Not all become files;
  some are duplicates, too small, or dead links (step 3 breaks this down).
- **Step 4 is the AI vision check.** It looks at each image and keeps only those
  matching your description. It is paced to respect the free Groq token limit, so
  a few hundred images take a few minutes. Use `--no-verify` to skip it.
- **If the final count is below your target**, the web genuinely did not have more
  that match. Rare subjects are limited online.
- A source that fails twice in a row is dropped for the rest of the run.

> Progress shows as a single live line that updates in place, so the final
> screen stays clean on any terminal.

---

## 4. Output structure

```
scraped_images/
+-- golden_retriever/
    +-- pixabay_00001.jpg        <-- images that MATCHED your description
    +-- duckduckgo_00007.jpg
    +-- ...
    +-- REPORT.md                <-- structured run report (see below)
    +-- rejected/
    |   +-- duckduckgo_00005.jpg <-- images the vision check removed
    |   +-- ...
    |   +-- REJECTED.md          <-- lists every removed file and WHY
    +-- unverified/              <-- only if the daily vision budget ran out
        +-- ...                  <-- downloaded but not yet checked
        +-- UNVERIFIED.md        <-- what to do with them
```

The top folder is your **clean, verified dataset**. It contains ONLY images the
vision model approved against your description. The `rejected/` folder holds what
was removed, each with a reason in `REJECTED.md`. **Nothing is deleted**; if the
check got one wrong, just move it back up into the dataset folder.

`unverified/` only appears if the free daily vision budget runs out mid-run (see
below). Those images were downloaded but not checked. Re-run later to verify
them, or use them as-is.

Every image is a standard `.jpg` (unless you chose `--format png/original`), so
they all open in any viewer and load cleanly in any CV framework.

**`REPORT.md`** is the same structured template every run: **1. Settings used**,
**2. Results**, **3. AI vision check** (kept vs removed), **4. Sources**,
**5. Search queries**, **6. Notes**. (With `--no-verify` it drops the vision
section.)

---

## 5. Limits and safe ranges

In normal use you set just one number: **`--target`** (how many images you
want). The two search knobs below are tuned automatically from it; the table
shows their safe range only for the rare case you override them.

| Flag | Min | Max | What changes if you raise / lower it |
|---|---|---|---|
| `--target` | 1 | ~2000 common / ~500-800 niche | The goal. It cannot invent images that do not exist on the web, so for rare topics you will hit a real ceiling below your number. |
| `--min-dim` | 50 | ~1024 | Higher = fewer but sharper images, more rejects. Lower = more images but you will collect thumbnails/logos. 300 is the standard CV floor. **It is a side length in pixels, not megapixels.** |
| `--per-query` *(override)* | 3 | ~80 effective | Each source caps it (Pixabay 200, Pexels 80, Wikimedia 50, Openverse 50, Unsplash 30). Raising it pulls more per source per search. |
| `--queries` *(override)* | 1 | ~30 | Biggest lever for **variety**. The AI output is token-capped (~30 queries max). Runtime scales with this. Past ~25 you get looser, less relevant matches. |

Values outside these ranges are clamped automatically, so a typo cannot break a run.

**The AI vision check** (on by default) is paced to respect the free Groq token
limit (~35 images/minute), and verifies up to ~300 images per run, so a run takes
a few minutes. The free tier also has a **daily budget of ~1,000 vision checks**
(500k tokens/day). If it runs out mid-run, SCRAPR stops the check cleanly, puts
the unchecked images in `unverified/`, and tells you. It never pollutes the
dataset. The budget resets within a day; re-run to finish, or use `--no-verify`
to skip the check entirely.

**Realistic yield:** common subjects easily reach the target; rare subjects are
limited by what exists online *and* by how many genuinely match. SCRAPR tells
you the honest number. The vision check trades quantity for relevance, which is
exactly what you want for rare datasets.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Missing dependency: X` | Run `pip install -r requirements.txt`. |
| `AI vision check off: ...` | Either the Groq key is missing, or the **daily vision budget is used up** (resets within a day). The run continues without verification, like `--no-verify`. |
| Images ended up in `unverified/` | The daily vision budget ran out mid-run. Those images were not checked. Re-run later to verify them, or use them as-is. |
| `searching your exact words (AI expansion unavailable)` | Groq unreachable for query writing. The scrape still runs on your exact words. |
| Vision step feels slow | It is paced for the free rate limit (~35/min). That is normal. Use `--no-verify` to skip it, or a smaller `--target`. |
| Too many images removed / too few kept | Your description may be too strict, or the subject is genuinely rare online. Loosen the wording, or review `rejected/REJECTED.md` and move good ones back up. |
| A source shows "stopped" or "access denied" | That source is temporarily unavailable or the API key is invalid. The run continues with the remaining sources. |
| Many "unreachable" in the breakdown | Normal for free scraping. Dead links and hotlink-protected hosts are skipped automatically. |
| `Disk ran out of space` | The run stopped early and kept what fit. Free some space and run again. |
| Final count below your target | The web genuinely did not have more that match your description. Loosen the description or accept it as seed data for augmentation. |

---

## Built with

`requests`, `Pillow`, `ImageHash`, `groq`, `ddgs`, `python-dotenv`, `rich`

AI models: Llama 3.3 70B (query expansion) and Llama-4 Scout (vision verification), both via the Groq API.
