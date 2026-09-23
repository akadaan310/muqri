#!/usr/bin/env python3
"""Build the reciter FAISS index from the EveryAyah.com catalogue.

For every Hafs reciter in the EveryAyah catalogue (auto-discovered from
``https://everyayah.com/data/recitations.js``, de-duplicated by name, highest bitrate kept) the
script downloads a set of sample ayahs, runs the full qaari-eval pipeline on each, averages the
per-ayah timbre embeddings and style vectors into one profile, and writes
``index/reciters_faiss.index`` + ``index/reciters_meta.json``.

EveryAyah hosts ~50 distinct Hafs reciters (Murattal and Mujawwad variants count separately).
To reach 100+ profiles, pass ``--extra-catalog`` with a JSON list of additional sources::

    [{"id": "my_reciter", "name": "Some Reciter", "url_template": "https://host/path/{surah:03d}{ayah:03d}.mp3"}]

Usage::

    python datasets/index_reciters.py                       # all reciters, default ayahs
    python datasets/index_reciters.py --limit 5 --ayahs 113:1-5
    python datasets/index_reciters.py --reciters Husary Minshawy --aligner heuristic
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audio import AudioError, load_audio  # noqa: E402
from app.pipeline import AnalysisOptions, QaariEvaluator  # noqa: E402
from app.profiling import ReciterIndex, merge_profiles  # noqa: E402
from app.quran_text import get_ayah_text, validate_reference  # noqa: E402

logger = logging.getLogger("index_reciters")

CATALOG_URL = "https://everyayah.com/data/recitations.js"
AUDIO_URL = "https://everyayah.com/data/{subfolder}/{surah:03d}{ayah:03d}.mp3"
DEFAULT_AYAHS = "1:1-7,112:1-4,113:1-5,114:1-6"
USER_AGENT = "qaari-eval-indexer/0.1 (+https://github.com/akadaan310/muqri)"
_EXCLUDE = re.compile(r"(^translations/|^English/|^MultiLanguage/|^warsh/|Parhizgar)", re.IGNORECASE)

# Fallback catalogue used when the live catalogue cannot be fetched.
FALLBACK_CATALOG: dict[str, str] = {
    "Husary_128kbps": "Husary",
    "Husary_128kbps_Mujawwad": "Husary Mujawwad",
    "Husary_Muallim_128kbps": "Husary (Muallim)",
    "Minshawy_Murattal_128kbps": "Minshawy Murattal",
    "Minshawy_Mujawwad_192kbps": "Minshawy Mujawwad",
    "Abdul_Basit_Murattal_192kbps": "Abdul Basit Murattal",
    "Abdul_Basit_Mujawwad_128kbps": "Abdul Basit Mujawwad",
    "Alafasy_128kbps": "Alafasy",
    "Abdurrahmaan_As-Sudais_192kbps": "Abdurrahmaan As-Sudais",
    "Saood_ash-Shuraym_128kbps": "Saood bin Ibraaheem Ash-Shuraym",
    "Maher_AlMuaiqly_64kbps": "Maher Al Muaiqly",
    "Muhammad_Ayyoub_128kbps": "Muhammad Ayyoub",
    "Muhammad_Jibreel_128kbps": "Muhammad Jibreel",
    "Abu_Bakr_Ash-Shaatree_128kbps": "Abu Bakr Ash-Shaatree",
    "Hudhaify_128kbps": "Hudhaify",
    "Ghamadi_40kbps": "Ghamadi",
    "Mohammad_al_Tablaway_128kbps": "Mohammad al Tablaway",
    "Mustafa_Ismail_48kbps": "Mustafa Ismail",
    "Yasser_Ad-Dussary_128kbps": "Yasser Ad-Dussary",
    "Nasser_Alqatami_128kbps": "Nasser Alqatami",
}


@dataclass(slots=True)
class ReciterSource:
    reciter_id: str
    name: str
    url_template: str


def parse_ayah_spec(spec: str) -> list[tuple[int, int]]:
    """Parse ``"1:1-7,113:1"`` into [(1,1),...,(1,7),(113,1)]."""
    out: list[tuple[int, int]] = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        m = re.fullmatch(r"(\d+):(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"Bad ayah spec {part!r}; expected SURAH:AYAH or SURAH:FROM-TO")
        surah, a0 = int(m.group(1)), int(m.group(2))
        a1 = int(m.group(3) or a0)
        for ayah in range(a0, a1 + 1):
            validate_reference(surah, ayah)
            out.append((surah, ayah))
    return out


def _bitrate(subfolder: str) -> int:
    m = re.search(r"(\d+)kbps", subfolder, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _http_get(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https only
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def clean_name(name: str) -> str:
    """Display name without hosting-site suffixes (e.g. 'AbdulSamad QuranExplorer.Com')."""
    name = re.sub(r"\s*(QuranExplorer\.Com|KetabAllah\.Net)\s*", " ", name.replace("_", " "), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def fetch_catalog() -> list[ReciterSource]:
    try:
        data = json.loads(_http_get(CATALOG_URL).decode("utf-8"))
        entries = {k: v for k, v in data.items() if k != "ayahCount" and isinstance(v, dict)}
        raw = {v["subfolder"]: v["name"] for v in entries.values() if "subfolder" in v}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch EveryAyah catalogue (%s); using built-in fallback list", exc)
        raw = dict(FALLBACK_CATALOG)
    best: dict[str, str] = {}
    for sub, name in raw.items():
        if _EXCLUDE.search(sub):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", clean_name(name).lower()).strip()
        if key not in best or _bitrate(sub) > _bitrate(best[key]):
            best[key] = sub
    sources = [
        ReciterSource(reciter_id=sub, name=clean_name(raw[sub]), url_template=AUDIO_URL.replace("{subfolder}", sub))
        for sub in sorted(best.values())
    ]
    return sources


def load_extra_catalog(path: Path) -> list[ReciterSource]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [ReciterSource(str(i["id"]), str(i["name"]), str(i["url_template"])) for i in items]


def download(source: ReciterSource, refs: list[tuple[int, int]], cache: Path,
             workers: int) -> list[tuple[int, int, Path]]:
    folder = cache / re.sub(r"[^A-Za-z0-9_.-]", "_", source.reciter_id)
    folder.mkdir(parents=True, exist_ok=True)

    def one(ref: tuple[int, int]) -> tuple[int, int, Path] | None:
        surah, ayah = ref
        dest = folder / f"{surah:03d}{ayah:03d}.mp3"
        if dest.exists() and dest.stat().st_size > 1024:
            return surah, ayah, dest
        url = source.url_template.format(surah=surah, ayah=ayah)
        try:
            payload = _http_get(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s: failed to download %s (%s)", source.name, url, exc)
            return None
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(payload)
        tmp.replace(dest)
        return surah, ayah, dest

    results: list[tuple[int, int, Path]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, r) for r in refs]):
            if (res := fut.result()) is not None:
                results.append(res)
    return sorted(results)


def build_index(args: argparse.Namespace) -> int:
    refs = parse_ayah_spec(args.ayahs)
    texts = {ref: get_ayah_text(*ref) for ref in refs}
    sources = fetch_catalog()
    if args.extra_catalog:
        sources += load_extra_catalog(Path(args.extra_catalog))
    if args.reciters:
        pats = [p.lower() for p in args.reciters]
        sources = [s for s in sources if any(p in s.reciter_id.lower() or p in s.name.lower() for p in pats)]
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        logger.error("No reciters selected")
        return 1
    logger.info("Indexing %d reciters x %d ayahs", len(sources), len(refs))

    evaluator = QaariEvaluator(AnalysisOptions(
        aligner=args.aligner, index_dir=None, timbre_backend=args.timbre_backend, denoise="never",
    ))
    backend = evaluator.embedder.backend
    out_dir = Path(args.output)
    index = ReciterIndex(backend)
    if args.resume and (out_dir / "reciters_meta.json").exists():
        try:
            existing = ReciterIndex.load(out_dir)
            if existing.backend == backend:
                index = existing
                logger.info("Resuming: %d reciters already indexed", len(index))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resume from existing index: %s", exc)
    done = {p.reciter_id for p in index.profiles}

    cache = Path(args.cache_dir)
    for n, source in enumerate(sources, 1):
        if source.reciter_id in done:
            continue
        logger.info("[%d/%d] %s", n, len(sources), source.name)
        files = download(source, refs, cache, args.workers)
        parts = []
        for surah, ayah, path in files:
            try:
                res = evaluator.analyze_signal(load_audio(path, denoise="never"), texts[(surah, ayah)],
                                               reference=f"{surah}:{ayah}")
            except (AudioError, RuntimeError, ValueError) as exc:
                logger.warning("  %s %d:%d skipped: %s", source.name, surah, ayah, exc)
                continue
            if res.timbre is not None:
                parts.append((res.timbre, res.style))
        if len(parts) < args.min_segments:
            logger.warning("  %s: only %d usable ayahs; not indexed", source.name, len(parts))
            continue
        index.add(merge_profiles(source.reciter_id, source.name, parts, backend,
                                 source=source.url_template.split("{")[0]))
        index.save(out_dir)  # checkpoint after every reciter
    if len(index) == 0:
        logger.error("Nothing was indexed")
        return 1
    path = index.save(out_dir)
    logger.info("Wrote %d reciter profiles to %s", len(index), path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ayahs", default=DEFAULT_AYAHS, help=f"ayahs to sample (default {DEFAULT_AYAHS})")
    p.add_argument("--reciters", nargs="*", help="substring filters on reciter id/name")
    p.add_argument("--limit", type=int, help="index at most N reciters")
    p.add_argument("--extra-catalog", help="JSON list of extra reciter sources")
    p.add_argument("--output", default=str(ROOT / "index"))
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "qaari-eval" / "everyayah"))
    p.add_argument("--aligner", choices=["auto", "ctc", "heuristic"], default="auto")
    p.add_argument("--timbre-backend", choices=["auto", "ecapa", "mfcc"], default="auto")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads per reciter")
    p.add_argument("--min-segments", type=int, default=2)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    return build_index(args)


if __name__ == "__main__":
    sys.exit(main())
