#!/usr/bin/env python3
"""qaari-eval command line and HTTP API runner.

Examples::

    python main.py analyze --audio user_recitation.wav --surah 113 --ayah 1
    python main.py analyze --audio fatiha.wav --surah 1 --tareeq shatibiyyah --mode taraweeh_adapted --benchmark dosari
    python main.py rules --surah 1 --ayah 7
    python main.py serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from app.aligner import AlignmentError
from app.audio import AudioError
from app.pipeline import AnalysisOptions, QaariEvaluator
from app.quran_text import QuranTextError, get_ayah_text
from app.tajweed_rules import TajweedParseError, TajweedParser

STATUS_MARK = {"PASS": "✔", "WARNING": "!", "FAIL": "✘", "SKIPPED": "·"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qaari-eval", description="Quranic Tajweed analysis and reciter fingerprinting")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze a recitation audio file")
    a.add_argument("--audio", required=True, help="WAV/MP3/FLAC/OGG recitation")
    a.add_argument("--surah", type=int)
    a.add_argument("--ayah", type=int, help="first ayah (omit to analyse the whole surah)")
    a.add_argument("--ayah-end", type=int, help="last ayah of a multi-ayah recording")
    a.add_argument("--text", help="target Uthmani text (overrides --surah/--ayah lookup)")
    a.add_argument("--tareeq", choices=["shatibiyyah", "tayyibah"], default="shatibiyyah",
                   help="Shatibiyyah (Munfasil 4-5) or Tayyibah with qasr al-munfasil (2)")
    a.add_argument("--mode", choices=["auto", "studio", "taraweeh_adapted"], default="auto",
                   help="taraweeh_adapted: dereverberation, local tempo and breath-aware waqf")
    a.add_argument("--benchmark", help="compare with a reciter from the indices, e.g. dosari, hussary")
    a.add_argument("--no-sifaat", action="store_true", help="skip the articulation-quality (sifaat) checks")
    a.add_argument("--aligner", choices=["auto", "ctc", "heuristic", "json"], default="auto")
    a.add_argument("--aligner-model", help="HuggingFace CTC model id")
    a.add_argument("--alignment-json", help="pre-computed letter alignment (see README)")
    a.add_argument("--index-dir", default="index", help="directory holding the reciter FAISS index")
    a.add_argument("--top-k", type=int, default=3)
    a.add_argument("--style-weight", type=float, default=0.6,
                   help="weight of the Tajweed vector vs. timbre in reciter matching (0 = timbre only)")
    a.add_argument("--timbre-backend", choices=["auto", "ecapa", "mfcc"], default="auto")
    a.add_argument("--no-stop", action="store_true", help="the recitation continues past the ayah end (wasl)")
    a.add_argument("--denoise", choices=["auto", "always", "never"], default="auto")
    a.add_argument("--output", default="analysis_report.json", help="where to save the JSON report")
    a.add_argument("--include-alignment", action="store_true", help="add letter timings to the JSON report")
    a.add_argument("--json", action="store_true", help="print the JSON report instead of the text summary")
    a.add_argument("--offline", action="store_true", help="never fetch ayah text from the network")

    r = sub.add_parser("rules", help="list the Tajweed rules required by an ayah")
    r.add_argument("--surah", type=int)
    r.add_argument("--ayah", type=int)
    r.add_argument("--text")
    r.add_argument("--no-stop", action="store_true")
    r.add_argument("--tareeq", choices=["shatibiyyah", "tayyibah"], default="shatibiyyah")
    r.add_argument("--no-sifaat", action="store_true")

    s = sub.add_parser("serve", help="run the HTTP API (requires fastapi + uvicorn)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--index-dir", default="index")
    s.add_argument("--aligner", choices=["auto", "ctc", "heuristic"], default="auto")
    return p


def _fmt_num(v: Any, unit: str = "") -> str:
    return "n/a" if v is None else f"{v}{unit}"


def render_text_report(report: dict[str, Any]) -> str:
    s = report["recitation_summary"]
    env = s.get("acoustic_environment") or {}
    lines = [
        "═" * 78,
        f" qaari-eval report {('— ' + s['reference']) if s.get('reference') else ''}  "
        f"[{s.get('tareeq', '')}, {s.get('mode', '')}]",
        f" {s['text'][:300]}",
        "═" * 78,
        f" Tajweed perfection index : {_fmt_num(s['tajweed_perfection_index'])} / 100",
        f" Sifaat score             : {_fmt_num(s.get('sifaat_score'))} / 100",
        " Category scores          : " + (", ".join(f"{k} {v}" for k, v in s["category_scores"].items()) or "n/a"),
        f" Base harakah             : {s['base_haraka_duration_ms']} ms  ({s['tempo_bpm_harakat']} harakat/min, "
        f"{s.get('pace', '')})",
        f" Rules evaluated          : {s['total_rules_evaluated']}  "
        + " ".join(f"{k}={v}" for k, v in s["status_counts"].items() if v),
        f" Alignment                : {s['alignment']['method']} (confidence {s['alignment']['mean_confidence']})",
        f" Environment              : RT60 {_fmt_num(env.get('rt60_reverberation_time'), 's')}, "
        f"SNR {_fmt_num(env.get('snr_db'), ' dB')}",
    ]
    if s.get("taraweeh_adapter"):
        lines.append(f" Taraweeh adapter         : {', '.join(s['taraweeh_adapter']['applied'])}")
    lines.append("─" * 78)
    for d in report["detailed_rule_diagnostics"]:
        mark = STATUS_MARK.get(d["status"], "?")
        loc = d["location"]
        counts = ""
        if "measured_harakat" in d:
            rng = d.get("expected_harakat_range")
            exp = f"{rng[0]:g}-{rng[1]:g}" if rng and rng[0] != rng[1] else f"{d.get('expected_harakat', '')}"
            counts = f" [{d['measured_harakat']:.1f}/{exp}]"
        lines.append(f" {mark} {d['status'][:7]:<7} {d['rule_type']:<22} {d['word']:<14} "
                     f"{loc['start_ms']:>6}-{loc['end_ms']:<6}ms{counts}")
        lines.append(f"     {d['feedback']}")
    for note in s.get("notes", []):
        lines.append(f" note: {note}")
    lines.append("─" * 78)
    match = report["reciter_similarity_match"]
    if match.get("top_matches"):
        w = match.get("style_weight", 0.6)
        lines.append(f" Closest reciters (S = {w:g}·tajweed + {1 - w:g}·timbre):")
        for m in match["top_matches"]:
            lines.append(f"   • {m['reciter_name']:<36} [{m.get('index', '')}] "
                         f"tajweed {m['style_similarity_pct']:5.1f}%  timbre {m['timbre_similarity_pct']:5.1f}%")
            if m["matched_traits"]:
                lines.append(f"       {', '.join(m['matched_traits'])}")
    else:
        lines.append(f" Reciter matching: {match.get('note', 'unavailable')}")
    bench = report.get("benchmark_comparison")
    if bench:
        lines.append(f" Benchmark: {bench.get('benchmark')}"
                     + (f" (timbre similarity {bench['timbre_similarity_pct']}%)" if bench.get("timbre_similarity_pct")
                        is not None else ""))
        for row in bench.get("features", [])[:12]:
            lines.append(f"   {row['feature']:<36} you {row['you']:>9}  benchmark {row['benchmark']:>9}")
        if bench.get("note"):
            lines.append(f"   {bench['note']}")
    for w in s.get("warnings", []):
        lines.append(f" ⚠ {w}")
    lines.append("═" * 78)
    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace) -> int:
    opts = AnalysisOptions(
        aligner=args.aligner, aligner_model=args.aligner_model, alignment_json=args.alignment_json,
        index_dir=args.index_dir, top_k=args.top_k, style_weight=args.style_weight, timbre_backend=args.timbre_backend,
        stop_at_end=not args.no_stop, denoise=args.denoise, include_alignment=args.include_alignment,
        allow_network=not args.offline, tareeq=args.tareeq, mode=args.mode, benchmark=args.benchmark,
        include_sifaat=not args.no_sifaat,
    )
    result = QaariEvaluator(opts).analyze_file(args.audio, surah=args.surah, ayah=args.ayah, ayah_end=args.ayah_end,
                                               text=args.text)
    out = Path(args.output)
    out.write_text(json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result.report, ensure_ascii=False, indent=2))
    else:
        print(render_text_report(result.report))
        print(f"Saved JSON report to {out}")
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    text = args.text or get_ayah_text(args.surah, args.ayah)
    parsed = TajweedParser(stop_at_end=not args.no_stop, tareeq=args.tareeq,
                           include_sifaat=not args.no_sifaat).parse(text)
    print(parsed.text)
    for r in parsed.rules:
        exp = f" expected {r.expected_harakat[0]:g}-{r.expected_harakat[1]:g} harakat" if r.expected_harakat else ""
        print(f"  {r.rule_type.value:<22} {r.word:<16} {r.letter or '':<2} {r.detail}{exp}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from app.api import create_app

    try:
        import uvicorn
    except ImportError:
        print("The API needs extras: pip install fastapi uvicorn python-multipart", file=sys.stderr)
        return 2
    uvicorn.run(create_app(AnalysisOptions(index_dir=args.index_dir, aligner=args.aligner)),
                host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "analyze":
            if args.text is None and args.surah is None:
                print("error: provide --text or --surah (with --ayah for a single ayah)", file=sys.stderr)
                return 2
            return cmd_analyze(args)
        if args.command == "rules":
            if args.text is None and (args.surah is None or args.ayah is None):
                print("error: provide --text or both --surah and --ayah", file=sys.stderr)
                return 2
            return cmd_rules(args)
        return cmd_serve(args)
    except (AudioError, AlignmentError, QuranTextError, TajweedParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
