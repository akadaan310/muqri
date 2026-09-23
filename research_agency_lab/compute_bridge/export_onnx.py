#!/usr/bin/env python3
"""Export muaalem-v3.2 to ONNX (fp32 + dynamic INT8) for the on-device SDK.

The exported graph takes the feature-extractor output and returns **one** tensor: the per-frame
log-softmax of every CTC level concatenated in the dump's column order (phonemes first, then the ten
sifat levels alphabetically) — byte-identical in layout to the raw ``<id>.f32`` files written by
``muaalem_dump.py``, so ``layout.json`` describes the ONNX output too and every Julia/Octave analysis
and the C++ SDK read the same thing.

    .venv/bin/python -m research_agency_lab.compute_bridge.export_onnx export  [--out DIR]
    .venv/bin/python -m research_agency_lab.compute_bridge.export_onnx verify  [--dump DIR] [--clips N]
    .venv/bin/python -m research_agency_lab.compute_bridge.export_onnx bench   [--threads 1,4]

`export` writes model_fp32.onnx, model_int8.onnx and layout.json.
`verify` checks ONNX vs torch and INT8 vs fp32 on real clips, and — the gate that actually matters —
that the greedy phoneme decode is unchanged.
`bench` reports the real-time factor per thread count (SDK KPI 8).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))

OUT = ROOT / "research_agency_lab/experiments/sdk/onnx"
DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T10"
MODEL = "obadx/muaalem-model-v3_2"
OPSET = 17


def _load_torch():  # type: ignore[no-untyped-def]
    import torch

    import muaalem_dump as md  # noqa: F401  (applies the transformers shim via muaalem_eval)
    from muaalem_eval import Muaalem

    model = Muaalem(MODEL, device="cpu", dtype=torch.float32)
    return torch, md, model


class _Concat:
    """Wrapper turning the model's dict of per-level logits into one log-softmax tensor."""

    def __new__(cls, torch, model, order):  # type: ignore[no-untyped-def]
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = model
                self.order = order

            def forward(self, input_features, attention_mask):  # type: ignore[no-untyped-def]
                out = self.m(input_features=input_features, attention_mask=attention_mask,
                             return_dict=False)[0]
                return torch.cat([torch.log_softmax(out[k].float(), dim=-1) for k in self.order], dim=-1)

        return Mod()


def export(out_dir: Path) -> int:
    torch, md, mu = _load_torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(mu.model.config.level_to_vocab_size, key=md._level_order)

    wave = np.zeros(16000 * 3, dtype=np.float32)
    feats = mu.processor([wave], sampling_rate=16000, return_tensors="pt")
    args = (feats["input_features"].float(), feats["attention_mask"])

    wrapper = _Concat(torch, mu.model, order).eval()
    with torch.no_grad():
        ref = wrapper(*args)
    print(f"levels {order}\nsample output {tuple(ref.shape)} (frames × {ref.shape[-1]} cols)")

    # 605 M params = 2.4 GB of fp32 weights, over protobuf's 2 GB single-file ceiling, so torch spills
    # every tensor to a separate side-car file. Export into its own directory, then consolidate the
    # spill into ONE external-data file so the SDK ships two files, not eight hundred.
    raw = out_dir / "fp32_raw"
    raw.mkdir(parents=True, exist_ok=True)
    tmp = raw / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, args, str(tmp), opset_version=OPSET,
            input_names=["input_features", "attention_mask"], output_names=["logprobs"],
            dynamic_axes={"input_features": {0: "batch", 1: "frames_in"},
                          "attention_mask": {0: "batch", 1: "frames_in"},
                          "logprobs": {0: "batch", 1: "frames_out"}},
            dynamo=False,
        )

    import onnx

    fp32 = out_dir / "model_fp32.onnx"
    m = onnx.load(str(tmp))  # pulls the spilled tensors back in
    onnx.save_model(m, str(fp32), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model_fp32.onnx.data", size_threshold=1024, convert_attribute=False)
    del m
    shutil.rmtree(raw)
    data = out_dir / "model_fp32.onnx.data"
    print(f"fp32 onnx: {fp32.stat().st_size / 1e6:.1f} MB graph + "
          f"{data.stat().st_size / 1e6:.1f} MB weights")

    # the dump layout, so the SDK and Julia read the ONNX output with the same column map
    lay = out_dir / "layout.json"
    md.write_layout(lay, mu, {k: np.zeros((1, mu.model.config.level_to_vocab_size[k])) for k in order})
    print(f"layout: {lay}")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Naive dynamic quantization of every op destroys this model (measured: 6.9 max abs error on
    # log-probs, 1/3 of clips decoding differently). The Conformer depthwise/pointwise convolutions and
    # the embedding Gathers are the sensitive parts, so quantize only the MatMuls, per output channel.
    int8 = out_dir / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                     per_channel=True, reduce_range=False, op_types_to_quantize=["MatMul"],
                     extra_options={"MatMulConstBOnly": True})
    print(f"int8 onnx: {int8.stat().st_size / 1e6:.1f} MB  "
          f"({(fp32.stat().st_size + data.stat().st_size) / max(int8.stat().st_size, 1):.2f}× smaller than fp32)")
    return 0


def _clips(dump: Path, n: int) -> list[dict]:  # type: ignore[type-arg]
    rows = [json.loads(l) for l in (dump / "index.jsonl").open()]
    return [r for r in rows if "file" in r][:n]


def _session(path: Path, threads: int = 0):  # type: ignore[no-untyped-def]
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _decode(lp: np.ndarray, width: int) -> str:
    """Greedy CTC collapse of the phoneme block (first `width` columns), blank = 0."""
    ids = lp[:, :width].argmax(axis=1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(int(i))
        prev = int(i)
    return ",".join(map(str, out))


def verify(out_dir: Path, dump: Path, n: int) -> int:
    torch, md, mu = _load_torch()
    order = sorted(mu.model.config.level_to_vocab_size, key=md._level_order)
    wrapper = _Concat(torch, mu.model, order).eval()
    width = int(mu.model.config.level_to_vocab_size["phonemes"])

    s32, s8 = _session(out_dir / "model_fp32.onnx"), _session(out_dir / "model_int8.onnx")
    rows = _clips(dump, n)
    print(f"verifying {len(rows)} clips from {dump.name}")

    e_onnx, e_int8, dec_bad32, dec_bad8 = [], [], 0, 0
    for r in rows:
        wave = md.load_16k(_wav_for(r, dump))
        f = mu.processor([wave], sampling_rate=16000, return_tensors="pt")
        a = {"input_features": f["input_features"].float().numpy(), "attention_mask": f["attention_mask"].numpy()}
        with torch.no_grad():
            ref = wrapper(f["input_features"].float(), f["attention_mask"]).numpy()[0]
        o32 = s32.run(None, a)[0][0]
        o8 = s8.run(None, a)[0][0]
        e_onnx.append(float(np.abs(o32 - ref).max()))
        e_int8.append(float(np.abs(o8 - o32).max()))
        d_ref = _decode(ref, width)
        dec_bad32 += _decode(o32, width) != d_ref
        dec_bad8 += _decode(o8, width) != d_ref

    res = {"clips": len(rows),
           "onnx_fp32_vs_torch_max_abs": max(e_onnx), "onnx_fp32_vs_torch_mean_abs": float(np.mean(e_onnx)),
           "int8_vs_fp32_max_abs": max(e_int8), "int8_vs_fp32_mean_abs": float(np.mean(e_int8)),
           "decode_mismatch_fp32": dec_bad32, "decode_mismatch_int8": dec_bad8,
           "decode_agreement_int8": 1 - dec_bad8 / max(len(rows), 1)}
    print(json.dumps(res, indent=1))
    (out_dir / "parity.json").write_text(json.dumps(res, indent=1))
    ok = res["onnx_fp32_vs_torch_max_abs"] < 1e-3 and dec_bad32 == 0
    print("GATE fp32 parity:", "PASS" if ok else "FAIL")
    print(f"GATE int8 decode agreement: {res['decode_agreement_int8']:.3f}")
    return 0 if ok else 1


def _wav_for(rec: dict, dump: Path) -> Path:  # type: ignore[type-arg]
    """Local audio for a dump record (EveryAyah mp3 cache)."""
    import muaalem_dump as md

    s, a = int(rec["sura"]), int(rec["aya"])
    return md.EVERYAYAH / str(rec["speaker"]) / f"{s:03d}{a:03d}.mp3"


def bench(out_dir: Path, dump: Path, threads: list[int], n: int = 8) -> int:
    torch, md, mu = _load_torch()
    rows = _clips(dump, n)
    waves = [md.load_16k(_wav_for(r, dump)) for r in rows]
    audio_s = sum(len(w) / 16000 for w in waves)
    feats = [mu.processor([w], sampling_rate=16000, return_tensors="pt") for w in waves]

    out = {"clips": len(rows), "audio_s": round(audio_s, 1), "threads": {}}
    for name, path in (("fp32", out_dir / "model_fp32.onnx"), ("int8", out_dir / "model_int8.onnx")):
        for t in threads:
            s = _session(path, t)
            a0 = {"input_features": feats[0]["input_features"].float().numpy(),
                  "attention_mask": feats[0]["attention_mask"].numpy()}
            s.run(None, a0)  # warm up
            t0 = time.time()
            for f in feats:
                s.run(None, {"input_features": f["input_features"].float().numpy(),
                             "attention_mask": f["attention_mask"].numpy()})
            wall = time.time() - t0
            out["threads"][f"{name}_t{t}"] = {"wall_s": round(wall, 2), "rtf": round(wall / audio_s, 4),
                                              "x_realtime": round(audio_s / wall, 2)}
            print(f"{name} threads={t}: {wall:.2f}s for {audio_s:.1f}s audio  RTF={wall / audio_s:.4f} "
                  f"({audio_s / wall:.1f}× real time)")
    (out_dir / "bench.json").write_text(json.dumps(out, indent=1))
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["export", "verify", "bench"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dump", type=Path, default=DUMP)
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--threads", default="1,4")
    a = p.parse_args(argv)
    if a.cmd == "export":
        return export(a.out)
    if a.cmd == "verify":
        return verify(a.out, a.dump, a.clips)
    return bench(a.out, a.dump, [int(x) for x in a.threads.split(",")], a.clips)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
