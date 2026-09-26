"""Build the audio demo page from the evaluation runs.

Everything under ``final_eval/runs/main`` is read, never written. Audio files are copied byte for
byte (no re-encoding), and every score and decision on the page is copied from the per-clip score
tables at the paper's fixed identity threshold, with the strict rule ``score > threshold``. The
page is rendered from ``template.html`` and every copied clip is listed in ``provenance.csv``.

Selection rule (per section, see ``selection.json``): TEST split, realization r0, status ok in every
column, a non-empty text stream in every generated Moshi column, and a post-hoc clip that differs
from its carrier. Eligible prompt IDs are sorted, shuffled with ``random.Random(f"{seed}:{section}")``
and the first ``n`` are taken. A section with a non-empty ``prompts`` list uses it as given and is
labelled hand-picked. Each robustness block uses the first row of its backbone's sample section.

Usage: python build.py
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ---------------------------------------------------------------------------------------- names
METHOD = "Ours"  # method name placeholder: change here, it fills the whole page
TITLE = "Tokens Change, Structure Endures: Spectral Watermarking for Generated Speech"
PAPER_URL = ""  # OpenReview link, filled after review

HERE = Path(__file__).resolve().parent
# Machine-specific source locations live in local_paths.json (not in the repo):
# {"final_eval": "<path to final_eval>", "wildvoice": "<path to voicebench/wildvoice>"}
_LOCAL = json.loads((HERE / "local_paths.json").read_text())
FINAL = Path(_LOCAL["final_eval"])
RUNS = FINAL / "runs/main"
WILDVOICE = Path(_LOCAL["wildvoice"])
AUDIO = HERE / "audio"
SR = 24000
sys.path.insert(0, str(FINAL / "src"))
from feval.thresholds import strict_threshold  # noqa: E402  (campaign's own threshold rule)

CTRL = re.compile(r"<0x[0-9A-Fa-f]+>")
CODEC = {"mimi": "Mimi", "encodec6k": "EnCodec 6 kbps", "dac16": "DAC 16 kHz",
         "cosy_resyn": "CosyVoice3 codec (native)", "moss_resyn": "MOSS-TTS codec (native)"}
BACKBONE = {"moshi": "Moshi", "cosyvoice3": "CosyVoice3", "moss": "MOSS-TTS Delay-8B"}

# Arms of record and their detectors. thr = ("csv", file, arm, detector) reads thr_fixed at clean;
# ("posthoc", backbone, detector) reads posthoc_test.csv; ("wmar", tag) applies strict_threshold to
# the arm's own FT-decoded CAL nulls exactly as tools/wmar_cross_report.py does.
ARMS = {
    "moshi": {
        "null": {"arm": "null", "label": "Unwatermarked"},
        "ours": {"arm": "spectral_v3_s4_d070", "label": METHOD, "z": True,
                 "scores": "pm2/scores_pm2_moshi.csv", "det": "spectral_native_s4",
                 "thr": ("csv", "report/pm2/test_s4_pm2_detection.csv")},
        "kgw": {"arm": "kgw_d2_s4", "label": "KGW", "z": True,
                "scores": "pm2/scores_pm2_moshi.csv", "det": "kgw_s4",
                "thr": ("csv", "report/pm2/test_s4_pm2_detection.csv")},
        "null_wmar": {"arm": "null_wmar_ft", "label": "Unwatermarked (FT decoder)"},
        "wmar": {"arm": "wmar_ft", "label": "WMAR FT", "z": True,
                 "scores": "detect/scores_wmar_ft.csv", "det": "kgw_s4", "thr": ("wmar", "ft")},
        "craw": {"arm": "posthoc_craw", "label": "CRAW (post-hoc)", "z": False,
                 "scores": "detect/scores_posthoc.csv", "det": "craw", "thr": ("posthoc", "moshi")},
    },
    "cosyvoice3": {
        "null": {"arm": "cosy_null", "label": "Unwatermarked"},
        "ours": {"arm": "cosy_v3_d090", "label": METHOD, "z": True,
                 "scores": "pm2/scores_pm2_cosy.csv", "det": "spectral_native",
                 "thr": ("csv", "report/pm2/cosy_test_pm2_detection.csv")},
        "kgw": {"arm": "cosy_kgw_d2", "label": "KGW", "z": True,
                "scores": "pm2/scores_pm2_cosy.csv", "det": "cosy_kgw",
                "thr": ("csv", "report/pm2/cosy_test_pm2_detection.csv")},
        "craw": {"arm": "posthoc_craw", "label": "CRAW (post-hoc)", "z": False,
                 "scores": "detect/scores_posthoc.csv", "det": "craw", "thr": ("posthoc", "cosyvoice3")},
    },
    "moss": {
        "null": {"arm": "moss_null", "label": "Unwatermarked"},
        "ours": {"arm": "moss_lib16_d090", "label": METHOD, "z": True,
                 "scores": "moss_lib16/scores_pm2_mosslib16.csv", "det": "spectral_native_lib16",
                 "thr": ("csv", "report/mosslib16/mosslib16_test_pm2_detection.csv")},
        "kgw": {"arm": "moss_kgw_d2_s16", "label": "KGW", "z": True,
                "scores": "moss_lib16/scores_pm2_mosslib16.csv", "det": "moss_kgw_s16",
                "thr": ("csv", "report/mosslib16/mosslib16_test_pm2_detection.csv")},
        "craw": {"arm": "posthoc_craw", "label": "CRAW (post-hoc)", "z": False,
                 "scores": "detect/scores_posthoc.csv", "det": "craw", "thr": ("posthoc", "moss")},
    },
}
SAMPLE_COLS = {"moshi": ["null", "ours", "kgw", "null_wmar", "wmar", "craw"],
               "cosyvoice3": ["null", "ours", "kgw", "craw"], "moss": ["null", "ours", "kgw", "craw"]}
ROBUST_COLS = {"moshi": ["null", "ours", "kgw", "wmar", "craw"],
               "cosyvoice3": ["null", "ours", "kgw"], "moss": ["null", "ours", "kgw"]}
ROBUST_CONDS = {"moshi": [("mimi", 1), ("mimi", 8), ("encodec6k", 8), ("dac16", 8)],
                "cosyvoice3": [("cosy_resyn", 8)], "moss": [("moss_resyn", 8)]}


def rd(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kw)


# ---------------------------------------------------------------------------------------- paths
def gen_rows(bb: str) -> dict[tuple[str, str], dict]:
    test = set(json.loads((RUNS / "splits" / f"{bb}.json").read_text())["splits"]["TEST"])
    out = {}
    with open(RUNS / "generate" / bb / "manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "TEST" and r["realization"] == 0:
                assert r["prompt_id"] in test, r["clip_id"]
                out[(r["arm"], r["prompt_id"])] = r
    return out, test


def clip_id(bb: str, arm: str, pid: str) -> str:
    return f"{bb}/{arm}/natural/{pid}/r0"


def posthoc_clip(bb: str, pid: str) -> Path:
    return RUNS / "posthoc" / bb / "craw" / "clips" / f"{bb}__posthoc_craw__natural__{pid}__r0.wav"


def attacked(bb: str, arm: str, pid: str, codec: str, n: int) -> Path:
    name = f"{bb}__{arm}__natural__{pid}__r0.wav"
    if bb == "moshi":
        root = RUNS / ("attack_posthoc/moshi" if arm.startswith("posthoc_") else "attack/full")
        return root / codec / f"pass_{n}" / name
    roots = {("cosyvoice3", "cosy_v3_d090"): "attack_cosy_test/resyn", ("cosyvoice3", "cosy_kgw_d2"): "attack_cosy/resyn",
             ("cosyvoice3", "cosy_null"): "attack_cosy/resyn", ("moss", "moss_lib16_d090"): "attack_mosslib16/test/resyn",
             ("moss", "moss_kgw_d2_s16"): "attack_moss16/resyn", ("moss", "moss_null"): "attack_moss/resyn"}
    return RUNS / roots[(bb, arm)] / codec / f"pass_{n}" / name


def source_wav(bb: str, col: str, pid: str, man: dict, cond: str) -> Path:
    arm = ARMS[bb][col]["arm"]
    if cond == "clean":
        return posthoc_clip(bb, pid) if col == "craw" else Path(man[(arm, pid)]["wav"])
    codec, n = cond.rsplit("_x", 1)
    return attacked(bb, arm, pid, codec, int(n))


def clean_text(t: str) -> str:
    return " ".join(CTRL.sub(" ", t or "").split())


# ---------------------------------------------------------------------------------------- selection
def differs(a: Path, b: Path) -> bool:
    x, _ = sf.read(a, dtype="float32")
    y, _ = sf.read(b, dtype="float32")
    n = min(len(x), len(y))
    return len(x) != len(y) or float(np.max(np.abs(x[:n] - y[:n]))) > 1e-6


def eligible(bb: str, man: dict, pid: str) -> bool:
    for col in SAMPLE_COLS[bb]:
        if col == "craw":
            p = posthoc_clip(bb, pid)
            if not p.exists():
                return False
            continue
        r = man.get((ARMS[bb][col]["arm"], pid))
        if r is None or r.get("status") != "ok" or not Path(r["wav"]).exists():
            return False
        if bb == "moshi" and not clean_text(r.get("text", "")):
            return False
    if bb != "moshi":
        texts = {man[(ARMS[bb][c]["arm"], pid)]["text"] for c in SAMPLE_COLS[bb] if c != "craw"}
        if len(texts) != 1:
            return False
    if bb == "moshi" and not (WILDVOICE / "wav" / f"{pid}.wav").exists():
        return False
    return True


def select(bb: str, spec: dict, seed: int, man: dict, test: set[str]) -> tuple[list[str], bool]:
    if spec.get("prompts"):
        for p in spec["prompts"]:
            if p not in test or not eligible(bb, man, p):
                raise SystemExit(f"{bb}: hand-picked {p} is not an eligible TEST prompt")
        return list(spec["prompts"]), True
    pool = sorted(p for p in test if eligible(bb, man, p))
    random.Random(f"{seed}:{bb}").shuffle(pool)
    picked = []
    for p in pool:
        null = Path(man[(ARMS[bb]["null"]["arm"], p)]["wav"])
        if not differs(posthoc_clip(bb, p), null):  # post-hoc watermark actually embedded
            continue
        picked.append(p)
        if len(picked) == spec["n"]:
            break
    return picked, False


# ---------------------------------------------------------------------------------------- scores
class Scores:
    """Per-clip scores and fixed thresholds, copied from the evaluation outputs."""

    def __init__(self) -> None:
        self.cache: dict[tuple, pd.DataFrame] = {}
        self.thr: dict[tuple[str, str], tuple[float, str]] = {}

    def load(self, rel: str, clip_ids: set[str]) -> pd.DataFrame:
        key = (rel, frozenset(clip_ids))
        if key not in self.cache:
            parts = []
            for ch in rd(RUNS / rel, usecols=["clip_id", "condition", "detector", "z"], chunksize=2_000_000):
                parts.append(ch[ch.clip_id.isin(clip_ids)])
            self.cache[key] = pd.concat(parts).drop_duplicates(["clip_id", "condition", "detector"])
        return self.cache[key]

    def threshold(self, bb: str, col: str) -> tuple[float, str]:
        key = (bb, col)
        if key in self.thr:
            return self.thr[key]
        a = ARMS[bb][col]
        kind = a["thr"][0]
        if kind == "csv":
            d = rd(RUNS / a["thr"][1])
            row = d[(d.arm == a["arm"]) & (d.detector == a["det"]) & (d.condition == "clean")]
            assert len(row) == 1, (bb, col)
            out = (float(row.thr_fixed.iloc[0]), f"{a['thr'][1]} (arm={a['arm']}, detector={a['det']}, condition=clean, thr_fixed)")
        elif kind == "posthoc":
            d = rd(RUNS / "report/posthoc/posthoc_test.csv")
            row = d[(d.backbone == a["thr"][1]) & (d.detector == a["det"]) & (d.condition == "clean")]
            assert len(row) == 1, (bb, col)
            out = (float(row.threshold.iloc[0]), f"report/posthoc/posthoc_test.csv (backbone={a['thr'][1]}, detector=craw, condition=clean, threshold)")
        else:  # WMAR: same call as tools/wmar_cross_report.py
            tag = a["thr"][1]
            s = rd(RUNS / a["scores"], usecols=["arm", "realization", "condition", "detector", "z"])
            cal = s[(s.arm == f"null_wmar_{tag}") & (s.realization.isin([1, 2, 3])) & (s.condition == "clean")]
            out = (float(strict_threshold(cal.z.values, 0.01).threshold),
                   f"strict_threshold(0.01) on null_wmar_{tag} r1-3 clean in {a['scores']} (n={len(cal)}), as in tools/wmar_cross_report.py")
        self.thr[key] = out
        return out

    def z(self, bb: str, col: str, pid: str, cond: str, ids: set[str]) -> float:
        a = ARMS[bb][col]
        s = self.load(a["scores"], ids)
        cid = clip_id(bb, a["arm"], pid)
        row = s[(s.clip_id == cid) & (s.condition == cond) & (s.detector == a["det"])]
        if len(row) != 1:
            raise SystemExit(f"no unique score for {cid} {cond} {a['det']} in {a['scores']}")
        return float(row.z.iloc[0])


def check_k_fixed(sc: Scores) -> None:
    """Reproduce published TPR counts at the thresholds used on the page (guards against a wrong source)."""
    checks = [("moshi", "wmar", "clean", "report/wmar_cross_test.csv", "ft|ft_enc|tpr"),
              ("moshi", "wmar", "mimi_x8", "report/wmar_cross_test.csv", "ft|ft_enc|tpr")]
    for bb, col, cond, rel, key in checks:
        a = ARMS[bb][col]
        thr, _ = sc.threshold(bb, col)
        s = rd(RUNS / a["scores"], usecols=["clip_id", "arm", "realization", "condition", "detector", "z"])
        w = s[(s.arm == a["arm"]) & (s.realization == 0) & (s.condition == cond) & (s.detector == a["det"])]
        w = w.drop_duplicates(["clip_id", "condition", "detector"])
        mine = float((w.z > thr).mean())
        ref = rd(RUNS / rel)
        pub = float(ref[ref.condition == cond][key].iloc[0])
        assert abs(mine - pub) < 1e-9, (bb, col, cond, mine, pub)
        print(f"check {bb}/{col}/{cond}: {mine:.4f} == {pub:.4f} (thr {thr:.4f})")
    d = rd(RUNS / "report/posthoc/posthoc_test.csv")
    for bb in ("moshi", "cosyvoice3", "moss"):
        for cond in ("clean", "mimi_x8"):
            thr, _ = sc.threshold(bb, "craw")
            s = rd(RUNS / "detect/scores_posthoc.csv", usecols=["clip_id", "arm", "condition", "detector", "z"])
            w = s[(s.clip_id.str.startswith(f"{bb}/")) & (s.arm == "posthoc_craw") & (s.condition == cond)
                  & (s.detector == "craw")].drop_duplicates(["clip_id", "condition", "detector"])
            mine = float((w.z > thr).mean())
            pub = float(d[(d.backbone == bb) & (d.detector == "craw") & (d.condition == cond)].tpr.iloc[0])
            assert abs(mine - pub) < 1e-9, (bb, cond, mine, pub)
            print(f"check {bb}/craw/{cond}: {mine:.4f} == {pub:.4f} (thr {thr:.4f})")


# ---------------------------------------------------------------------------------------- audio
def riff_chunks(path: Path) -> list[str]:
    with open(path, "rb") as f:
        head = f.read(12)
        assert head[:4] == b"RIFF" and head[8:12] == b"WAVE", path
        ids = []
        while True:
            h = f.read(8)
            if len(h) < 8:
                return ids
            cid, size = h[:4].decode("latin1"), struct.unpack("<I", h[4:])[0]
            ids.append(cid)
            f.seek(size + (size & 1), 1)


ALLOWED_CHUNKS = {"fmt ", "data", "fact", "PEAK"}  # PEAK: libsndfile peak amplitude, no text


def copy_audio(src: Path, rel: str, expect_sr: int | None = SR) -> str:
    dst = AUDIO / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = sf.info(src)
    if expect_sr is not None and info.samplerate != expect_sr:
        raise SystemExit(f"{src}: {info.samplerate} Hz, expected {expect_sr}")
    extra = set(riff_chunks(src)) - ALLOWED_CHUNKS
    if extra:
        raise SystemExit(f"{src}: unexpected RIFF chunks {extra} (possible metadata)")
    shutil.copyfile(src, dst)
    return f"audio/{rel}"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------- html
def mark(det: bool | None, z: float | None, thr: float | None) -> str:
    if det is None:
        return ""
    cls, txt = ("yes", "Detected") if det else ("no", "Not detected")
    num = f'<span class="z">z = {z:.2f} · threshold {thr:.2f}</span>' if z is not None else ""
    return f'<div class="mark {cls}"><span class="dot"></span>{txt}{num}</div>'


def tile(label: str, path: str, det=None, z=None, thr=None, said: str = "", note: str = "", kind: str = "") -> str:
    said_html = f'<p class="said">{html.escape(said)}</p>' if said else ""
    note_html = f'<p class="note">{note}</p>' if note else ""
    return (f'<div class="tile {kind}"><div class="lab">{html.escape(label)}</div>'
            f'<audio controls preload="none" src="{path}"></audio>{mark(det, z, thr)}{note_html}{said_html}</div>')


def pick_badge(hand: bool) -> str:
    return ('<span class="pick hand">Hand-picked by the authors</span>' if hand
            else '<span class="pick">Drawn by the stated rule, not hand-picked</span>')


# ---------------------------------------------------------------------------------------- main
def main() -> None:
    sel = json.loads((HERE / "selection.json").read_text())
    seed = int(sel["seed"])
    if AUDIO.exists():
        shutil.rmtree(AUDIO)  # the page's own copies only; rebuilt from runs/ every time
    questions = {}
    with open(WILDVOICE / "manifest.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            assert Path(r["audio_filepath"]).stem == f"wildvoice_{i:04d}"
            questions[f"wildvoice_{i:04d}"] = r["text"]

    sc = Scores()
    prov: list[dict] = []
    parts: dict[str, str] = {}
    resolved = {}
    for bb in ("moshi", "cosyvoice3", "moss"):
        man, test = gen_rows(bb)
        pids, hand = select(bb, sel["sections"][bb], seed, man, test)
        resolved[bb] = {"prompts": pids, "hand_picked": hand}
        print(f"{bb:11s} {'hand' if hand else 'rule'} {pids}")
        ids = {clip_id(bb, ARMS[bb][c]["arm"], p) for c in ARMS[bb] for p in pids}

        def cell(col: str, pid: str, cond: str, row_label: str, section: str, said: str = "", note: str = "",
                 _bb=bb, _man=man, _ids=ids) -> str:
            a = ARMS[_bb][col]
            src = source_wav(_bb, col, pid, _man, cond)
            if not src.exists():
                raise SystemExit(f"missing {src}")
            page = copy_audio(src, f"{section}/{pid}/{cond}/{col}.wav")
            det = z = thr = None
            rec = {"section": section, "row": row_label, "backbone": _bb, "prompt_id": pid, "column": a["label"],
                   "arm": a["arm"], "condition": cond, "page_file": page, "source_file": str(src.relative_to(FINAL)),
                   "sha256": sha256(src), "score_file": "", "detector": "", "clip_id": "", "score": "",
                   "threshold": "", "threshold_source": "", "rule": "", "decision": ""}
            if "scores" in a:
                zz = sc.z(_bb, col, pid, cond, _ids)
                thr, thr_src = sc.threshold(_bb, col)
                det = zz > thr
                z = zz if a["z"] else None
                rec.update(score_file=a["scores"], detector=a["det"], clip_id=clip_id(_bb, a["arm"], pid),
                           score=repr(zz), threshold=repr(thr), threshold_source=thr_src, rule="score > threshold",
                           decision="detected" if det else "not detected")
            prov.append(rec)
            return tile(a["label"], page, det, z, thr, said=said, note=note, kind=col)

        # ---- sample rows
        cards = []
        for i, p in enumerate(pids):
            tiles = []
            if bb == "moshi":
                pr = copy_audio(WILDVOICE / "wav" / f"{p}.wav", f"prompts/{p}.wav", expect_sr=None)
                prov.append({"section": "moshi", "row": str(i + 1), "backbone": bb, "prompt_id": p,
                             "column": "Spoken prompt (WildVoice)", "arm": "", "condition": "", "page_file": pr,
                             "source_file": f"data/voicebench/wildvoice/wav/{p}.wav", "sha256": sha256(WILDVOICE / "wav" / f"{p}.wav"),
                             "score_file": "", "detector": "", "clip_id": "", "score": "", "threshold": "",
                             "threshold_source": "", "rule": "", "decision": ""})
                head = (f'<div class="prompt"><div class="lab">Spoken prompt (human recording)</div>'
                        f'<audio controls preload="none" src="{pr}"></audio>'
                        f'<p class="q">“{html.escape(questions[p])}”</p></div>')
                groups = [("Token domain", ["null", "ours", "kgw"]), ("WMAR (fine-tuned Mimi decoder)", ["null_wmar", "wmar"]),
                          ("Post-hoc waveform watermark", ["craw"])]
            else:
                text = man[(ARMS[bb]["null"]["arm"], p)]["text"]
                head = f'<div class="prompt"><div class="lab">Sentence read</div><p class="q">“{html.escape(text)}”</p></div>'
                groups = [("Token domain", ["null", "ours", "kgw"]), ("Post-hoc waveform watermark", ["craw"])]
            for gname, cols in groups:
                ts = []
                for c in cols:
                    said = ""
                    if bb == "moshi" and c != "craw":
                        said = clean_text(man[(ARMS[bb][c]["arm"], p)]["text"])
                    note = "Embedded into the Unwatermarked clip of this row." if c == "craw" else ""
                    ts.append(cell(c, p, "clean", str(i + 1), bb, said=said, note=note))
                tiles.append(f'<div class="group"><div class="gname">{gname}</div><div class="tiles">{"".join(ts)}</div></div>')
            cards.append(f'<article class="card"><div class="rowno">{i + 1}</div>{head}{"".join(tiles)}</article>')
        parts[bb.upper()] = pick_badge(hand) + "".join(cards)

        # ---- robustness: first sample row of this backbone
        p = pids[0]
        cols = ROBUST_COLS[bb]
        conds = ["clean"] + [f"{c}_x{n}" for c, n in ROBUST_CONDS[bb]]
        th = "".join(f"<th>{html.escape(ARMS[bb][c]['label'])}</th>" for c in cols)
        body = []
        for cond in conds:
            if cond == "clean":
                name = "As generated"
            else:
                codec, n = cond.rsplit("_x", 1)
                name = f"{CODEC[codec]} × {n}"
            tds = "".join(f"<td>{cell(c, p, cond, name, 'robust_' + bb)}</td>" for c in cols)
            body.append(f"<tr><th class='cond'>{name}</th>{tds}</tr>")
        cap = (f"Row 1 of the {BACKBONE[bb]} samples" +
               (f" (question: “{html.escape(questions[p])}”)" if bb == "moshi" else f" (“{html.escape(man[(ARMS[bb]['null']['arm'], p)]['text'])}”)"))
        parts["ROBUST_" + bb.upper()] = (f'<p class="lede">{cap}.</p><div class="scroll"><table class="robust">'
                                         f'<thead><tr><th></th>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')

    check_k_fixed(sc)
    parts.update(METHOD=html.escape(METHOD), TITLE=html.escape(TITLE), SEED=str(seed),
                 PAPER=(f'<a class="btn" href="{PAPER_URL}">Paper</a>' if PAPER_URL
                        else '<span class="btn disabled">Paper (OpenReview link after review)</span>'))
    page = (HERE / "template.html").read_text()
    for k, v in parts.items():
        page = page.replace("{{" + k + "}}", v)
    assert "{{" not in page, re.findall(r"\{\{\w+\}\}", page)
    (HERE / "index.html").write_text(page)
    with open(HERE / "provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0]))
        w.writeheader()
        w.writerows(prov)
    (HERE / "selection.resolved.json").write_text(json.dumps({"seed": seed, **resolved}, indent=2) + "\n")
    wavs = list(AUDIO.rglob("*.wav"))
    print(f"wrote index.html, provenance.csv; {len(wavs)} wav files, {sum(x.stat().st_size for x in wavs) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
