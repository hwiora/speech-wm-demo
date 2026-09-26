# Audio demo page

Static page (`index.html`, `style.css`, `assets/`, `audio/`) with audio samples for an anonymous ICLR 2027 submission.

## Contents

- `audio/` holds the evaluated clips, copied byte for byte from the evaluation runs as 24 kHz WAV, with no re-encoding.
  `audio/prompts/` holds the WildVoice prompt recordings, unmodified (see `NOTICE`).
- `provenance.csv` lists every clip on the page with its source file, SHA-256, score file, detector, score, fixed
  threshold, threshold source and decision (`score > threshold`).
- `selection.json` sets the seed (2027) and the number of samples per model. `selection.resolved.json` records the prompts
  used in the last build.

## Selection rule

For each model, eligible prompts are TEST split, realization r0, status ok in every column, a non-empty text stream in
every generated Moshi column, and a post-hoc clip that differs from its carrier. There is no length filter. Eligible IDs
are sorted, shuffled with `random.Random(f"{seed}:{model}")`, and the first three are used. Each robustness table uses
row 1 of its model. A non-empty `prompts` list in `selection.json` overrides the draw, and the page then labels that
section as hand-picked.

## Rebuilding

`build.py` needs `local_paths.json` (not in the repository) with the locations of the evaluation outputs and the WildVoice
recordings:

```json
{"final_eval": "/path/to/final_eval", "wildvoice": "/path/to/voicebench/wildvoice"}
```

Run `python build.py`, then preview with `python -m http.server 8765`. The method name is the `METHOD` constant at the top
of `build.py`.
