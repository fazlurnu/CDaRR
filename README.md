# sweep_full_adaptive (adaptive-only sweep)

This is a refactor of `sweep_sampling_full_adaptive.py` into a small package with a single entrypoint:
`sweep_full_adaptive/sweep_adaptive.py`.

## Install / Run

From your project root:

```bash
python sweep_full_adaptive/sweep_adaptive.py --script adaptive_sampling/adaptive_main.py
```

The script you pass must accept the adaptive flags:
`--out-dir --tag --dpsi --reception-prob --pos-uncertainty --vel-uncertainty --seed --adaptive ...`
