import numpy as np

def counting_rearrange(y: np.ndarray, sr: int, n_events: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    y = y.astype(np.float32, copy=False)
    N = len(y)

    win = int(sr * 0.25)
    win = max(256, min(win, N//2))
    energy = np.convolve(y*y, np.ones(win, dtype=np.float32), mode="same")
    center = int(np.argmax(energy))
    seg_len = int(sr * 0.20)
    seg_len = max(256, min(seg_len, N//3))

    s = max(0, center - seg_len//2)
    e = min(N, s + seg_len)
    seg = y[s:e].copy()

    out = np.zeros_like(y, dtype=np.float32)
    gap = N // n_events
    for i in range(n_events):
        base = i * gap
        jitter = int(rng.randn() * (0.03 * sr))
        pos = int(np.clip(base + jitter, 0, N - len(seg) - 1))
        out[pos:pos+len(seg)] += seg

    m = np.max(np.abs(out)) + 1e-12
    return (out / m * 0.98).astype(np.float32)
