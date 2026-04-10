import numpy as np

def wsola_time_scale(y: np.ndarray, ratio: float, sr: int, win_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    N = len(y)
    win = int(sr * win_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if win < 32:
        win = 32
    if hop < 8:
        hop = 8

    Ha = hop
    Hs = int(round(hop * ratio))
    if Hs < 1:
        Hs = 1

    w = np.hanning(win).astype(np.float32)
    out_len = int(N * ratio)
    out = np.zeros(out_len + win, dtype=np.float32)
    a = 0
    s = 0
    out[s:s+win] += y[a:a+win] * w
    search = hop

    while (a + win + Ha) < N and (s + win + Hs) < len(out):
        a_next = a + Ha
        s_next = s + Hs
        start = max(0, a_next - search)
        end = min(N - win - 1, a_next + search)

        ref = out[s_next:s_next+win] 
        if np.max(np.abs(ref)) < 1e-6:
            best = a_next
        else:
            best = a_next
            best_score = -1e18
            for cand in range(start, end, max(1, search//20)):
                seg = y[cand:cand+win]
                num = float(np.dot(ref, seg))
                den = float(np.linalg.norm(ref) * np.linalg.norm(seg) + 1e-12)
                score = num / den
                if score > best_score:
                    best_score = score
                    best = cand

        out[s_next:s_next+win] += y[best:best+win] * w
        a = best
        s = s_next

    out = out[:out_len]
    m = np.max(np.abs(out)) + 1e-12
    out = out / m * min(0.98, m)
    return out.astype(np.float32)
