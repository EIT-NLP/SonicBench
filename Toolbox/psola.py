import numpy as np
import librosa

def psola_pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    y = y.astype(np.float32, copy=False)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    pitches, mags = librosa.piptrack(S=S, sr=sr, hop_length=256, fmin=50, fmax=1000)
    f0 = np.zeros(pitches.shape[1], dtype=np.float32)
    for t in range(pitches.shape[1]):
        idx = mags[:, t].argmax()
        f0[t] = pitches[idx, t]
    f0 = librosa.util.fix_length(f0, size=int(np.ceil(len(y)/256)))
    f0 = np.where(f0 > 0, f0, np.nan)
    nans = np.isnan(f0)
    if np.all(nans):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)
    x = np.arange(len(f0))
    f0[nans] = np.interp(x[nans], x[~nans], f0[~nans])
    f0 = np.clip(f0, 50, 1000)

    period_hop = (sr / f0)  
    marks = []
    i = 0
    hop = 256
    pos = 0
    while pos < len(y) - 1:
        frame = min(len(period_hop)-1, int(pos / hop))
        p = float(period_hop[frame])
        marks.append(int(pos))
        pos += max(20.0, p)  
    marks = np.array(marks, dtype=np.int32)
    if len(marks) < 5:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)

    factor = 2 ** (semitones / 12.0)
    out = np.zeros_like(y, dtype=np.float32)

    for k in range(1, len(marks)-1):
        m = marks[k]
        # local period
        frame = min(len(period_hop)-1, int(m / hop))
        p = int(np.clip(period_hop[frame], 40, 2000))
        win = 2 * p
        start = m - win//2
        end = start + win
        if start < 0 or end > len(y):
            continue
        seg = y[start:end].copy()

        w = np.hanning(len(seg)).astype(np.float32)
        seg *= w

        m_syn = int(m / factor)
        s2 = m_syn - win//2
        e2 = s2 + win
        if s2 < 0 or e2 > len(out):
            continue
        out[s2:e2] += seg

    m = np.max(np.abs(out)) + 1e-12
    out = out / m * 0.98
    return out.astype(np.float32)
