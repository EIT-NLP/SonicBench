import numpy as np

def granular_texture(y: np.ndarray, sr: int, roughness: float, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    y = y.astype(np.float32, copy=False)
    N = len(y)

    grain_ms = 30 + 70 * (1.0 - roughness) 
    grain = int(sr * grain_ms / 1000.0)
    grain = max(64, min(grain, N//4))

    hop = int(grain * (0.25 + 0.55 * (1.0 - roughness)))  
    hop = max(16, hop)

    out = np.zeros_like(y, dtype=np.float32)
    w = np.hanning(grain).astype(np.float32)

    pos = 0
    while pos + grain < N:
        jitter = int(rng.randn() * roughness * hop)
        src = np.clip(pos + jitter, 0, N - grain - 1)
        g = y[src:src+grain] * w

        amp = 1.0 + rng.randn() * roughness * 0.2
        out[pos:pos+grain] += (g * amp).astype(np.float32)

        pos += hop

    out += rng.randn(N).astype(np.float32) * (roughness * 0.003)

    m = np.max(np.abs(out)) + 1e-12
    return (out / m * 0.98).astype(np.float32)
