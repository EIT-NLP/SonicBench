import numpy as np
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, irfft
from scipy.signal import fftconvolve
import random

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

def ensure_mono_float(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    return y

def peak_normalize(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = np.max(np.abs(y)) + 1e-12
    return (y / m) * peak

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def lufs_normalize_to(y: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)  # EBU R128
        loudness = meter.integrated_loudness(y.astype(np.float64))
        gain_db = target_lufs - loudness
        gain = 10 ** (gain_db / 20.0)
        return (y * gain).astype(np.float32)
    except Exception:
        rms = np.sqrt(np.mean(y**2) + 1e-12)
        target_rms = 10 ** ((target_lufs + 23.0) / 20.0) * 0.1  # heuristic
        g = target_rms / (rms + 1e-12)
        return (y * g).astype(np.float32)

def shelving_filter(y: np.ndarray, sr: int, shelf_gain_db: float, shelf_freq_hz: float, high_shelf: bool=True) -> np.ndarray:
    Y = rfft(y)
    freqs = np.linspace(0, sr/2, len(Y))
    gain = 10 ** (shelf_gain_db / 20.0)

    if high_shelf:
        w = 1 / (1 + np.exp(-(freqs - shelf_freq_hz) / (shelf_freq_hz * 0.1 + 1e-6)))
    else:
        w = 1 - 1 / (1 + np.exp(-(freqs - shelf_freq_hz) / (shelf_freq_hz * 0.1 + 1e-6)))

    H = 1 + (gain - 1) * w
    return irfft(Y * H).astype(np.float32)

def bandshape_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, gain_db: float) -> np.ndarray:
    sos = butter(4, [low_hz/(sr/2), high_hz/(sr/2)], btype="band", output="sos")
    band = sosfilt(sos, y)
    gain = 10 ** (gain_db / 20.0)
    return (y + (gain - 1.0) * band).astype(np.float32)

def air_absorption_filter(y: np.ndarray, sr: int, amount: float) -> np.ndarray:
    cutoff = 12000 * (1.0 - 0.85 * amount) + 800
    sos = butter(4, cutoff/(sr/2), btype="low", output="sos")
    return sosfilt(sos, y).astype(np.float32)

def fractional_delay(y: np.ndarray, sr: int, delay_seconds: float) -> np.ndarray:
    d = delay_seconds * sr
    if d <= 0:
        return y
    n = np.arange(len(y))
    ksize = 41
    t = np.arange(-ksize//2, ksize//2 + 1)
    h = np.sinc(t - d)
    h *= np.hamming(len(h))
    h /= np.sum(h) + 1e-12
    out = np.convolve(y, h, mode="same")
    return out.astype(np.float32)

def early_reflections_ir(sr: int, dist01: float) -> np.ndarray:
    base_delays_ms = np.array([6, 11, 17, 23, 31, 41], dtype=np.float32)
    delays = base_delays_ms * (1.0 + 0.8 * dist01)
    gains = np.array([0.6, 0.45, 0.35, 0.28, 0.22, 0.18], dtype=np.float32) * (0.3 + 0.9*dist01)

    n = int(sr * 0.10)  # 100ms ER window
    ir = np.zeros(n, dtype=np.float32)
    ir[0] = 1.0
    for d_ms, g in zip(delays, gains):
        idx = int((d_ms/1000.0) * sr)
        if idx < n:
            ir[idx] += g
    ir = shelving_filter(ir, sr, shelf_gain_db=-2.0*dist01, shelf_freq_hz=3000.0, high_shelf=True)
    return ir

def late_reverb_ir(sr: int, rt60: float, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    n = int(sr * max(0.3, min(3.0, rt60)))
    noise = rng.randn(n).astype(np.float32)
    t = np.arange(n) / sr
    decay = np.exp(-6.91 * t / max(1e-3, rt60)).astype(np.float32)  # -60dB at rt60
    ir = noise * decay
    ir[0] += 1.0
    ir = air_absorption_filter(ir, sr, amount=min(1.0, rt60/3.0))
    return ir.astype(np.float32)

def convolve_same(y: np.ndarray, ir: np.ndarray) -> np.ndarray:
    out = fftconvolve(y, ir, mode="same")
    return out.astype(np.float32)

def adsr_envelope(n: int, sr: int, attack_ms: float, decay_ms: float, sustain_level: float, release_ms: float) -> np.ndarray:
    a = int(sr * attack_ms / 1000.0)
    d = int(sr * decay_ms / 1000.0)
    r = int(sr * release_ms / 1000.0)
    s = max(0, n - (a + d + r))

    env = np.zeros(n, dtype=np.float32)
    idx = 0
    if a > 0:
        env[idx:idx+a] = np.linspace(0, 1, a, endpoint=False)
        idx += a
    if d > 0:
        env[idx:idx+d] = np.linspace(1, sustain_level, d, endpoint=False)
        idx += d
    if s > 0:
        env[idx:idx+s] = sustain_level
        idx += s
    if r > 0 and idx < n:
        env[idx:n] = np.linspace(sustain_level, 0, n-idx, endpoint=True)
    return env

def cepstral_envelope_warp(y: np.ndarray, sr: int, amount: float) -> np.ndarray:
    Y = rfft(y)
    mag = np.abs(Y) + 1e-12
    phase = np.angle(Y)

    logmag = np.log(mag)
    cep = np.fft.irfft(logmag)

    lifter_len = min(200, len(cep))
    env = cep.copy()
    env[lifter_len:] = 0.0
    env_logmag = np.fft.rfft(env, n=len(logmag))
    env_mag = np.exp(np.real(env_logmag))

    freqs = np.linspace(0, 1, len(env_mag))
    tilt = np.exp(amount * (freqs - 0.5))  
    new_mag = env_mag * tilt
    new_Y = new_mag * np.exp(1j * phase)

    out = irfft(new_Y, n=len(y))
    return out.astype(np.float32)
