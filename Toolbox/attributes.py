from __future__ import annotations
import numpy as np

from .dsp import (
    set_global_seed,
    peak_normalize,
    lufs_normalize_to,
    shelving_filter,
    bandshape_filter,
    air_absorption_filter,
    fractional_delay,
    early_reflections_ir,
    late_reverb_ir,
    convolve_same,
    adsr_envelope,
    cepstral_envelope_warp,
)
from .wsola import wsola_time_scale
from .psola import psola_pitch_shift
from .granular import granular_texture
from .events import counting_rearrange


class AttributeController:
    def __init__(self, sr: int = 48000):
        self.sr = sr

    def apply(self, y: np.ndarray, attribute: str, value: float, seed: int = 1234) -> np.ndarray:
        set_global_seed(seed)
        y_in = y.astype(np.float32, copy=False)

        if attribute == "pitch":
            y2 = psola_pitch_shift(y_in, self.sr, semitones=value)
            return peak_normalize(y2)

        if attribute == "brightness":
            gain_db = float(value) * 10.0
            y2 = shelving_filter(y_in, self.sr, shelf_gain_db=gain_db, shelf_freq_hz=2000.0, high_shelf=True)
            return peak_normalize(y2)

        if attribute == "loudness":
            base_target = -23.0
            target = base_target + float(value)
            y2 = lufs_normalize_to(y_in, self.sr, target_lufs=target)
            return peak_normalize(y2)

        if attribute == "velocity":
            v = float(np.clip(value, 0.0, 1.0))
            env = adsr_envelope(
                n=len(y_in),
                sr=self.sr,
                attack_ms=5.0 + (1.0 - v) * 60.0,
                decay_ms=30.0,
                sustain_level=0.7 + 0.25 * v,
                release_ms=50.0,
            )
            y2 = y_in * env
            return peak_normalize(y2)

        if attribute == "duration":
            ratio = float(max(0.25, min(4.0, value)))
            y2 = wsola_time_scale(y_in, ratio=ratio, sr=self.sr)
            return peak_normalize(y2)

        if attribute == "tempo":
            tempo_ratio = float(max(0.25, min(4.0, value)))
            y2 = wsola_time_scale(y_in, ratio=1.0 / tempo_ratio, sr=self.sr)
            return peak_normalize(y2)

        if attribute == "direction":
            ang = float(np.clip(value, -90.0, 90.0))
            y2 = self._hrtf_cues_mono_to_mono(y_in, angle_deg=ang)
            return peak_normalize(y2)

        if attribute == "distance":
            d = float(np.clip(value, 0.0, 1.0))
            y2 = self._distance_drr(y_in, dist01=d)
            return peak_normalize(y2)

        if attribute == "reverberation":
            wet = float(np.clip(value, 0.0, 1.0))
            y2 = self._reverb_control(y_in, wet=wet)
            return peak_normalize(y2)

        if attribute == "timbre":
            t = float(np.clip(value, -1.0, 1.0))
            y2 = cepstral_envelope_warp(y_in, sr=self.sr, amount=t)
            return peak_normalize(y2)

        if attribute == "texture":
            r = float(np.clip(value, 0.0, 1.0))
            y2 = granular_texture(y_in, sr=self.sr, roughness=r, seed=seed)
            return peak_normalize(y2)

        if attribute == "counting":
            n = int(np.clip(int(round(value)), 1, 7))
            y2 = counting_rearrange(y_in, sr=self.sr, n_events=n, seed=seed)
            return peak_normalize(y2)

        raise ValueError(f"Unknown attribute: {attribute}")

    def _hrtf_cues_mono_to_mono(self, y: np.ndarray, angle_deg: float) -> np.ndarray:
        itd_max_s = 0.00065
        itd = (angle_deg / 90.0) * itd_max_s 
        ild_db = (abs(angle_deg) / 90.0) * 10.0

        yL = y.copy()
        yR = y.copy()

        if angle_deg > 0:  
            yL = shelving_filter(yL, self.sr, shelf_gain_db=-ild_db, shelf_freq_hz=1500.0, high_shelf=True)
            yL = fractional_delay(yL, self.sr, delay_seconds=abs(itd))
        else:              
            yR = shelving_filter(yR, self.sr, shelf_gain_db=-ild_db, shelf_freq_hz=1500.0, high_shelf=True)
            yR = fractional_delay(yR, self.sr, delay_seconds=abs(itd))

        mono = 0.5 * (yL + yR)
        return mono

    def _distance_drr(self, y: np.ndarray, dist01: float) -> np.ndarray:
        direct_gain = 1.0 - 0.65 * dist01
        er_gain = 0.10 + 0.80 * dist01

        ir_er = early_reflections_ir(sr=self.sr, dist01=dist01)
        early = convolve_same(y, ir_er)

        y_direct = air_absorption_filter(y, sr=self.sr, amount=dist01)

        out = direct_gain * y_direct + er_gain * early

        out = lufs_normalize_to(out, self.sr, target_lufs=-23.0)
        return out

    def _reverb_control(self, y: np.ndarray, wet: float) -> np.ndarray:
        rt60 = 0.2 + wet * 2.0  
        ir_late = late_reverb_ir(sr=self.sr, rt60=rt60, seed=123)
        late = convolve_same(y, ir_late)

        out = (1.0 - wet) * y + wet * late
        out = lufs_normalize_to(out, self.sr, target_lufs=-23.0)
        return out
