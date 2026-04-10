import os, json, csv, random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import librosa
import soundfile as sf

from .dsp import (
    ensure_mono_float,
    peak_normalize,
    pad_or_trim,
    set_global_seed,
    lufs_normalize_to,
)
from .attributes import AttributeController


@dataclass
class TaskSpec:
    task_type: str                 # "recognition" or "comparison"
    attribute: str                 # 12 attrs
    value_range: Tuple[float, float]
    n_samples: int = 10
    language: str = "en"
    seed: int = 1234


class SonicBenchToolbox:
    def __init__(self, output_dir: str = "outputs", sr: int = 48000, target_sec: float = 4.0):
        self.sr = sr
        self.target_len = int(sr * target_sec)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.controller = AttributeController(sr=sr)


    def preprocess(self, audio_path: str) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        y = ensure_mono_float(y)
        y = peak_normalize(y)               
        y = pad_or_trim(y, self.target_len)   
        return y


    def user_customized_generate(
        self,
        audio_path: str,
        attribute: str,
        value: float,
        out_wav: Optional[str] = None,
        seed: int = 1234
    ) -> Dict[str, Any]:
        set_global_seed(seed)
        y = self.preprocess(audio_path)
        y2 = self.controller.apply(y, attribute, value, seed=seed)

        if out_wav is None:
            out_wav = os.path.join(self.output_dir, f"custom_{attribute}_{value:.3f}.wav")
        sf.write(out_wav, y2, self.sr)

        return {
            "mode": "user_customized",
            "attribute": attribute,
            "value": float(value),
            "input_audio": audio_path,
            "output_audio": out_wav,
            "sr": self.sr,
            "duration_sec": len(y2) / self.sr,
            "seed": seed
        }

    def large_scale_generate(
        self,
        audio_path: str,
        spec: TaskSpec
    ) -> Dict[str, str]:
        """
        Generates:
        - audio files (ref + transformed)
        - annotations.json
        - index.csv
        - optionally jsonl index
        """
        set_global_seed(spec.seed)
        rng = random.Random(spec.seed)

        y_ref = self.preprocess(audio_path)

        annotations = []
        index_rows = []

        for i in range(spec.n_samples):
            value = rng.uniform(spec.value_range[0], spec.value_range[1])
            item_seed = (spec.seed * 1000003 + i) & 0xFFFFFFFF
            y_mod = self.controller.apply(y_ref, spec.attribute, value, seed=item_seed)
            ref_name = f"ref_{i:05d}.wav"
            mod_name = f"{spec.attribute}_{value:+.3f}_{i:05d}.wav"

            ref_path = os.path.join(self.output_dir, ref_name)
            mod_path = os.path.join(self.output_dir, mod_name)

            sf.write(ref_path, y_ref, self.sr)
            sf.write(mod_path, y_mod, self.sr)

            if spec.task_type == "comparison":
                prompt = self._prompt_comparison(spec.attribute, spec.language)
                gt = "B" 
            elif spec.task_type == "recognition":
                prompt = self._prompt_recognition(spec.attribute, value, spec.language)
                gt = self._recognition_gt(spec.attribute, value)
            else:
                raise ValueError(f"Unknown task_type: {spec.task_type}")

            ann = {
                "task_id": i,
                "task_type": spec.task_type,
                "attribute": spec.attribute,
                "value": float(value),
                "reference_audio": ref_path,
                "transformed_audio": mod_path,
                "prompt": prompt,
                "ground_truth": gt,
                "seed": int(item_seed)
            }
            annotations.append(ann)
            index_rows.append([i, spec.task_type, spec.attribute, value, ref_path, mod_path, gt, item_seed])

        ann_path = os.path.join(self.output_dir, "annotations.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(self.output_dir, "index.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task_id", "task_type", "attribute", "value", "ref_audio", "trans_audio", "ground_truth", "seed"])
            w.writerows(index_rows)

        return {"annotations_json": ann_path, "index_csv": csv_path, "output_dir": self.output_dir}


    def _prompt_comparison(self, attribute: str, language: str) -> str:
        if language.lower().startswith("zh"):
            mapping = {
                "pitch": "哪个音频的音高更高？",
                "brightness": "哪个音频更明亮？",
                "loudness": "哪个音频更响？",
                "velocity": "哪个音频的起音更“用力/更强” (velocity 更高)？",
                "duration": "哪个音频更长？",
                "tempo": "哪个音频更快（节奏/速度更快）？",
                "direction": "声音更偏左还是更偏右？请选择更靠右的那个。",
                "distance": "哪个音频听起来更远？",
                "reverberation": "哪个音频混响更强？",
                "timbre": "哪个音频音色更偏亮/薄（或更偏暗/厚）？请选择更偏亮/薄的那个。",
                "texture": "哪个音频更粗糙/更颗粒/更噪？",
                "counting": "哪个音频中事件数量更多？"
            }
            return mapping.get(attribute, f"比较两段音频：哪个更{attribute}？")
        else:
            mapping = {
                "pitch": "Which clip has a higher pitch?",
                "brightness": "Which clip is brighter?",
                "loudness": "Which clip is louder?",
                "velocity": "Which clip has higher velocity (stronger attack)?",
                "duration": "Which clip is longer?",
                "tempo": "Which clip is faster (higher tempo)?",
                "direction": "Which clip is perceived more to the right?",
                "distance": "Which clip sounds farther away?",
                "reverberation": "Which clip has stronger reverberation?",
                "timbre": "Which clip has a brighter/thinner timbre?",
                "texture": "Which clip has rougher/noisier texture?",
                "counting": "Which clip contains more sound events?"
            }
            return mapping.get(attribute, f"Which clip is more {attribute}?")

    def _prompt_recognition(self, attribute: str, value: float, language: str) -> str:
        if language.lower().startswith("zh"):
            return f"这段音频在属性 {attribute} 上发生了变化。请判断变化的方向/类别。"
        return f"This clip has been modified on attribute '{attribute}'. Identify the change direction/category."

    def _recognition_gt(self, attribute: str, value: float) -> str:
        if attribute in {"counting"}:
            return str(int(round(value)))
        return "increase" if value >= 0 else "decrease"
