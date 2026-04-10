# SonicBench Toolbox

SonicBench Toolbox is a rule-based audio generation and sampling toolkit for constructing controlled perceptual evaluation datasets.

It enables reproducible stimulus creation by modifying only a single target attribute while keeping other acoustic dimensions fixed.

---

## Modes of Use

The toolbox supports two unified modes:

### 1. User Customized Mode
Users provide a short reference audio clip together with a target transformation (e.g., pitch shift, loudness change, duration scaling).  
The toolbox applies deterministic DSP rules to isolate the intended attribute contrast.

### 2. Large Scale Sampling Mode (Used in This Work)
Instead of reference clips, users provide a task specification:

- task type (comparison / recognition)
- target attribute (one or more of the 12 attributes)
- sampling constraints or value ranges

The toolbox automatically samples values and synthesizes large-scale evaluation sets.

---

## Supported Attributes (12 Total)

The toolbox provides fine-grained control over:

- **Pitch**
- **Brightness**
- **Loudness**
- **Velocity**
- **Duration**
- **Tempo**
- **Direction**
- **Distance**
- **Reverberation**
- **Timbre**
- **Texture**
- **Counting**

All transformations are implemented with deterministic signal-processing pipelines, including PSOLA pitch shifting, LUFS loudness calibration, WSOLA time-scaling, DRR-based distance control, HRTF-inspired spatial cues, convolutional reverberation, and granular texture synthesis.

---

## Installation

Install required dependencies:

```bash
pip install numpy scipy librosa soundfile pyloudnorm
```
## Output Format
Each run produces:

Audio files with explicit attribute/value naming
JSON annotations including prompts and ground truth
CSV/JSONL index files for downstream evaluation

All outputs are fully traceable and reproducible.


## Notes
Input clips should be mono, clean, and contain a single stable sound event
Toolbox is designed for dataset construction and perceptual evaluation tasks
Spatial and scene-level controls follow controlled DSP approximations (DRR, HRTF cues, granular synthesis)