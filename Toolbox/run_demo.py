from sonicbench_toolbox import SonicBenchToolbox, TaskSpec

tb = SonicBenchToolbox(output_dir="demo_outputs", sr=48000)

# Large-scale sampling 
spec = TaskSpec(
    task_type="comparison",
    attribute="distance",
    value_range=(0.0, 1.0),
    n_samples=8,
    language="en",
    seed=2026
)

res = tb.large_scale_generate("example.wav", spec)
print(res)

# User customized 
info = tb.user_customized_generate("example.wav", attribute="loudness", value=+6.0, seed=7)
print(info)
