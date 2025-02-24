# WIP
# Installation (linux)
1. Make sure that openal-soft is installed.
2. Clone this repository.
3. Open terminal in project's directory, run ```pip install -e .```
4. Install torchaudio ```pip install torchaudio```

# Run all experiments
```python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.paper_doom_all_basic_envs --backend=processes --num_gpus=1 --max_parallel=2  --pause_between=0 --experiments_per_gpu=2```