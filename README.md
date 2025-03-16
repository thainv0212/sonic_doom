# SonicDoom
# Installation (only linux is supported)
1. Make sure that openal-soft is installed.
2. Clone this repository.
3. Open terminal in project's directory, run ```pip install -e .```
4. Install torchaudio ```pip install torchaudio```
5. Install vizdoom ```pip install vizdoom```

# Run experiments
Run the following command to start a training run on the basic scenario

1. Vision only ```python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --train_dir=train_dir/doom_basic_no_sound```

2. Vision + fft ```python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=fft --train_dir=train_dir/doom_basic_vision_fft```

3. No vision + fft ```python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env=doom_basic --use_sound --audio_encoder=fft --train_dir=train_dir/doom_basic_no_vision_fft --encoder_conv_architecture=none```

4. No vision + fft + auto aim ```python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env=doom_basic --use_sound --audio_encoder=fft --train_dir=train_dir/doom_basic_no_vision_fft_auto_aim --encoder_conv_architecture=none --use_auto_aim_support```


5. No vision + fft + sonic aim ```python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env=doom_basic --use_sound --audio_encoder=fft --train_dir=train_dir/doom_basic_no_vision_fft_sonic_aim --encoder_conv_architecture=none --use_sonic_aim_support```

All scenarios are available [here](sf_examples/vizdoom/doom/doom_utils.py).

Training data for all scenarios is available [here.](https://drive.proton.me/urls/187JD4X2PC#kgaCjTpyEzp9).
