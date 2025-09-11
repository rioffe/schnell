## UNIX installation
```
uv venv --python=3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install accelerate diffusers transformers protobuf sentencepiece 
time python run_flux5.py   --prompt "Pink elephant in a modern conference room"   --guidance_scale 1.5   --num_inference_steps 6   --max_sequence_length 256   --seed 777   --width 1280   --height 720   --batch-size 16 --output pink_elephant.png
```

## MacOS installation
```
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.7.0 torchvision torchaudio==2.7.0 --index-url https://pypi.apple.com/simple
uv pip install transformers protobuf accelerate diffusers sentencepiece
```
