  uv venv --python=3.11
  source .venv/bin/activate
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install accelerate diffusers transformers protobuf sentencepiece 
  time python run_flux5.py   --prompt "Pink elephant in a modern conference room"   --guidance_scale 1.5   --num_inference_steps 6   --max_sequence_length 256   --seed 777   --width 1280   --height 720   --batch-size 16 --output pink_elephant.png
