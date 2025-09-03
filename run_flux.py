import argparse
import torch
from diffusers import FluxPipeline

def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX.1-schnell")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale (e.g., 0.0 for free generation)")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--output", type=str, default="flux-schnell.png", help="Output filename")

    args = parser.parse_args()

    # Load pipeline
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    # Generate image
    image = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        height=args.height,
        width=args.width
    ).images[0]

    # Save image
    image.save(args.output)
    print(f"✅ Image saved to {args.output}")

if __name__ == "__main__":
    main()

