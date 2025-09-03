import argparse
import time
import torch
from diffusers import FluxPipeline

def format_bytes(bytes_val):
    return f"{bytes_val / (1024 ** 2):.2f} MB"

def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX.1-schnell with timing and GPU stats")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--output", type=str, default="flux-schnell.png", help="Output filename")

    args = parser.parse_args()
    timings = {}

    # Clear GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # ⏱️ Time model loading
    start = time.time()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    timings["Model loading"] = time.time() - start

    # ⏱️ Time image generation
    start = time.time()
    image = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        height=args.height,
        width=args.width
    ).images[0]
    timings["Image generation"] = time.time() - start

    # ⏱️ Time image saving
    start = time.time()
    image.save(args.output)
    timings["Image saving"] = time.time() - start

    # 📊 GPU Memory Stats
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved = torch.cuda.memory_reserved(device)
        mem_peak = torch.cuda.max_memory_allocated(device)
    else:
        mem_allocated = mem_reserved = mem_peak = 0

    # 🧾 Final Report
    print("\n🧾 Timing Report")
    print("-" * 30)
    for step, duration in timings.items():
        print(f"{step:<20}: {duration:.2f} seconds")

    print("\n📊 GPU Memory Usage")
    print("-" * 30)
    print(f"Memory Allocated     : {format_bytes(mem_allocated)}")
    print(f"Memory Reserved      : {format_bytes(mem_reserved)}")
    print(f"Peak Memory Allocated: {format_bytes(mem_peak)}")
    print("-" * 30)
    print(f"✅ Image saved to: {args.output}")

if __name__ == "__main__":
    main()

