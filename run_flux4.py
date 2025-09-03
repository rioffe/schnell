import argparse
import time
import torch
from diffusers import FluxPipeline

def format_bytes(bytes_val):
    return f"{bytes_val / (1024 ** 2):.2f} MB"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX.1-schnell with timing and device stats")
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

    device = get_device()
    print(f"\n🚀 Using device: {device}")

    # Reset memory stats if CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # ⏱️ Model loading
    start = time.time()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    timings["Model loading"] = time.time() - start

    # ⏱️ Image generation
    start = time.time()
    image = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        generator=torch.Generator(device).manual_seed(args.seed),
        height=args.height,
        width=args.width
    ).images[0]
    timings["Image generation"] = time.time() - start

    # ⏱️ Image saving
    start = time.time()
    image.save(args.output)
    timings["Image saving"] = time.time() - start

    # 📊 Memory Stats
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved = torch.cuda.memory_reserved(device)
        mem_peak = torch.cuda.max_memory_allocated(device)
        mem_info = {
            "Memory Allocated": format_bytes(mem_allocated),
            "Memory Reserved": format_bytes(mem_reserved),
            "Peak Memory Allocated": format_bytes(mem_peak)
        }
    elif device.type == "mps":
        mem_allocated = torch.mps.current_allocated_memory()
        mem_driver = torch.mps.driver_allocated_memory()
        mem_recommended = torch.mps.recommended_max_memory()
        mem_info = {
            "Current Allocated Memory": format_bytes(mem_allocated),
            "Driver Allocated Memory": format_bytes(mem_driver),
            "Recommended Max Memory": format_bytes(mem_recommended)
        }
    else:
        mem_info = {"Info": "Memory stats not available for CPU"}

    # 🧾 Final Report
    print("\n🧾 Timing Report")
    print("-" * 30)
    for step, duration in timings.items():
        print(f"{step:<20}: {duration:.2f} seconds")

    print("\n📊 Memory Usage")
    print("-" * 30)
    for k, v in mem_info.items():
        print(f"{k:<28}: {v}")
    print("-" * 30)
    print(f"✅ Image saved to: {args.output}")

if __name__ == "__main__":
    main()

