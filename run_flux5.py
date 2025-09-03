import argparse
import time
import torch
import os
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
    parser = argparse.ArgumentParser(description="Generate images using FLUX.1-schnell with batching, timing, and device stats")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=0, help="Starting random seed")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--output", type=str, default="flux-schnell.png", help="Base output filename")
    parser.add_argument("--batch-size", type=int, default=1, help="Total number of images to generate")
    parser.add_argument("--sub-batch-size", type=int, default=1, help="Number of images to generate per sub-batch")
    parser.add_argument("--wait-between-sub-batches", type=float, default=0.0, help="Seconds to wait between sub-batches")

    args = parser.parse_args()

    device = get_device()
    print(f"\n🚀 Using device: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    timings = {}
    total_start = time.time()

    # Model loading
    start = time.time()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    timings["Model loading"] = time.time() - start

    # Prepare output dir
    out_dir = os.path.dirname(args.output) or "."
    base_name = os.path.splitext(os.path.basename(args.output))[0]
    ext = os.path.splitext(args.output)[1] or ".png"

    # Batch processing
    num_batches = (args.batch_size + args.sub_batch_size - 1) // args.sub_batch_size
    image_counter = 0
    sub_batch_times = []

    for batch_idx in range(num_batches):
        sub_start = time.time()
        current_sub_batch_size = min(args.sub_batch_size, args.batch_size - image_counter)
        print(f"\n📦 Sub-batch {batch_idx+1}/{num_batches} — generating {current_sub_batch_size} image(s)")

        for i in range(current_sub_batch_size):
            seed_val = args.seed + image_counter
            img_start = time.time()
            image = pipe(
                prompt=args.prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                generator=torch.Generator(device).manual_seed(seed_val),
                height=args.height,
                width=args.width
            ).images[0]
            img_time = time.time() - img_start

            filename = os.path.join(out_dir, f"{base_name}_batch{batch_idx+1}_img{i+1}_seed{seed_val}{ext}")
            image.save(filename)
            print(f"   🖼️ Saved {filename} ({img_time:.2f}s)")
            image_counter += 1

        sub_time = time.time() - sub_start
        sub_batch_times.append(sub_time)
        print(f"⏱️ Sub-batch {batch_idx+1} time: {sub_time:.2f} seconds")

        if batch_idx < num_batches - 1 and args.wait_between_sub_batches > 0:
            print(f"⏳ Waiting {args.wait_between_sub_batches} seconds before next sub-batch...")
            time.sleep(args.wait_between_sub_batches)

    total_time = time.time() - total_start

    # Memory stats
    if device.type == "cuda":
        mem_info = {
            "Memory Allocated": format_bytes(torch.cuda.memory_allocated(device)),
            "Memory Reserved": format_bytes(torch.cuda.memory_reserved(device)),
            "Peak Memory Allocated": format_bytes(torch.cuda.max_memory_allocated(device))
        }
    elif device.type == "mps":
        mem_info = {
            "Current Allocated Memory": format_bytes(torch.mps.current_allocated_memory()),
            "Driver Allocated Memory": format_bytes(torch.mps.driver_allocated_memory()),
            "Recommended Max Memory": format_bytes(torch.mps.recommended_max_memory())
        }
    else:
        mem_info = {"Info": "Memory stats not available for CPU"}

    # Final report
    print("\n🧾 Timing Report")
    print("-" * 40)
    print(f"Model loading         : {timings['Model loading']:.2f} seconds")
    for idx, t in enumerate(sub_batch_times, 1):
        print(f"Sub-batch {idx:<2} time     : {t:.2f} seconds")
    print(f"Total generation time : {total_time:.2f} seconds")
    print(f"Average per image     : {total_time / args.batch_size:.2f} seconds")

    print("\n📊 Memory Usage")
    print("-" * 40)
    for k, v in mem_info.items():
        print(f"{k:<28}: {v}")
    print("-" * 40)
    print(f"✅ Generated {args.batch_size} image(s) in {num_batches} sub-batches")

if __name__ == "__main__":
    main()

