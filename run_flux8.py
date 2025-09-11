import argparse
import time
import torch
import os
import json
import platform
import sys
import subprocess
import psutil
from datetime import datetime
from diffusers import FluxPipeline
try:
    from importlib.metadata import distributions
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import distributions

def format_bytes(bytes_val):
    return f"{bytes_val / (1024 ** 2):.2f} MB"

def format_gb(bytes_val):
    return f"{bytes_val / (1024 ** 3):.2f} GB"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_ram_usage():
    """Get current RAM usage statistics"""
    memory = psutil.virtual_memory()
    return {
        "total_ram": memory.total,
        "total_ram_gb": format_gb(memory.total),
        "available_ram": memory.available,
        "available_ram_gb": format_gb(memory.available),
        "used_ram": memory.used,
        "used_ram_gb": format_gb(memory.used),
        "ram_percent": memory.percent,
        "process_memory": psutil.Process().memory_info().rss,
        "process_memory_gb": format_gb(psutil.Process().memory_info().rss)
    }

def get_cuda_info():
    """Get CUDA and driver version information"""
    cuda_info = {}
    
    if torch.cuda.is_available():
        cuda_info["cuda_available"] = True
        cuda_info["cuda_version"] = torch.version.cuda
        cuda_info["cudnn_version"] = torch.backends.cudnn.version()
        cuda_info["cuda_device_count"] = torch.cuda.device_count()
        
        # Get device properties
        device_props = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_props.append({
                "device_id": i,
                "name": props.name,
                "total_memory": props.total_memory,
                "total_memory_gb": format_gb(props.total_memory),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor
            })
        cuda_info["device_properties"] = device_props
        
        # Try to get driver version using nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            cuda_info["driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            # Fallback: try nvidia-smi command
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    cuda_info["driver_version"] = result.stdout.strip().split('\n')[0]
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                cuda_info["driver_version"] = "Unable to determine"
    else:
        cuda_info["cuda_available"] = False
        cuda_info["reason"] = "CUDA not available"
    
    return cuda_info

def get_system_info():
    """Collect comprehensive system information"""
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "release": platform.release(),
            "version": platform.version()
        },
        "python": {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro
            },
            "executable": sys.executable
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
    }
    
    # Add CUDA information
    system_info["cuda"] = get_cuda_info()
    
    # Add initial RAM info
    system_info["initial_ram"] = get_ram_usage()
    
    # Get installed package versions using modern importlib.metadata
    try:
        installed_packages = {}
        key_packages = [
            'torch', 'diffusers', 'transformers', 'accelerate', 'safetensors',
            'pillow', 'numpy', 'pynvml', 'tokenizers', 'huggingface-hub', 'psutil'
        ]
        
        # Get all installed packages using modern API
        all_packages = {dist.metadata['name'].lower(): dist.version for dist in distributions()}
        
        # Focus on key packages first
        for pkg in key_packages:
            if pkg.lower() in all_packages:
                installed_packages[pkg] = all_packages[pkg.lower()]
        
        # Add other relevant packages
        for pkg_name, version in all_packages.items():
            if any(keyword in pkg_name for keyword in ['torch', 'cuda', 'nvidia', 'diffus', 'transform']):
                if pkg_name not in installed_packages:
                    installed_packages[pkg_name] = version
                    
        system_info["packages"] = installed_packages
        
    except Exception as e:
        system_info["packages"] = {"error": f"Could not retrieve package information: {str(e)}"}
    
    return system_info

def save_json_report(args, timings, mem_info, ram_info, system_info, sub_batch_times, total_time, output_file):
    """Save comprehensive report to JSON file"""
    report = {
        "system_info": system_info,
        "generation_parameters": {
            "prompt": args.prompt,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "max_sequence_length": args.max_sequence_length,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "batch_size": args.batch_size,
            "sub_batch_size": args.sub_batch_size,
            "wait_between_sub_batches": args.wait_between_sub_batches,
            "output_pattern": args.output
        },
        "performance": {
            "model_loading_time": timings.get("Model loading", 0),
            "sub_batch_times": sub_batch_times,
            "total_generation_time": total_time,
            "average_time_per_image": total_time / args.batch_size,
            "images_per_second": args.batch_size / total_time if total_time > 0 else 0,
            "num_sub_batches": len(sub_batch_times)
        },
        "memory_usage": {
            "gpu_memory": mem_info,
            "ram_usage": ram_info
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📄 Performance and system data saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")

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
    parser.add_argument("--json", type=str, help="Save performance data and system info to JSON file")

    args = parser.parse_args()

    device = get_device()
    print(f"\n🚀 Using device: {device}")

    # Collect system information early
    system_info = get_system_info()
    initial_ram = get_ram_usage()
    
    print(f"💾 Initial RAM: {initial_ram['used_ram_gb']} used / {initial_ram['total_ram_gb']} total ({initial_ram['ram_percent']:.1f}%)")
    print(f"🔧 Process RAM: {initial_ram['process_memory_gb']}")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    timings = {}
    total_start = time.time()

    # Model loading
    start = time.time()
    print("\n📥 Loading FLUX.1-schnell model...")
    if device.type == "cuda":
        # Load with CPU offloading enabled from the start to avoid GPU OOM
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    else:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.to(device)
    timings["Model loading"] = time.time() - start
    
    # Check RAM after model loading
    post_load_ram = get_ram_usage()
    print(f"💾 RAM after model load: {post_load_ram['used_ram_gb']} used ({post_load_ram['ram_percent']:.1f}%)")
    print(f"🔧 Process RAM: {post_load_ram['process_memory_gb']}")

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
    final_ram = get_ram_usage()

    # Memory stats
    if device.type == "cuda":
        mem_info = {
            "device_type": "cuda",
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_allocated_mb": format_bytes(torch.cuda.memory_allocated(device)),
            "memory_allocated_gb": format_gb(torch.cuda.memory_allocated(device)),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "memory_reserved_mb": format_bytes(torch.cuda.memory_reserved(device)),
            "memory_reserved_gb": format_gb(torch.cuda.memory_reserved(device)),
            "peak_memory_allocated": torch.cuda.max_memory_allocated(device),
            "peak_memory_allocated_mb": format_bytes(torch.cuda.max_memory_allocated(device)),
            "peak_memory_allocated_gb": format_gb(torch.cuda.max_memory_allocated(device))
        }
    elif device.type == "mps":
        mem_info = {
            "device_type": "mps",
            "current_allocated_memory": torch.mps.current_allocated_memory(),
            "current_allocated_memory_mb": format_bytes(torch.mps.current_allocated_memory()),
            "current_allocated_memory_gb": format_gb(torch.mps.current_allocated_memory()),
            "driver_allocated_memory": torch.mps.driver_allocated_memory(),
            "driver_allocated_memory_mb": format_bytes(torch.mps.driver_allocated_memory()),
            "driver_allocated_memory_gb": format_gb(torch.mps.driver_allocated_memory()),
            "recommended_max_memory": torch.mps.recommended_max_memory(),
            "recommended_max_memory_mb": format_bytes(torch.mps.recommended_max_memory()),
            "recommended_max_memory_gb": format_gb(torch.mps.recommended_max_memory())
        }
    else:
        mem_info = {
            "device_type": "cpu",
            "info": "GPU memory stats not available for CPU"
        }

    # RAM usage summary
    ram_info = {
        "initial": initial_ram,
        "after_model_load": post_load_ram,
        "final": final_ram,
        "peak_process_memory": max(initial_ram["process_memory"], post_load_ram["process_memory"], final_ram["process_memory"]),
        "peak_process_memory_gb": format_gb(max(initial_ram["process_memory"], post_load_ram["process_memory"], final_ram["process_memory"])),
        "ram_increase_during_generation": final_ram["process_memory"] - post_load_ram["process_memory"],
        "ram_increase_during_generation_gb": format_gb(final_ram["process_memory"] - post_load_ram["process_memory"])
    }

    # Save JSON report if requested
    if args.json:
        save_json_report(args, timings, mem_info, ram_info, system_info, sub_batch_times, total_time, args.json)

    # Final console report
    print("\n🧾 Timing Report")
    print("-" * 40)
    print(f"Model loading         : {timings['Model loading']:.2f} seconds")
    for idx, t in enumerate(sub_batch_times, 1):
        print(f"Sub-batch {idx:<2} time     : {t:.2f} seconds")
    print(f"Total generation time : {total_time:.2f} seconds")
    print(f"Average per image     : {total_time / args.batch_size:.2f} seconds")

    print("\n🖥️ GPU Memory Usage")
    print("-" * 40)
    if mem_info["device_type"] == "cuda":
        print(f"{'Memory Allocated':<28}: {mem_info['memory_allocated_gb']}")
        print(f"{'Memory Reserved':<28}: {mem_info['memory_reserved_gb']}")
        print(f"{'Peak Memory Allocated':<28}: {mem_info['peak_memory_allocated_gb']}")
    elif mem_info["device_type"] == "mps":
        print(f"{'Current Allocated Memory':<28}: {mem_info['current_allocated_memory_gb']}")
        print(f"{'Driver Allocated Memory':<28}: {mem_info['driver_allocated_memory_gb']}")
        print(f"{'Recommended Max Memory':<28}: {mem_info['recommended_max_memory_gb']}")
    else:
        print(f"{'Info':<28}: {mem_info['info']}")

    print("\n💾 RAM Usage")
    print("-" * 40)
    print(f"{'Initial Process RAM':<28}: {ram_info['initial']['process_memory_gb']}")
    print(f"{'After Model Load':<28}: {ram_info['after_model_load']['process_memory_gb']}")
    print(f"{'Final Process RAM':<28}: {ram_info['final']['process_memory_gb']}")
    print(f"{'Peak Process RAM':<28}: {ram_info['peak_process_memory_gb']}")
    print(f"{'System RAM Used':<28}: {ram_info['final']['used_ram_gb']} / {ram_info['final']['total_ram_gb']} ({ram_info['final']['ram_percent']:.1f}%)")
    
    print("-" * 40)
    print(f"✅ Generated {args.batch_size} image(s) in {num_batches} sub-batches")

if __name__ == "__main__":
    main()
