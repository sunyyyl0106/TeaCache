"""
Nirvana-style cross-request latent cache for TeaCache video (Open-Sora).
Uses k_values = [5, 10, 15]; CLIP for prompt similarity; FAISS + KMinHeapCache.
"""
import os
import re
import time
import queue
import argparse
import json
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import faiss
from transformers import CLIPModel, CLIPProcessor

from serving_system_N import evict_from_faiss, KMinHeapCache

# Video pipeline + TeaCache
from videosys import OpenSoraConfig, VideoSysEngine
from eval.teacache.experiments.opensora import teacache_forward

# Default generation params (fixed for cache compatibility)
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_NUM_FRAMES = 51
K_VALUES_VIDEO = [5, 10, 15]


def request_scheduler_video(
    req_queue,
    selected_requests,
    start_time,
    index,
    cache,
    new_cache_queue,
    cached_requests,
    final_text_embeddings,
    k_values,
    processor,
    clip_model,
    worker_status,
    log_file="request_throughput_video_teacache.csv",
):
    device = clip_model.device
    agg_k_distribution = {k: 0 for k in k_values}
    minute = 0
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("timestamp,request_rate,throughput\n")
    last_check_time_queue = time.time()
    request_count_per_min = 0
    size_of_queues = 0

    for _, row in selected_requests.iterrows():
        while time.time() - start_time < row["seconds_from_start"]:
            time.sleep(0.1)

        request_arrival_time = time.time()
        row["start_time"] = request_arrival_time

        while not new_cache_queue.empty():
            cache_data = new_cache_queue.get()
            new_cached_latents = cache_data["cached_latents"]
            new_cached_prompt = cache_data["prompt"]
            new_query_embedding = cache_data["query_embedding"]
            while len(cache.item_map) + len(cache.k_values) > cache.max_size:
                evicted_index = cache.evict()
                if evicted_index is not None:
                    del cached_requests[evicted_index]
                    index, final_text_embeddings = evict_from_faiss(
                        index, final_text_embeddings, evicted_index
                    )

            num_embeddings = index.ntotal
            cached_requests.append(new_cached_prompt)
            index.add(new_query_embedding)
            final_text_embeddings = np.concatenate(
                (final_text_embeddings, new_query_embedding), axis=0
            )
            for idx, k in enumerate(k_values):
                cache.insert(num_embeddings, 0, k, new_cached_latents[idx])

        prompt = row["prompt"]
        texts = processor(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=77,
        ).to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**texts).cpu()
        query_embedding = text_embedding.numpy().reshape(1, -1)

        if index.ntotal == 0:
            row["cached"] = None
            row["k"] = None
            row["latent"] = None
            row["query_embedding"] = text_embedding.clone()
            req_queue.put(row.to_dict())
        else:
            distances, indices = index.search(query_embedding, k=1)
            closest_prompt = cached_requests[indices[0][0]]
            closest_texts = processor(
                text=closest_prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=77,
            ).to(device)
            with torch.no_grad():
                closest_text_embedding = clip_model.get_text_features(**closest_texts)
            text_embedding_device = text_embedding.to(device)
            with torch.no_grad():
                text_norm = text_embedding_device / text_embedding_device.norm(
                    dim=-1, keepdim=True
                )
                closest_text_norm = closest_text_embedding / closest_text_embedding.norm(
                    dim=-1, keepdim=True
                )
                text_similarity_scores = torch.matmul(text_norm, closest_text_norm.T)
            text_similarity_scores = torch.clamp(text_similarity_scores, min=0)
            text_embedding = text_embedding_device.cpu()

            if text_similarity_scores.item() > 0.65:
                if text_similarity_scores.item() > 0.95:
                    closest_index = 15
                elif text_similarity_scores.item() > 0.85:
                    closest_index = 10
                else:
                    closest_index = 5
                best_candidate = cache.retrieve(closest_index, indices[0][0])
                if best_candidate:
                    score, (idex, k_i, latent) = best_candidate
                    row["cached"] = True
                    row["k"] = k_i
                    row["latent"] = latent.clone().to(dtype=torch.float32).cpu()
                    row["query_embedding"] = text_embedding.clone()
                    req_queue.put(row.to_dict())
                    agg_k_distribution[k_i] += 1
                else:
                    row["cached"] = None
                    row["k"] = None
                    row["latent"] = None
                    row["query_embedding"] = text_embedding.clone()
                    req_queue.put(row.to_dict())
            else:
                row["cached"] = None
                row["k"] = None
                row["latent"] = None
                row["query_embedding"] = text_embedding.clone()
                req_queue.put(row.to_dict())

        request_count_per_min += 1
        current_time = time.time()
        if current_time - last_check_time_queue >= 60:
            elapsed_time = current_time - last_check_time_queue
            minute += 1
            new_size_of_queues = req_queue.qsize()
            throughput = size_of_queues + request_count_per_min - new_size_of_queues
            size_of_queues = new_size_of_queues
            with open(log_file, "a") as f:
                f.write(
                    f"{minute},{request_count_per_min / elapsed_time * 60},{throughput / elapsed_time * 60}\n"
                )
            request_count_per_min = 0
            last_check_time_queue = current_time

    while not req_queue.empty():
        while not new_cache_queue.empty():
            cache_data = new_cache_queue.get()
            new_cached_latents = cache_data["cached_latents"]
            new_cached_prompt = cache_data["prompt"]
            new_query_embedding = cache_data["query_embedding"]
            while len(cache.item_map) + len(cache.k_values) > cache.max_size:
                evicted_index = cache.evict()
                if evicted_index is not None:
                    del cached_requests[evicted_index]
                    index, final_text_embeddings = evict_from_faiss(
                        index, final_text_embeddings, evicted_index
                    )
            num_embeddings = index.ntotal
            cached_requests.append(new_cached_prompt)
            index.add(new_query_embedding)
            final_text_embeddings = np.concatenate(
                (final_text_embeddings, new_query_embedding), axis=0
            )
            for idx, k in enumerate(k_values):
                cache.insert(num_embeddings, 0, k, new_cached_latents[idx])

    while True:
        if req_queue.empty():
            all_done = all(
                status in ["finished", "dropped"] for status in worker_status.values()
            )
            if all_done:
                break
        time.sleep(1)


def worker_video(
    gpu_id,
    req_queue,
    new_cache_queue,
    latency_queue,
    worker_status,
    video_directory,
    resolution,
    aspect_ratio,
    num_frames,
    teacache_thresh,
):
    device = f"cuda:{gpu_id}"
    config = OpenSoraConfig(num_gpus=1, num_sampling_steps=30)
    engine = VideoSysEngine(config)

    # TeaCache patch
    trans = engine.driver_worker.transformer
    trans.__class__.enable_teacache = True
    trans.__class__.rel_l1_thresh = teacache_thresh
    trans.__class__.accumulated_rel_l1_distance = 0
    trans.__class__.previous_modulated_input = None
    trans.__class__.previous_residual = None
    trans.__class__.forward = teacache_forward

    pipeline = engine.driver_worker
    idle_counter = 0
    max_idle_iterations = 100

    while True:
        try:
            request = req_queue.get(timeout=10)
            idle_counter = 0
            prompt = request["prompt"]
            clean_prompt = re.sub(r"[^\w\-_\.]", "_", prompt)[:200]
            out_path = os.path.join(
                video_directory, f"{clean_prompt}_{request.get('start_time', 0)}.mp4"
            )

            if request["cached"] is None:
                result = pipeline.generate(
                    prompt,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    num_frames=num_frames,
                    seed=42,
                    verbose=False,
                    collect_latents_at_steps=tuple(K_VALUES_VIDEO),
                )
                if isinstance(result, tuple):
                    output, collected_latents = result
                else:
                    output = result
                    collected_latents = None
                video = output.video[0]
                engine.save_video(video, out_path)
                if collected_latents is not None and request.get("query_embedding") is not None:
                    cached_latents = [z.cpu() for z in collected_latents]
                    qe = request["query_embedding"]
                    qe_np = qe.numpy().reshape(1, -1) if hasattr(qe, "numpy") else np.array(qe).reshape(1, -1)
                    new_cache_queue.put(
                        {
                            "cached_latents": cached_latents,
                            "prompt": prompt,
                            "query_embedding": qe_np,
                        }
                    )
            else:
                cache_latent = request["latent"].unsqueeze(0)
                k = request["k"]
                result = pipeline.generate(
                    prompt,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    num_frames=num_frames,
                    seed=42,
                    verbose=False,
                    cache_latent=cache_latent,
                    cache_start_step=k,
                )
                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result
                video = output.video[0]
                engine.save_video(video, out_path)

            finish_time = time.time() - request["start_time"]
            latency_queue.put(finish_time)

        except queue.Empty:
            idle_counter += 1
            if idle_counter >= max_idle_iterations:
                worker_status[gpu_id] = "dropped"
                break
            continue


def generate_rapidly_increasing_seconds_from_start(
    num_requests, min_rate=2, max_rate=9, duration=100 * 60
):
    min_rate_per_sec = min_rate / 60
    max_rate_per_sec = max_rate / 60
    x = np.linspace(-2, 6, num_requests)
    sigmoid_growth = 1 / (1 + np.exp(-1.5 * x))
    request_rates = min_rate_per_sec + (max_rate_per_sec - min_rate_per_sec) * sigmoid_growth
    interarrival_times = 1 / np.maximum(request_rates, 1e-3)
    return np.cumsum(interarrival_times)


def main():
    parser = argparse.ArgumentParser(
        description="TeaCache video serving with Nirvana-style cross-request latent cache"
    )
    parser.add_argument("--num_req", type=int, default=50, help="number of requests")
    parser.add_argument("--cache_size", type=int, default=1000, help="cache size (entries)")
    parser.add_argument(
        "--video_directory",
        type=str,
        default="./video_outputs_teacache",
        help="directory for generated videos",
    )
    parser.add_argument(
        "--prompt_list",
        type=str,
        default=None,
        help="JSON file with list of prompts (e.g. VBench_full_info.json); each item can have 'prompt_en'",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=DEFAULT_RESOLUTION,
        help="resolution (e.g. 480p)",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default=DEFAULT_ASPECT_RATIO,
        help="aspect ratio (e.g. 9:16)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="number of frames",
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.15,
        help="TeaCache rel_l1_thresh (higher = more skip)",
    )
    args = parser.parse_args()

    os.makedirs(args.video_directory, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices")

    # Prompts
    if args.prompt_list and os.path.isfile(args.prompt_list):
        with open(args.prompt_list, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            prompts = [d.get("prompt_en", d.get("prompt", str(d))) for d in data]
        else:
            prompts = [str(p) for p in data]
    else:
        prompts = [
            "A cat walking on the street.",
            "Ocean waves under sunset.",
            "A dog running in the park.",
        ] * max(1, (args.num_req + 2) // 3)
    prompts = prompts[: args.num_req]

    # Request schedule
    seconds_from_start = generate_rapidly_increasing_seconds_from_start(
        len(prompts), min_rate=0.5, max_rate=4
    )
    selected_requests = pd.DataFrame(
        {"prompt": prompts, "seconds_from_start": seconds_from_start}
    )

    # Empty initial cache
    embedding_dim = 768
    index = faiss.IndexFlatL2(embedding_dim)
    final_text_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
    cached_requests = []
    # KMinHeapCache with empty initial state and k_values [5, 10, 15]
    cache = KMinHeapCache(
        max_size=args.cache_size * len(K_VALUES_VIDEO),
        initial_embeddings=final_text_embeddings,
        latents=torch.empty(0),
        k_values=K_VALUES_VIDEO,
    )

    req_queue = mp.Queue()
    new_cache_queue = mp.Queue()
    latency_queue = mp.Queue()
    manager = mp.Manager()
    worker_status = manager.dict()

    device = "cuda:0"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    start_time = time.time()
    scheduler = mp.Process(
        target=request_scheduler_video,
        args=(
            req_queue,
            selected_requests,
            start_time,
            index,
            cache,
            new_cache_queue,
            cached_requests,
            final_text_embeddings,
            K_VALUES_VIDEO,
            processor,
            clip_model,
            worker_status,
        ),
        kwargs={"log_file": "request_throughput_video_teacache.csv"},
    )
    scheduler.start()

    workers = []
    for gpu_id in range(num_gpus):
        worker_status[gpu_id] = "starting"
        p = mp.Process(
            target=worker_video,
            args=(
                gpu_id,
                req_queue,
                new_cache_queue,
                latency_queue,
                worker_status,
                args.video_directory,
                args.resolution,
                args.aspect_ratio,
                args.num_frames,
                args.teacache_thresh,
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
    scheduler.join()

    all_latencies = []
    while not latency_queue.empty():
        all_latencies.append(latency_queue.get())
    if all_latencies:
        print(
            f"Latencies: min={min(all_latencies):.2f}s max={max(all_latencies):.2f}s avg={np.mean(all_latencies):.2f}s"
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
