"""
Nirvana-style cross-request latent cache for TeaCache video (Wan2.1 T2V).
Uses k_values = [5, 10, 15]; CLIP for prompt similarity; FAISS + KMinHeapCache.

Usage:
  python serving_system_video_teacache.py \
      --ckpt_dir ./Wan2.1-T2V-14B \
      --task t2v-14B \
      --size 832*480 \
      --num_req 100 \
      --teacache_thresh 0.2
"""
import gc
import importlib.util
import json
import math
import os
import queue
import sys
import time
import argparse
from contextlib import contextmanager

import faiss
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from serving_system_N import evict_from_faiss, KMinHeapCache

# ── generation defaults ───────────────────────────────────────────────────────
DEFAULT_TASK           = "t2v-14B"
DEFAULT_SIZE           = "832*480"
DEFAULT_FRAME_NUM      = 81          # must be 4n+1
DEFAULT_SAMPLING_STEPS = 50
DEFAULT_GUIDE_SCALE    = 5.0
DEFAULT_SHIFT          = 5.0
K_VALUES_VIDEO         = [5, 10, 15]

# TeaCache polynomial coefficients (from TeaCache4Wan2.1/teacache_generate.py)
_COEFFS = {
    "1.3B": {
        True:  [-5.21862437e+04,  9.23041404e+03, -5.28275948e+02,  1.36987616e+01, -4.99875664e-02],
        False: [ 2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00,  1.37887774e-01],
    },
    "14B": {
        True:  [-3.03318725e+05,  4.90537029e+04, -2.65530556e+03,  5.87365115e+01, -3.15583525e-01],
        False: [-5784.54975374,   5449.50911966,  -1811.16591783,    256.27178429,   -13.02252404],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_teacache_forward():
    """Load teacache_forward from TeaCache4Wan2.1/ via importlib.
    The directory name contains '.' so normal package import fails."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "TeaCache4Wan2.1",
        "teacache_generate.py",
    )
    spec = importlib.util.spec_from_file_location("teacache_wan21", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["teacache_wan21"] = mod
    spec.loader.exec_module(mod)
    return mod.teacache_forward


def read_prompt_list(path):
    with open(path) as f:
        data = json.load(f)
    return [item["prompt_en"] for item in data]


def _drain_new_cache_queue(
    new_cache_queue, cache, cached_requests, index, final_text_embeddings, k_values
):
    """Drain pending latent-cache entries from workers into the cache.

    Normalizes embeddings for IndexFlatIP, evicts if needed, and remaps
    KMinHeapCache indices to stay aligned with the FAISS index after eviction.
    """
    while not new_cache_queue.empty():
        cache_data         = new_cache_queue.get()
        new_cached_latents = [z.clone() for z in cache_data["cached_latents"]]
        new_cached_prompt  = cache_data["prompt"]
        new_query_emb      = cache_data["query_embedding"]

        while len(cache.item_map) + len(cache.k_values) > cache.max_size:
            evicted_index = cache.evict()
            if evicted_index is not None:
                del cached_requests[evicted_index]
                index, final_text_embeddings = evict_from_faiss(
                    index, final_text_embeddings, evicted_index, use_ip=True
                )
                cache.remap_index_after_eviction(evicted_index)

        # Normalize before inserting into IndexFlatIP so inner-product == cosine.
        normalized_emb = new_query_emb.copy()
        faiss.normalize_L2(normalized_emb)

        num_embeddings = index.ntotal
        cached_requests.append(new_cached_prompt)
        index.add(normalized_emb)
        final_text_embeddings = np.concatenate(
            (final_text_embeddings, normalized_emb), axis=0
        )
        for _idx, k in enumerate(k_values):
            cache.insert(num_embeddings, 0, k, new_cached_latents[_idx])

    return index, final_text_embeddings


# ── Wan2.1 generate with Nirvana latent-cache support ────────────────────────

def t2v_generate_nirvana(
    self,
    input_prompt,
    size=(1280, 720),
    frame_num=81,
    shift=5.0,
    sample_solver="unipc",
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
    collect_latents_at_steps=None,
    cache_latent=None,
    cache_start_step=None,
):
    """Wan2.1 T2V generate patched for Nirvana cross-request caching.

    collect_latents_at_steps:
        Tuple of 1-based step numbers at which to snapshot the intermediate
        latent (e.g. (5, 10, 15)).  Used on cache-miss to populate the cache.
    cache_latent / cache_start_step:
        On a Nirvana hit, inject the cached latent and skip the first
        cache_start_step denoising steps.
    """
    import random as _random
    F = frame_num
    target_shape = (
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )
    seq_len = (
        math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        )
        * self.sp_size
    )

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else _random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context      = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt],     self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context      = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt],     torch.device("cpu"))
        context      = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            *target_shape, dtype=torch.float32,
            device=self.device, generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False,
            )
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")

        # Nirvana: inject cached latent and trim already-completed steps.
        if cache_latent is not None and cache_start_step is not None:
            latents      = [cache_latent.to(self.device, dtype=torch.float32)]
            timesteps    = timesteps[cache_start_step:]
            start_offset = cache_start_step
        else:
            latents      = noise
            start_offset = 0

        collected = {}
        arg_c    = {"context": context,      "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        self.model.to(self.device)
        for step_idx, t in enumerate(tqdm(timesteps, leave=False)):
            ts = torch.stack([t])
            noise_pred_cond   = self.model(latents, t=ts, **arg_c)[0]
            noise_pred_uncond = self.model(latents, t=ts, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

            # Snapshot at 1-based global step number for Nirvana cache.
            global_step = step_idx + start_offset + 1
            if collect_latents_at_steps and global_step in collect_latents_at_steps:
                collected[global_step] = latents[0].clone().cpu()

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()

    video = videos[0] if self.rank == 0 else None
    if collect_latents_at_steps:
        return video, [collected[k] for k in collect_latents_at_steps if k in collected]
    return video


# ── scheduler ─────────────────────────────────────────────────────────────────

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
    clip_model_name,
    clip_device,
    worker_status,
    done_event,
    log_file="request_throughput_video_teacache.csv",
    eval_mode=False,
    no_nirvana=False,
    cache_stats=None,
):
    # Load CLIP inside the subprocess so it owns the CUDA tensors and can
    # release them cleanly on exit (avoids CudaIPCTypes producer warning).
    processor  = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(clip_device)
    device = clip_device

    agg_k_distribution = {k: 0 for k in k_values}
    if cache_stats is None:
        cache_stats = {"hits": 0, "misses": 0}
    minute = 0
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("timestamp,request_rate,throughput\n")
    last_check_time_queue = time.time()
    request_count_per_min = 0
    size_of_queues = 0

    for _, row in selected_requests.iterrows():
        if not eval_mode:
            while time.time() - start_time < row["seconds_from_start"]:
                time.sleep(0.1)

        row["start_time"] = time.time()

        index, final_text_embeddings = _drain_new_cache_queue(
            new_cache_queue, cache, cached_requests, index, final_text_embeddings, k_values
        )

        prompt = row["prompt"]
        if no_nirvana:
            with torch.no_grad():
                texts = processor(
                    text=[prompt], return_tensors="pt",
                    truncation=True, padding=True, max_length=77,
                ).to(device)
                text_embedding = clip_model.get_text_features(**texts).cpu()
            row["cached"]          = None
            row["k"]               = None
            row["latent"]          = None
            row["query_embedding"] = text_embedding.clone()
            cache_stats["misses"]  = cache_stats.get("misses", 0) + 1
            req_queue.put(row.to_dict())
            request_count_per_min += 1
            continue

        texts = processor(
            text=[prompt], return_tensors="pt",
            truncation=True, padding=True, max_length=77,
        ).to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**texts).cpu()

        # Normalize for IndexFlatIP: inner-product on unit vectors == cosine.
        query_embedding = text_embedding.numpy().reshape(1, -1).copy()
        faiss.normalize_L2(query_embedding)

        if index.ntotal == 0:
            row["cached"]          = None
            row["k"]               = None
            row["latent"]          = None
            row["query_embedding"] = text_embedding.clone()
            cache_stats["misses"]  = cache_stats.get("misses", 0) + 1
            req_queue.put(row.to_dict())
        else:
            distances, indices = index.search(query_embedding, k=1)
            sim = float(distances[0][0])

            if sim > 0.65:
                if   sim > 0.95: closest_index = 15
                elif sim > 0.85: closest_index = 10
                elif sim > 0.75: closest_index = 10
                else:            closest_index = 5

                best_candidate = cache.retrieve(closest_index, indices[0][0])
                if best_candidate:
                    _score, (_idx, k_i, latent) = best_candidate
                    row["cached"]          = True
                    row["k"]               = k_i
                    row["latent"]          = latent.clone().to(dtype=torch.float32).cpu()
                    row["query_embedding"] = text_embedding.clone()
                    cache_stats["hits"]    = cache_stats.get("hits", 0) + 1
                    req_queue.put(row.to_dict())
                    agg_k_distribution[k_i] += 1
                else:
                    row["cached"]          = None
                    row["k"]               = None
                    row["latent"]          = None
                    row["query_embedding"] = text_embedding.clone()
                    cache_stats["misses"]  = cache_stats.get("misses", 0) + 1
                    req_queue.put(row.to_dict())
            else:
                row["cached"]          = None
                row["k"]               = None
                row["latent"]          = None
                row["query_embedding"] = text_embedding.clone()
                cache_stats["misses"]  = cache_stats.get("misses", 0) + 1
                req_queue.put(row.to_dict())

        request_count_per_min += 1
        current_time = time.time()
        if current_time - last_check_time_queue >= 60:
            elapsed_time = current_time - last_check_time_queue
            minute += 1
            new_size = req_queue.qsize()
            throughput = size_of_queues + request_count_per_min - new_size
            size_of_queues = new_size
            with open(log_file, "a") as f:
                f.write(
                    f"{minute},"
                    f"{request_count_per_min / elapsed_time * 60},"
                    f"{throughput / elapsed_time * 60}\n"
                )
            request_count_per_min = 0
            last_check_time_queue = current_time

    while not req_queue.empty():
        index, final_text_embeddings = _drain_new_cache_queue(
            new_cache_queue, cache, cached_requests, index, final_text_embeddings, k_values
        )
        time.sleep(0.05)

    print(f"[Scheduler] k distribution: {agg_k_distribution}")
    done_event.set()

    while True:
        if req_queue.empty():
            if all(s in ("finished", "dropped") for s in worker_status.values()):
                break
        time.sleep(1)


# ── worker ────────────────────────────────────────────────────────────────────

def worker_video(
    gpu_id,
    req_queue,
    new_cache_queue,
    latency_queue,
    worker_status,
    video_directory,
    ckpt_dir,
    task,
    size,
    frame_num,
    sampling_steps,
    guide_scale,
    shift,
    teacache_thresh,
    done_event,
    loop=1,
    no_nirvana=False,
    offload_model=True,
    use_ret_steps=False,
):
    teacache_forward = _load_teacache_forward()

    cfg   = WAN_CONFIGS[task]
    try:
        model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=gpu_id,
            rank=0,
            t5_cpu=True,
            offload_model=offload_model,
        )
    except TypeError as e:
        if "offload_model" not in str(e):
            raise
        model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=gpu_id,
            rank=0,
            t5_cpu=True,
        )

    # TeaCache patch (instance-level to avoid cross-worker state pollution).
    m        = model.model
    size_key = "1.3B" if "1.3B" in task else "14B"
    coeffs   = _COEFFS[size_key][use_ret_steps]

    m.enable_teacache                  = True
    m.cnt                              = 0
    m.num_steps                        = sampling_steps * 2   # cond + uncond per step
    m.teacache_thresh                  = teacache_thresh
    m.accumulated_rel_l1_distance_even = 0
    m.accumulated_rel_l1_distance_odd  = 0
    m.previous_e0_even                 = None
    m.previous_e0_odd                  = None
    m.previous_residual_even           = None
    m.previous_residual_odd            = None
    m.use_ref_steps                    = use_ret_steps
    m.coefficients                     = coeffs
    m.ret_steps                        = (5 if use_ret_steps else 1) * 2
    m.cutoff_steps                     = sampling_steps * 2 - (0 if use_ret_steps else 2)
    m.__class__.forward                = teacache_forward

    # Replace generate with the Nirvana-aware version.
    model.__class__.generate = t2v_generate_nirvana

    video_size   = SIZE_CONFIGS[size]
    idle_counter = 0

    while True:
        try:
            request       = req_queue.get(timeout=10)
            process_start = time.time()
            idle_counter  = 0
            prompt        = request["prompt"]

            for loop_idx in range(loop):
                out_path = os.path.join(video_directory, f"{prompt}-{loop_idx}.mp4")

                if request["cached"] is None:
                    # Cache miss: full generation, snapshot latents on first loop.
                    collect_steps = tuple(K_VALUES_VIDEO) if loop_idx == 0 else None
                    result = model.generate(
                        prompt,
                        size=video_size,
                        frame_num=frame_num,
                        shift=shift,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=loop_idx,
                        offload_model=offload_model,
                        collect_latents_at_steps=collect_steps,
                    )
                    if isinstance(result, tuple):
                        video, collected_latents = result
                    else:
                        video, collected_latents = result, None

                    if video is not None:
                        cache_video(
                            tensor=video[None], save_file=out_path,
                            fps=cfg.sample_fps, nrow=1,
                            normalize=True, value_range=(-1, 1),
                        )
                    if (not no_nirvana
                            and collected_latents
                            and request.get("query_embedding") is not None):
                        qe    = request["query_embedding"]
                        qe_np = (qe.numpy().reshape(1, -1)
                                 if hasattr(qe, "numpy")
                                 else np.array(qe).reshape(1, -1))
                        new_cache_queue.put({
                            "cached_latents": [z.cpu().clone() for z in collected_latents],
                            "prompt":          prompt,
                            "query_embedding": qe_np,
                        })

                else:
                    # Cache hit: resume from cached latent.
                    cache_latent = request["latent"]
                    if isinstance(cache_latent, torch.Tensor):
                        cache_latent = cache_latent.to(dtype=torch.float32)
                    k = request["k"]
                    result = model.generate(
                        prompt,
                        size=video_size,
                        frame_num=frame_num,
                        shift=shift,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=loop_idx,
                        offload_model=offload_model,
                        cache_latent=cache_latent,
                        cache_start_step=k,
                    )
                    video = result[0] if isinstance(result, tuple) else result
                    if video is not None:
                        cache_video(
                            tensor=video[None], save_file=out_path,
                            fps=cfg.sample_fps, nrow=1,
                            normalize=True, value_range=(-1, 1),
                        )

            latency_queue.put((
                time.time() - request["start_time"],
                time.time() - process_start,
            ))

        except queue.Empty:
            if done_event.is_set():
                worker_status[gpu_id] = "finished"
                break
            idle_counter += 1
            if idle_counter >= 100:
                worker_status[gpu_id] = "dropped"
                break
            continue


# ── request timing ────────────────────────────────────────────────────────────

def generate_rapidly_increasing_seconds_from_start(
    num_requests, min_rate=2, max_rate=9, duration=100 * 60
):
    min_rate_per_sec = min_rate / 60
    max_rate_per_sec = max_rate / 60
    x = np.linspace(-2, 6, num_requests)
    sigmoid_growth  = 1 / (1 + np.exp(-1.5 * x))
    request_rates   = min_rate_per_sec + (max_rate_per_sec - min_rate_per_sec) * sigmoid_growth
    interarrival    = 1 / np.maximum(request_rates, 1e-3)
    return np.cumsum(interarrival)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TeaCache + Nirvana serving with Wan2.1 T2V"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Wan2.1 checkpoint directory (e.g. ./Wan2.1-T2V-14B)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        choices=list(WAN_CONFIGS.keys()),
                        help=f"Wan2.1 task (default: {DEFAULT_TASK})")
    parser.add_argument("--size", type=str, default=DEFAULT_SIZE,
                        choices=list(SIZE_CONFIGS.keys()),
                        help=f"Output resolution WxH (default: {DEFAULT_SIZE})")
    parser.add_argument("--frame_num", type=int, default=DEFAULT_FRAME_NUM,
                        help=f"Frames per video, must be 4n+1 (default: {DEFAULT_FRAME_NUM})")
    parser.add_argument("--sampling_steps", type=int, default=DEFAULT_SAMPLING_STEPS,
                        help=f"Diffusion steps (default: {DEFAULT_SAMPLING_STEPS})")
    parser.add_argument("--guide_scale", type=float, default=DEFAULT_GUIDE_SCALE,
                        help=f"CFG scale (default: {DEFAULT_GUIDE_SCALE})")
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT,
                        help=f"Flow-matching shift (default: {DEFAULT_SHIFT})")
    parser.add_argument("--use_ret_steps", action="store_true",
                        help="Use retention steps (better quality, slightly slower TeaCache)")
    parser.add_argument("--offload_model", action="store_true", default=True,
                        help="Offload model to CPU between steps to save VRAM")
    parser.add_argument("--num_req", type=int, default=None,
                        help="Max prompts (default: all from --prompt_list, else 50)")
    parser.add_argument("--cache_size", type=int, default=1000,
                        help="Nirvana cache size in entries (default: 1000)")
    parser.add_argument("--video_directory", type=str, default="./video_outputs_wan21",
                        help="Output directory for generated videos")
    parser.add_argument("--prompt_list", type=str, default=None,
                        help="JSON file; each item must have a 'prompt_en' field")
    parser.add_argument("--teacache_thresh", type=float, default=0.2,
                        help="TeaCache threshold (0.1≈2x speedup, 0.2≈3x)")
    parser.add_argument("--loop", type=int, default=5,
                        help="Videos per prompt; only the first uses Nirvana cache")
    parser.add_argument("--eval_mode", action="store_true",
                        help="Submit all prompts immediately (no request-rate simulation)")
    parser.add_argument("--no_nirvana", action="store_true",
                        help="Disable Nirvana cache (A/B baseline comparison)")
    parser.add_argument("--log_file", type=str,
                        default="request_throughput_wan21_nirvana.csv",
                        help="Throughput CSV log path")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Worker processes (default: one per GPU)")
    args = parser.parse_args()

    os.makedirs(args.video_directory, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found")
    num_workers = min(args.num_workers or num_gpus, num_gpus)

    if args.prompt_list and os.path.isfile(args.prompt_list):
        prompts = read_prompt_list(args.prompt_list)
        if args.num_req is not None:
            prompts = prompts[: args.num_req]
    else:
        num_req = args.num_req or 50
        prompts = [
            "Two cats playing in the garden.",
            "Ocean waves crashing at sunset.",
            "A dog running through a snowy field.",
        ] * max(1, (num_req + 2) // 3)
        prompts = prompts[:num_req]

    seconds_from_start = generate_rapidly_increasing_seconds_from_start(
        len(prompts), min_rate=0.5, max_rate=4
    )
    selected_requests = pd.DataFrame(
        {"prompt": prompts, "seconds_from_start": seconds_from_start}
    )

    embedding_dim         = 768
    index                 = faiss.IndexFlatIP(embedding_dim)
    final_text_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
    cached_requests       = []
    cache = KMinHeapCache(
        max_size=args.cache_size * len(K_VALUES_VIDEO),
        initial_embeddings=final_text_embeddings,
        latents=torch.empty(0),
        k_values=K_VALUES_VIDEO,
    )

    req_queue       = mp.Queue()
    new_cache_queue = mp.Queue()
    latency_queue   = mp.Queue()
    manager         = mp.Manager()
    worker_status   = manager.dict()
    cache_stats     = manager.dict({"hits": 0, "misses": 0})
    done_event      = manager.Event()

    clip_model_name = "openai/clip-vit-large-patch14"
    clip_device     = "cuda:0" if torch.cuda.is_available() else "cpu"

    wall_start = time.time()
    scheduler  = mp.Process(
        target=request_scheduler_video,
        args=(
            req_queue, selected_requests, wall_start,
            index, cache, new_cache_queue, cached_requests,
            final_text_embeddings, K_VALUES_VIDEO,
            clip_model_name, clip_device,
            worker_status, done_event,
        ),
        kwargs={
            "log_file":    args.log_file,
            "eval_mode":   args.eval_mode,
            "no_nirvana":  args.no_nirvana,
            "cache_stats": cache_stats,
        },
    )
    scheduler.start()

    workers = []
    for gpu_id in range(num_workers):
        worker_status[gpu_id] = "starting"
        p = mp.Process(
            target=worker_video,
            args=(
                gpu_id, req_queue, new_cache_queue, latency_queue,
                worker_status, args.video_directory,
                args.ckpt_dir, args.task, args.size,
                args.frame_num, args.sampling_steps, args.guide_scale, args.shift,
                args.teacache_thresh, done_event,
                args.loop, args.no_nirvana, args.offload_model, args.use_ret_steps,
            ),
        )
        p.start()
        workers.append(p)

    all_latencies        = []
    all_processing_times = []
    with tqdm(total=len(prompts), desc="Requests", unit="req") as pbar:
        for _ in range(len(prompts)):
            finish_time, proc_time = latency_queue.get()
            all_latencies.append(finish_time)
            all_processing_times.append(proc_time)
            pbar.update(1)

    for p in workers:
        p.join()
    scheduler.join()

    wall_total = time.time() - wall_start
    print(f"[Total wall time] {wall_total:.2f}s")
    if all_latencies:
        print(
            f"[Per-request latency]  "
            f"min={min(all_latencies):.2f}s  "
            f"max={max(all_latencies):.2f}s  "
            f"avg={np.mean(all_latencies):.2f}s  "
            f"(n={len(all_latencies)})"
        )
    if all_processing_times:
        print(
            f"[Pure processing time] "
            f"min={min(all_processing_times):.2f}s  "
            f"max={max(all_processing_times):.2f}s  "
            f"avg={np.mean(all_processing_times):.2f}s"
        )
    hits   = cache_stats.get("hits",   0)
    misses = cache_stats.get("misses", 0)
    total  = hits + misses
    if total > 0:
        print(f"[Cache hit rate] {hits}/{total} = {100.0 * hits / total:.1f}%")
    elif args.no_nirvana:
        print("[Cache hit rate] N/A (Nirvana disabled)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
