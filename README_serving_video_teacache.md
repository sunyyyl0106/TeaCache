# TeaCache Video Serving (Nirvana-Style Cross-Request Latent Reuse)

This document describes how to run **TeaCache video pipeline + Nirvana-style cross-request latent cache**: CLIP similarity + FAISS to find similar prompts, reuse or write video latents at steps 5/10/15, with KMinHeapCache eviction and multi-GPU workers.

---

## Environment and Dependencies

- **Python**: 3.8+ recommended
- **CUDA**: Match your PyTorch version
- Install project dependencies (see `requirements.txt`) and additionally:
  - `faiss-cpu` or `faiss-gpu` (vector search)
  - `pandas` (for request scheduling)

```bash
# If not already installed
pip install faiss-cpu pandas
# Or FAISS with GPU
pip install faiss-gpu pandas
```

---

## How to Run

**Run from the project root** so that `videosys`, `eval.teacache`, `serving_system_N`, etc. are importable:

```bash
cd /path/to/TeaCache
python serving_system_video_teacache.py [options]
```

---

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_req` | int | 50 | Total number of requests. |
| `--cache_size` | int | 1000 | Max number of cached *requests* (each stores 3 steps 5/10/15; total cache entries = cache_size × 3). |
| `--video_directory` | str | `./video_outputs_teacache` | Directory to save generated videos. |
| `--prompt_list` | str | None | Path to a JSON file with prompts; if not set, 3 built-in prompts are cycled. |
| `--resolution` | str | 480p | Output resolution. |
| `--aspect_ratio` | str | 9:16 | Aspect ratio (e.g. 16:9, 1:1). |
| `--num_frames` | int | 51 | Number of video frames. |
| `--teacache_thresh` | float | 0.15 | TeaCache `rel_l1_thresh`: higher values skip more blocks (faster, slightly lower quality). |

---

## Run Examples

### 1. Quick run (default prompts, 50 requests)

```bash
python serving_system_video_teacache.py --num_req 50 --video_directory ./video_outputs_teacache
```

- Uses 3 built-in prompts in a loop for 50 requests.
- Videos are saved under `./video_outputs_teacache/`.

### 2. VBench prompt list (20 requests, 500 cache entries)

```bash
python serving_system_video_teacache.py \
  --num_req 20 \
  --prompt_list eval/teacache/vbench/VBench_full_info.json \
  --cache_size 500 \
  --video_directory ./vbench_teacache_out
```

- Prompts are read from `VBench_full_info.json` (prefers `prompt_en` per entry).
- Runs the first 20 prompts; cache holds up to 500 requests’ latents.

### 3. Small cache, higher TeaCache skip (faster)

```bash
python serving_system_video_teacache.py \
  --num_req 10 \
  --cache_size 200 \
  --teacache_thresh 0.2 \
  --video_directory ./fast_out
```

- Fewer requests and smaller cache; good for debugging or latency tests.
- `teacache_thresh=0.2` skips more Transformer blocks and speeds up each run.

### 4. Custom resolution and frame count

```bash
python serving_system_video_teacache.py \
  --num_req 30 \
  --resolution 360p \
  --aspect_ratio 16:9 \
  --num_frames 33 \
  --video_directory ./custom_out
```

- All requests use the same resolution, aspect ratio, and frame count (consistent with the cache policy).

---

## Behavior Overview

- **Scheduler process**: Dispatches requests by timestamp; encodes each prompt with CLIP and uses FAISS to find the nearest cached prompt; if similarity > 0.65, selects a step tier (0.95→step 15, 0.85→step 10, 0.65→step 5), retrieves the corresponding latent from KMinHeapCache and sends it with the request to a worker; otherwise marks as cache miss.
- **Worker process** (one per GPU): Loads Open-Sora + TeaCache (`teacache_forward`).  
  - **Miss**: Runs full 30-step generation, collects latents at steps 5/10/15, and pushes them with the CLIP embedding to `new_cache_queue`.  
  - **Hit**: Uses `cache_latent` and `cache_start_step` (5/10/15) from the request to continue denoising from that step to the end.
- **Cache**: KMinHeapCache with LCBFU eviction; k_values are fixed at `[5, 10, 15]`.

---

## Output and Logs

- **Videos**: Saved under `--video_directory`; filenames include prompt and timestamp.
- **Throughput log**: Written to `request_throughput_video_teacache.csv` in the current directory (timestamp, request_rate, throughput).
- **Console**: Prints latency stats (min/max/avg in seconds) when finished.

---

## FAQ

1. **ModuleNotFoundError: No module named 'eval'**  
   Run the script from the **project root** so the `eval` package is on the Python path.

2. **CUDA out of memory**  
   Reduce `--num_frames` or `--resolution`, or use fewer workers (currently one per GPU).

3. **FAISS-related errors**  
   Ensure `faiss-cpu` or `faiss-gpu` is installed and compatible with your Python/CUDA.

4. **Changing k_values**  
   The implementation uses [5, 10, 15] by default. To use other steps, edit `K_VALUES_VIDEO` in `serving_system_video_teacache.py` and the corresponding `collect_latents_at_steps` / `cache_start_step` in the pipeline and scheduler.
