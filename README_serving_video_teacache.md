# TeaCache Video Serving (Nirvana-Style Cross-Request Latent Reuse)

Combines **TeaCache** (skip redundant transformer steps within a request) and **Nirvana** (reuse intermediate latents across similar requests) on top of **Wan2.1 T2V**.

---

## Dependencies

```bash
pip install faiss-cpu pandas transformers torch
# or GPU FAISS:
pip install faiss-gpu pandas transformers torch
```

Also requires the `wan` package and a Wan2.1 checkpoint directory.

---

## Quick Start

```bash
python serving_system_video_teacache.py \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --task t2v-14B \
    --size 832*480 \
    --num_req 50
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt_dir` | **(required)** | Wan2.1 checkpoint directory |
| `--task` | `t2v-14B` | Task name: `t2v-14B` or `t2v-1.3B` |
| `--size` | `832*480` | Resolution `W*H` (e.g. `1280*720`, `832*480`, `480*832`) |
| `--frame_num` | `81` | Frames per video (must be `4n+1`) |
| `--sampling_steps` | `50` | Diffusion denoising steps |
| `--guide_scale` | `5.0` | Classifier-free guidance scale |
| `--shift` | `5.0` | Flow-matching noise schedule shift |
| `--use_ret_steps` | False | Use retention steps (better quality, slight speed trade-off) |
| `--offload_model` | True | Offload weights to CPU between steps to save VRAM |
| `--teacache_thresh` | `0.2` | Skip threshold (`0.1` ≈ 2× speedup, `0.2` ≈ 3×) |
| `--num_req` | None | Max prompts (default: all from `--prompt_list`, else 50) |
| `--loop` | `5` | Videos per prompt; only the first uses Nirvana cache |
| `--cache_size` | `1000` | Max cached requests (stores latents at steps 5/10/15; total entries = `cache_size × 3`) |
| `--video_directory` | `./video_outputs_wan21` | Output directory |
| `--prompt_list` | None | JSON file with `prompt_en` field per item (VBench format) |
| `--eval_mode` | False | Submit all prompts immediately, no request-rate simulation |
| `--no_nirvana` | False | Disable Nirvana cache (A/B baseline) |
| `--log_file` | `request_throughput_wan21_nirvana.csv` | Per-minute throughput CSV log |
| `--num_workers` | None | Worker processes (default: one per GPU) |

---

## Examples

### 1. Minimal run (built-in prompts, 14B model)

```bash
python serving_system_video_teacache.py \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --task t2v-14B \
    --num_req 20
```

### 2. VBench prompt list, eval mode, 480p

```bash
python serving_system_video_teacache.py \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --task t2v-14B \
    --size 832*480 \
    --prompt_list eval/teacache/vbench/VBench_full_info.json \
    --video_directory ./samples/wan21_nirvana \
    --eval_mode
```

### 3. 1.3B model, 720p, retention steps

```bash
python serving_system_video_teacache.py \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --task t2v-1.3B \
    --size 1280*720 \
    --teacache_thresh 0.1 \
    --use_ret_steps \
    --num_req 50 \
    --video_directory ./samples/wan21_1.3b_nirvana
```

### 4. A/B comparison: Nirvana on vs. off

```bash
# With Nirvana
python serving_system_video_teacache.py \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --task t2v-14B \
    --prompt_list eval/teacache/vbench/VBench_full_info.json \
    --num_req 20 --eval_mode \
    --video_directory ./out_with_nirvana

# Without Nirvana
python serving_system_video_teacache.py \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --task t2v-14B \
    --prompt_list eval/teacache/vbench/VBench_full_info.json \
    --num_req 20 --eval_mode --no_nirvana \
    --video_directory ./out_no_nirvana
```

Compare **Total wall time** and **Per-request latency (avg)** printed at the end.

---

## How It Works

**Scheduler process** — encodes each prompt with CLIP, queries FAISS for the nearest cached prompt. If cosine similarity > 0.65, selects a latent step tier and forwards the cached latent to a worker:

| Similarity | Reuse from step |
|---|---|
| > 0.95 | 15 (skips 15/50 steps) |
| > 0.85 | 10 |
| > 0.65 | 5 |

**Worker process** (one per GPU) — loads `wan.WanT2V` with:
- `teacache_forward` — instance-level TeaCache (even/odd cond/uncond paths)
- `t2v_generate_nirvana` — Nirvana-aware generate function

On a **cache miss**: runs full denoising, snapshots latents at steps 5/10/15, enqueues them for the scheduler.  
On a **cache hit**: injects the cached latent and resumes denoising from step 5, 10, or 15.

**Cache** — `KMinHeapCache` with LCBFU eviction; stays aligned with FAISS via `remap_index_after_eviction`.

---

## Output

- **Videos**: `{video_directory}/{prompt}-{loop_idx}.mp4`
- **Throughput log**: `--log_file` CSV (`timestamp, request_rate, throughput`)
- **Console summary**:
  ```
  [Total wall time] 342.10s
  [Per-request latency]  min=12.3s  max=89.4s  avg=34.2s  (n=50)
  [Pure processing time] min=11.8s  max=88.1s  avg=33.7s
  [Cache hit rate] 31/50 = 62.0%
  ```

---

## FAQ

**ModuleNotFoundError: No module named 'wan'**  
Run from the project root or install: `pip install -e .`

**CUDA out of memory**  
Use `--offload_model` (on by default), reduce `--frame_num`, lower `--size`, or use the 1.3B model.

**Multi-GPU**  
One worker per GPU by default. Use `--num_workers N` to limit.
