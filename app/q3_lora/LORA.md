# Q3 — LoRA Integration for Personalized AI Images

> How would you integrate pre-trained LoRA models into our inference pipeline
> for consistent branding/faces inside KeaBuilder?

## Storage layer (per-tenant LoRA registry)

Each KeaBuilder tenant can upload one or more LoRAs (a face LoRA, a brand-style
LoRA, etc.). We store:

- **Weights**: `s3://varynt-loras/{tenant_id}/{lora_name}-v{n}.safetensors`
  versioned by upload — trains can be retried without breaking live funnels.
- **Metadata**: Postgres `tenant_loras` table — `{id, tenant_id, lora_name,
  s3_key, trigger_word, default_scale, base_model, status, created_at}`.
- **Validation**: incoming LoRAs are sanity-checked (file size, header,
  rank ≤ 64, no NSFW signatures) before being marked `status='ready'`.

## Inference flow

```
                  per-tenant request
                          v
        +----------------------------+
        |  build prompt with trigger |   <-- "agent_jane, hero banner..."
        +----------------------------+
                          v
        +----------------------------+
        |  pipe.load_lora_weights()  |   <-- hot-load from S3 / local cache
        +----------------------------+
                          v
        +----------------------------+
        |  pipe.set_adapters()       |   <-- scale 0.7-0.9
        +----------------------------+
                          v
        +----------------------------+
        |  pipe(...)  -> image       |   <-- 25-40 inference steps
        +----------------------------+
                          v
        +----------------------------+
        |  pipe.unload_lora_weights()|   <-- free GPU before next tenant
        +----------------------------+
```

The hot-load + unload pattern is critical for multi-tenant serving — it lets
one long-lived worker process hundreds of tenants without unbounded GPU
memory growth. Diffusers caches the safetensors on disk after the first
load, so warm-cache calls skip the S3 read.

See `app/q3_lora/snippet.py` for the runnable reference (requires GPU + diffusers).

## Trigger words and scale

- **Trigger word**: every LoRA is trained with a unique token (`agent_jane`,
  `coach_mark`). The word MUST appear in the prompt or the LoRA is inactive.
- **Scale**: 0.7–0.9 is the safe default. 1.0 over-fits to training data;
  <0.5 barely activates the adapter. We store a per-LoRA `default_scale`
  measured at training time.

## Multi-LoRA composition (face + brand style)

Funnel images often need *both* the agent's face AND the brand's visual
style. Diffusers `set_adapters([...], adapter_weights=[...])` composes
weighted adapters in a single forward pass. We bake both trigger words
into the prompt:

```
"agent_jane, blue_brand, modern real-estate hero banner, ..."
```

Composition risks: face fidelity drops as more adapters compose. Cap at
2 simultaneous LoRAs in production; the 3rd starts degrading.

## Where this slots into the rest of the system

- **Q2 router**: today the image provider chain holds mocks
  (`MockOpenAIImageProvider`, `MockStabilityImageProvider`). In production
  we add `DiffusersLoRAProvider` as the primary for tenants with an active
  LoRA, falling back to DALL-E for tenants without one. The router's
  fallback chain (Q5) covers GPU pool exhaustion the same way it covers
  any other transient failure.
- **Asset registry**: every output is tagged with `lora_id` in the assets
  table so Q4 similarity search can scope queries to "images of this same
  agent" if needed.

## Training (not in scope here, but for completeness)

- Dataset: 15–30 face images, varied angles/lighting, captioned with the
  trigger word as a unique token.
- Base model: SDXL 1.0 (or Flux for higher fidelity).
- Trainer: `diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py`
  (or `kohya_ss` if a UI is preferred).
- Hyperparameters: rank 16–32, batch 1, LR 1e-4, 1500–2500 steps,
  validation every 250 steps. Stop early when face fidelity peaks
  before the model starts overfitting hands/clothes.

## Failure modes we've planned for

| Failure | Detection | Mitigation |
|---|---|---|
| LoRA weights corrupt / wrong base model | Sanity-check at upload, status=`failed` | Reject upload; user re-trains |
| Prompt missing trigger word | Linter on prompt-build step | Auto-prepend trigger; warn user |
| Output drifts off-identity | Periodic eval against ID-locked test set | Lower scale or retrain |
| Multi-LoRA conflict (color clash) | Per-tenant golden-set diff | Cap composition at 2; flag conflicts |
