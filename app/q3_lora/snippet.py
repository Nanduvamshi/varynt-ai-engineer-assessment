"""Q3 — LoRA integration reference (non-executing).

This file documents *exactly* how a per-tenant LoRA gets loaded into the
inference pipeline at request time. It's a reference implementation; the
real call lives behind the Q2 image provider chain (`MockOpenAIImageProvider`
would be replaced by `DiffusersImageProvider` here).

Run it only on a GPU host with diffusers + a base model checked out. Not
wired into the FastAPI app on purpose.
"""

from __future__ import annotations

import torch  # type: ignore[import-not-found]
from diffusers import StableDiffusionXLPipeline  # type: ignore[import-not-found]

# In production these come from a per-tenant config table.
TENANT_LORA_REGISTRY = {
    "acme-realestate": {
        "lora_path": "s3://varynt-loras/acme/agent-jane-v3.safetensors",
        "trigger": "agent_jane",
        "scale": 0.85,
    },
    "blue-fitness": {
        "lora_path": "s3://varynt-loras/blue/coach-mark-v1.safetensors",
        "trigger": "coach_mark",
        "scale": 0.8,
    },
}


def build_pipeline(base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    """One-time, process-level. Heavy — keep in a long-lived worker."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    return pipe


def generate_with_tenant_lora(pipe, tenant_id: str, prompt: str, num_images: int = 1):
    """Per-request: hot-load the tenant's LoRA, run inference, unload.

    The hot-load + unload pattern keeps a single worker process serving many
    tenants without growing GPU memory unboundedly. Diffusers caches LoRA
    weights, so subsequent calls for the same tenant skip the disk read.
    """
    cfg = TENANT_LORA_REGISTRY[tenant_id]

    pipe.load_lora_weights(
        cfg["lora_path"],
        adapter_name=tenant_id,
    )
    pipe.set_adapters([tenant_id], adapter_weights=[cfg["scale"]])

    # Trigger word MUST appear in the prompt for the LoRA to activate.
    prompt_with_trigger = f"{cfg['trigger']}, {prompt}"

    images = pipe(
        prompt=prompt_with_trigger,
        num_images_per_prompt=num_images,
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images

    pipe.unload_lora_weights()  # free GPU memory before next tenant
    return images


def generate_with_multi_lora(pipe, lora_specs: list[dict], prompt: str):
    """Compose multiple LoRAs (e.g., a face LoRA + a brand-style LoRA).

    Diffusers supports weighted multi-adapter composition. Useful when the
    funnel needs the agent's face AND the brand's visual style baked in.
    """
    triggers = []
    weights = []
    names = []
    for spec in lora_specs:
        pipe.load_lora_weights(spec["lora_path"], adapter_name=spec["name"])
        names.append(spec["name"])
        weights.append(spec["scale"])
        triggers.append(spec["trigger"])

    pipe.set_adapters(names, adapter_weights=weights)
    full_prompt = ", ".join(triggers) + ", " + prompt
    images = pipe(prompt=full_prompt, num_inference_steps=30).images
    pipe.unload_lora_weights()
    return images
