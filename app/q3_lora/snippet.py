from __future__ import annotations
import torch
from diffusers import StableDiffusionXLPipeline

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
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    return pipe


def generate_with_tenant_lora(pipe, tenant_id: str, prompt: str, num_images: int = 1):
    cfg = TENANT_LORA_REGISTRY[tenant_id]
    pipe.load_lora_weights(cfg["lora_path"], adapter_name=tenant_id)
    pipe.set_adapters([tenant_id], adapter_weights=[cfg["scale"]])
    prompt_with_trigger = f"{cfg['trigger']}, {prompt}"
    images = pipe(
        prompt=prompt_with_trigger,
        num_images_per_prompt=num_images,
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images
    pipe.unload_lora_weights()
    return images


def generate_with_multi_lora(pipe, lora_specs: list[dict], prompt: str):
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
