import torch
from torchvision.transforms import ToTensor, v2
from diffusers.pipelines import FluxPipeline
from PIL import Image
import comfy.model_management
import os
import folder_paths

from .src.condition import Condition
from .src.generate import generate, seed_everything

omini_path = os.path.join(folder_paths.models_dir, "diffusers", "OminiControl")
supported_condition = ["depth", "canny", "subject", "coloring", "deblurring", "fill"]

def guess_condition_from_name(model_name):
    for cond in supported_condition:
        if cond in model_name:
            return cond
    raise(Exception(f"Model name contains unknown condition. Supported conditions are: {supported_condition}"))


class OminiControlSampler:
    @classmethod
    def INPUT_TYPES(s):
        omini_checkpoints = [
            os.path.relpath(os.path.join(root, file), omini_path)
            for root, _, files in os.walk(omini_path)
            for file in files if file.endswith(".safetensors")
        ]
        return {
            "required": {
                "image": ("IMAGE", ),
                "flux_model": (["FLUX.1-dev", "FLUX.1-schnell"], ),
                "omini_control": (omini_checkpoints, ),
                "width": ("INT", {"default": 512, "min": 128}),
                "height": ("INT", {"default": 512, "min": 128}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "seed": ("INT", {"default": 1337,"min": 0, "max": 2**32 - 1, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "condition_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "inference"
    CATEGORY = "Flux/OmniControl"

    def inference(self, image, flux_model, omini_control, width, height, batch_size, seed, steps, condition_scale, prompt):
        condition = guess_condition_from_name(omini_control)

        # load model + condition lora
        comfy.model_management.unload_all_models()
        device = comfy.model_management.get_torch_device()
        pipe = FluxPipeline.from_pretrained(
            f"./models/diffusers/{flux_model}",
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(device)
        pipe.load_lora_weights(
            "./models/diffusers/OminiControl",
            weight_name=omini_control,
            adapter_name=condition,
            local_files_only=True
        )

        # while original authors recommend using 512x512
        # you can actually scale it to output image size
        # it shows ok results with higher condition_scale
        torch_img = image.squeeze(0).permute(2, 0, 1)
        torch_img = v2.Resize(size=(width, height))(torch_img)
        cond = Condition(condition, condition=torch_img)

        seed_everything(seed)
        decoder_output  = generate(
            pipe,
            conditions=[cond],
            condition_scale=condition_scale,
            num_inference_steps=steps,
            height=width,
            width=height,
            prompt=prompt,
            num_images_per_prompt=batch_size
        ).images

        tensors = [ToTensor()(img) for img in decoder_output]
        batch_tensor = torch.stack(tensors).permute(0, 2, 3, 1).cpu()
        return (batch_tensor,)