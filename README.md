# ComfyUI Diffusers OminiControl

This is an unofficial [ComfyUI](https://github.com/comfyanonymous/ComfyUI) node for [FLUX OminiControl](https://github.com/Yuanshi9815/OminiControl). 

Because it uses the [diffusers](https://huggingface.co/docs/diffusers/en/index) library, it doesn't work with the built-in ComfyUI optimization. For unquantized FLUX models, you would need **30+ GB VRAM**. I hope the comfy team will add OminiControl native support soon.

## Instalation

1. Install the latest version of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. In ComfuUI Manager install this repository by `"Install via Git URL"`
3. Download [OminiControl](https://huggingface.co/Yuanshi/OminiControl) and [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) or/and [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) repositories. Place them in `comfyui/models/diffusers`. Use git repositories directly, not models in model manager.
4. Restart comfyui and refresh nodes

## Examples

![omini-workflow](https://github.com/user-attachments/assets/8beabde5-936d-46fb-8615-c3f6b3fd7823)

## Acknowledgment

This project is based on original [OminiControl repository]((https://github.com/Yuanshi9815/OminiControl)).
