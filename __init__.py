from .omini_control_node import OminiControlSampler

NODE_CLASS_MAPPINGS = {
    "OminiControlSampler": OminiControlSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiControlSampler": "Omini Control Sampler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']