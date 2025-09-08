import os
import sys
import traceback
import torch
from typing import Optional, List, Tuple, Dict, Any
from .gemini_api import GeminiAPIClient


class NanoBananaGeminiImageNode:
    """
    Generate images and text using Google Gemini API.
    Creates images based on text prompts with optional image context.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt for image generation"
                }),
                "model": ("STRING", {
                    "default": "gemini-2.5-flash-image-preview",
                    "tooltip": "Gemini model (e.g., gemini-2.5-flash-image-preview, gemini-2.5-pro)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,  # Max value for 32-bit signed integer
                    "control_after_generate": "randomize",
                    "tooltip": "Seed for reproducible outputs (0 to 2147483647)"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional system prompt to guide the model's behavior"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Optional image(s) to use as context or reference"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (leave empty to use GEMINI_API_KEY env variable)",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "password": True
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"
    DESCRIPTION = "Generate images and text using Google Gemini API"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs before execution."""
        api_key = kwargs.get('api_key', '')
        prompt = kwargs.get('prompt', '')
        seed = kwargs.get('seed', 42)
        
        if not api_key and not os.environ.get("GEMINI_API_KEY"):
            return "No API key provided. Set GEMINI_API_KEY environment variable or provide API key in the node."
        
        if not prompt.strip():
            return "Prompt cannot be empty"
        
        if seed < 0 or seed > 2147483647:
            return f"Seed must be between 0 and 2147483647 (32-bit integer range), got {seed}"
        
        return True
    
    def generate(
        self,
        prompt: str,
        model: str,
        seed: int,
        system_prompt: str = "",
        images: Optional[torch.Tensor] = None,
        api_key: str = ""
    ) -> Tuple[torch.Tensor, str]:
        
        # Get API key
        api_key = api_key if api_key else os.environ.get("GEMINI_API_KEY")
        
        try:
            client = GeminiAPIClient(api_key=api_key)
            
            image_list = None
            if images is not None:
                image_list = []
                if images.dim() == 4:
                    for i in range(images.shape[0]):
                        image_list.append(images[i:i+1])
                else:
                    image_list.append(images.unsqueeze(0))
            
            # Check if model supports image generation
            if "image" in model.lower():
                # Use image generation for models that support it
                output_images, text_response = client.generate_image(
                    prompt=prompt,
                    model=model,
                    images=image_list,
                    seed=seed,
                    system_prompt=system_prompt if system_prompt else None
                )
            else:
                # Use text generation for text-only models
                text_response = client.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt if system_prompt else None,
                    model=model,
                    images=image_list,
                    seed=seed
                )
                # Return dummy image for text-only models
                output_images = [torch.zeros((1, 64, 64, 4))]
            
            if output_images:
                result_tensor = torch.cat(output_images, dim=0)
            else:
                result_tensor = torch.zeros((1, 64, 64, 4))
            
            return (result_tensor, text_response)
            
        except Exception as e:
            error_msg = str(e)
            print(f"[NanoBanana Gemini] Error: {error_msg}", file=sys.stderr)
            print(f"[NanoBanana Gemini] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            # Raise exception to show error in ComfyUI with red node border
            raise RuntimeError(f"Gemini API Error: {error_msg}")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NanoBananaGeminiImageNode": NanoBananaGeminiImageNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGeminiImageNode": "Nano Banana Gemini"
}