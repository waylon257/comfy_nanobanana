import os
import base64
from typing import Optional, List, Union, Dict, Any
from google import genai
from google.genai import types
import torch
import numpy as np
from PIL import Image
from io import BytesIO


class GeminiAPIClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY env variable not set")
        
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
        
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if np_image.shape[-1] == 1:
            np_image = np_image.squeeze(-1)
            return Image.fromarray(np_image, mode='L')
        elif np_image.shape[-1] == 3:
            return Image.fromarray(np_image, mode='RGB')
        elif np_image.shape[-1] == 4:
            # Convert RGBA to RGB for sending to API
            return Image.fromarray(np_image[:, :, :3], mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {np_image.shape}")
    
    def bytes_to_tensor(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to torch tensor."""
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGBA for ComfyUI
        if image.mode != 'RGBA':
            if image.mode == 'RGB':
                # Add alpha channel
                image = image.convert('RGBA')
            else:
                image = image.convert('RGBA')
        
        np_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_array)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        max_output_tokens: int = 8192
    ) -> str:
        """Generate text using Gemini API."""
        contents = []
        
        # Combine system prompt and main prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Add text prompt
        contents.append(full_prompt)
        
        # Add images if provided
        if images:
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                contents.append(pil_image)
        
        # Configure generation
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        
        if seed is not None:
            config.seed = seed
        
        # Generate content
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        # Extract text from response
        return response.text if response.text else ""
    
    def generate_image(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash-image-preview",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> tuple[List[torch.Tensor], str]:
        """Generate images using Gemini API with image generation model."""
        
        # For image generation with gemini-2.5-flash-image-preview, use chat interface
        chat = self.client.chats.create(model=model)
        
        # Prepare message contents
        message_contents = []
        
        # Combine system prompt and main prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        message_contents.append(full_prompt)
        
        # Add input images if provided
        if images:
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                message_contents.append(pil_image)
        
        # Send message to generate image
        response = chat.send_message(message_contents)
        
        output_images = []
        text_output = ""
        
        # Process response
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle text parts
                    if part.text is not None:
                        text_output += part.text
                    # Handle image parts
                    elif part.inline_data is not None:
                        try:
                            # Get image data - inline_data.data is already bytes
                            if hasattr(part.inline_data, 'data'):
                                image_bytes = part.inline_data.data
                            else:
                                # If it's a dict
                                image_bytes = part.inline_data.get('data', b'')
                            
                            if image_bytes:
                                # Convert bytes to tensor
                                tensor = self.bytes_to_tensor(image_bytes)
                                output_images.append(tensor)
                        except Exception as e:
                            print(f"Error processing image from response: {e}")
                            import traceback
                            traceback.print_exc()
        
        # Return empty tensor if no images generated
        if not output_images:
            dummy_tensor = torch.zeros((1, 64, 64, 4))
            output_images = [dummy_tensor]
        
        return output_images, text_output