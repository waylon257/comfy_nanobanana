# Comfy Nano Banana

Google Gemini API integration for ComfyUI - Generate images and text using Google's latest AI models.

## Features

- 🎨 **Image Generation** - Create images using Gemini's flash-image-preview model
- 📝 **Text Generation** - Generate text responses with optional image context
- 🖼️ **Multimodal Support** - Use images as input context for both text and image generation
- 🔒 **Secure API Key Handling** - Password field with environment variable support
- ⚡ **Smart Model Detection** - Automatically uses the right generation method based on model

## Installation

### Via ComfyUI Manager (Recommended)
1. Install [ComfyUI](https://docs.comfy.org/get_started)
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Search for "Nano Banana" in ComfyUI-Manager and install
4. Restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfy_nanobanana
cd comfy_nanobanana
pip install -r requirements.txt
```

## Setup

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set your API key:
   - **Option 1**: Set environment variable `GEMINI_API_KEY`
   - **Option 2**: Enter directly in the node's API key field

## Usage

The extension adds a **"Nano Banana Gemini"** node under the "Nano Banana" category.

### Node Inputs
- **prompt** (required): Text prompt for generation
- **model**: Gemini model to use (default: `gemini-2.5-flash-image-preview`)
- **seed**: For reproducible outputs (0-2147483647)
- **system_prompt** (optional): Instructions to guide the model's behavior
- **images** (optional): Input images for context
- **api_key** (optional): Override environment variable

### Node Outputs
- **images**: Generated images or placeholder for text-only models
- **text**: Text response from the model

### Supported Models
- `gemini-2.5-flash-image-preview` - Image and text generation
- `gemini-2.5-pro` - Text generation only
- `gemini-2.5-flash` - Text generation only

## Examples

### Image Generation
1. Add "Nano Banana Gemini" node
2. Enter a prompt like "A cat wearing a wizard hat"
3. Connect output to Preview Image node

### Image-to-Image
1. Load an image
2. Connect to "images" input
3. Add prompt describing desired changes
4. Model will use the image as context

## Development

### Project Structure
```
comfy_nanobanana/
├── src/comfy_nanobanana/
│   ├── __init__.py
│   ├── nodes.py          # ComfyUI node implementation
│   └── gemini_api.py     # Gemini API client
├── requirements.txt
└── pyproject.toml
```

### Dev Setup
```bash
cd comfy_nanobanana
pip install -e .[dev]
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

**"No API key provided"**
- Ensure `GEMINI_API_KEY` is set or enter key in node

**"Seed must be between 0 and 2147483647"**
- Gemini API requires 32-bit integer seeds

**Empty image output with text models**
- Normal behavior - text-only models return placeholder image

## Contributing

Pull requests welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation

## License

MIT License - see [LICENSE](LICENSE) file

## Credits

Created with [ComfyUI Extension Template](https://github.com/Comfy-Org/cookiecutter-comfy-extension)