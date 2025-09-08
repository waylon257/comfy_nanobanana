# Comfy Nano Banana

Google Gemini API integration for ComfyUI - Generate images and text using Google's latest AI models, plus a dynamic batch images utility node.

## Features

- 🎨 **Image Generation** - Create images using Gemini's flash-image-preview model
- 📝 **Text Generation** - Generate text responses with optional image context
- 🖼️ **Multimodal Support** - Use images as input context for both text and image generation
- 🚀 **Concurrent Batch Processing** - Generate 1-4 images concurrently with progress tracking
- 🔀 **Dynamic Batch Images** - Combine multiple images into a single batch with automatic resizing
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

The extension adds two nodes to ComfyUI:

### 1. Nano Banana Gemini Node
Located under the "Nano Banana" category, this node interfaces with Google's Gemini API.

#### Inputs
- **prompt** (required): Text prompt for generation
- **model**: Gemini model to use (default: `gemini-2.5-flash-image-preview`)
- **batch_size**: Number of images to generate concurrently (1-4)
- **seed**: For reproducible outputs (0-2147483647)
- **system_prompt** (optional): Instructions to guide the model's behavior
- **images** (optional): Input images for context
- **api_key** (optional): Override environment variable
- **top_p** (optional): Nucleus sampling for text generation (0.0-1.0, default: 0.95)
- **max_tokens** (optional): Maximum tokens for text output (1-8192, default: 2048)

#### Outputs
- **images**: Generated images or placeholder for text-only models
- **text**: Text response from the model

#### Supported Models
- `gemini-2.5-flash-image-preview` - Image and text generation
- `gemini-2.5-pro` - Text generation only
- `gemini-2.5-flash` - Text generation only

### 2. Batch Images Node
Located under the "image" category, this utility node dynamically combines multiple images into a single batch.

#### Features
- **Dynamic Inputs**: Automatically adds/removes image inputs as you connect/disconnect
- **Auto-resize**: Mismatched images are automatically resized to match the first image's dimensions
- **Clean Interface**: Unused inputs are automatically removed
- **Flexible**: Connect any number of images from different sources

#### How to Use
1. Add "Batch Images" node from the image category
2. Connect your first image - the node automatically creates a new input
3. Connect additional images - each connection creates a new input slot
4. Disconnecting removes the unused input automatically
5. All images are resized to match the first image's dimensions and combined into a batch

## Examples

### Image Generation
1. Add "Nano Banana Gemini" node
2. Enter a prompt like "A cat wearing a wizard hat"
3. Connect output to Preview Image node

### Batch Image Generation
1. Add "Nano Banana Gemini" node
2. Set batch_size to 4
3. Enter your prompt
4. Get 4 variations generated concurrently with progress tracking

### Image-to-Image
1. Load an image
2. Connect to "images" input of Gemini node
3. Add prompt describing desired changes
4. Model will use the image as context

### Combining Multiple Images
1. Add "Batch Images" node
2. Connect images from different sources
3. The node automatically creates new inputs as you connect
4. All images are resized and combined into a single batch
5. Use the batch for further processing or saving

## Development

### Project Structure
```
comfy_nanobanana/
├── src/comfy_nanobanana/
│   ├── __init__.py
│   ├── nodes.py          # ComfyUI node implementations
│   └── gemini_api.py     # Gemini API client with async support
├── web/
│   ├── index.js          # Extension entry point
│   └── node/
│       └── batch_images_dynamic.js  # Dynamic input handling for Batch Images
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