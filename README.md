# LLM Inference Hardware Calculator

A web-based calculator to estimate hardware requirements for running Large Language Models (LLMs) in inference mode. This tool helps you determine the VRAM and system RAM needed for running different LLM configurations.

## Features

- Calculate VRAM requirements based on:
  - Model size (number of parameters)
  - Quantization method (FP32/FP16/INT8/INT4/etc.)
  - Context length
  - KV cache settings
- Support for both discrete GPUs and unified memory systems
- Estimates for:
  - Required VRAM
  - Minimum system RAM
  - On-disk model size
  - Number of GPUs needed

## Development

This project uses React + TypeScript + Vite.

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Docker

- to use Docker and docker compose first create a `.env` file based on the [.env.example](.env.example) and set a PORT that should be exposed
- then run `docker compose up -d --build` to run the app

## Python CLI & Module

A standalone Python implementation (`calculator.py`) is also included, allowing hardware calculations directly from the command line or as an importable module:

### CLI Usage

```bash
# Basic calculation for an 8B model with Q4 quantization and 4096 context length
python calculator.py --params 8 --quant Q4 --context 4096

# Calculate for a 70B model with a 24GB GPU
python calculator.py --params 70 --quant Q4 --context 8192 --gpu-vram 24
```

### Module Usage

```python
from calculator import calculate_hardware_recommendation

recommendation = calculate_hardware_recommendation(
    params_billions=8.0,
    model_quant="Q4",
    context_length=4096,
    gpu_vram=24.0,
)
print(recommendation)
```

## Technical Notes

- Calculations are approximations and may vary based on specific implementations
- VRAM estimates include overhead for KV cache when enabled
- Unified memory calculations assume up to 75% of system RAM can be used as VRAM
- Discrete GPU calculations assume 24GB VRAM cards (like RTX 3090/4090)

## License

MIT
