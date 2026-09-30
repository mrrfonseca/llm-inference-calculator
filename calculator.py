"""
LLM Inference Hardware Calculator (Port em Python)
Baseado no projeto: https://github.com/alexziskind1/llm-inference-calculator
"""

import math
from typing import Literal, Dict, Any

MemoryMode = Literal['DISCRETE_GPU', 'UNIFIED_MEMORY']
ModelQuantization = Literal['F32', 'F16', 'Q8', 'Q6', 'Q5', 'Q4', 'Q3', 'Q2', 'GPTQ', 'AWQ']
KvCacheQuantization = Literal['F32', 'F16', 'Q8', 'Q5', 'Q4']
InferenceMode = Literal['incremental', 'bulk']

MODEL_QUANT_FACTORS: Dict[str, float] = {
    'F32': 4.0,
    'F16': 2.0,
    'Q8': 1.0,
    'Q6': 0.75,
    'Q5': 0.625,
    'Q4': 0.5,
    'Q3': 0.375,
    'Q2': 0.25,
    'GPTQ': 0.4,
    'AWQ': 0.35,
}

KV_CACHE_FACTORS: Dict[str, float] = {
    'F32': 4.0,
    'F16': 2.0,
    'Q8': 1.0,
    'Q5': 0.625,
    'Q4': 0.5,
}


def get_model_quant_factor(quant: str) -> float:
    return MODEL_QUANT_FACTORS.get(quant, 1.0)


def get_kv_cache_quant_factor(quant: str) -> float:
    return KV_CACHE_FACTORS.get(quant, 1.0)


def calculate_on_disk_size(params_billions: float, model_quant: str) -> float:
    model_factor = get_model_quant_factor(model_quant)
    bits_per_param = model_factor * 8
    total_bits = params_billions * 1e9 * bits_per_param
    return total_bits / 8 / 1e9


def calculate_required_vram(
    params_billions: float,
    model_quant: str = 'Q4',
    context_length: int = 4096,
    use_kv_cache: bool = True,
    kv_cache_quant: str = 'F16',
    inference_mode: str = 'incremental',
) -> float:
    base_model_mem = params_billions * get_model_quant_factor(model_quant)
    context_mem = 0.0

    if inference_mode == 'incremental':
        if use_kv_cache:
            alpha_at_2048 = 0.2
            kv_factor = get_kv_cache_quant_factor(kv_cache_quant)
            kv_scale = context_length / 2048.0
            context_mem = base_model_mem * alpha_at_2048 * kv_scale * kv_factor
    else:  # bulk
        bulk_alpha_at_2048 = 0.5
        bulk_scale = context_length / 2048.0
        context_mem = base_model_mem * bulk_alpha_at_2048 * bulk_scale

        if use_kv_cache:
            kv_factor = get_kv_cache_quant_factor(kv_cache_quant)
            context_mem += base_model_mem * 0.1 * kv_factor * bulk_scale

    total_vram = (base_model_mem + context_mem) * 1.1  # overhead de ~10%
    return total_vram


def calculate_hardware_recommendation(
    params_billions: float,
    model_quant: str = 'Q4',
    context_length: int = 4096,
    use_kv_cache: bool = True,
    kv_cache_quant: str = 'F16',
    memory_mode: str = 'DISCRETE_GPU',
    system_memory: float = 32.0,
    gpu_vram: float = 24.0,
    inference_mode: str = 'incremental',
) -> Dict[str, Any]:
    required_vram = calculate_required_vram(
        params_billions,
        model_quant,
        context_length,
        use_kv_cache,
        kv_cache_quant,
        inference_mode,
    )

    base_system_ram_needed = (
        params_billions * get_model_quant_factor(model_quant) * 0.5
        + (context_length / 1024.0 if inference_mode == 'bulk' else 0.0)
    )
    system_ram_needed = max(8.0, base_system_ram_needed)
    fits_unified = (memory_mode == 'UNIFIED_MEMORY') and (system_memory >= required_vram)

    gpus_required = 0
    gpu_type = ''

    if memory_mode == 'DISCRETE_GPU':
        gpus_required = math.ceil((required_vram * 1.2) / gpu_vram)
        if gpus_required == 1:
            gpu_type = f"1x GPU de {int(gpu_vram)}GB"
        elif gpus_required <= 8:
            gpu_type = f"{gpus_required}x GPUs de {int(gpu_vram)}GB"
        else:
            gpu_type = f"Excede 8x GPUs de {int(gpu_vram)}GB"
            gpus_required = 0
    else:
        gpu_type = f"Memória Unificada ({int(system_memory)}GB)"
        gpus_required = 0

    on_disk_gb = calculate_on_disk_size(params_billions, model_quant)

    return {
        "params_billions": params_billions,
        "model_quant": model_quant,
        "context_length": context_length,
        "on_disk_gb": round(on_disk_gb, 2),
        "vram_needed_gb": round(required_vram, 2),
        "system_ram_needed_gb": round(system_ram_needed, 2),
        "gpu_type": gpu_type,
        "gpus_required": gpus_required,
        "fits_unified": fits_unified,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Calculadora de Hardware para Inferência de LLM")
    parser.add_argument("--params", "-p", type=float, default=8.0, help="Parâmetros em bilhões (ex: 7, 8, 70)")
    parser.add_argument("--quant", "-q", choices=list(MODEL_QUANT_FACTORS.keys()), default="Q4", help="Quantização")
    parser.add_argument("--context", "-c", type=int, default=4096, help="Tamanho da janela de contexto")
    parser.add_argument("--gpu-vram", type=float, default=24.0, help="VRAM por GPU discreta em GB (padrão: 24)")
    parser.add_argument("--mode", choices=["incremental", "bulk"], default="incremental", help="Modo de inferência")

    args = parser.parse_args()
    res = calculate_hardware_recommendation(
        params_billions=args.params,
        model_quant=args.quant,
        context_length=args.context,
        gpu_vram=args.gpu_vram,
        inference_mode=args.mode,
    )

    print("=" * 60)
    print(f" LLM INFERENCE CALCULATOR ({args.params}B parâmetros - {args.quant})")
    print("=" * 60)
    print(f"• Tamanho em Disco:        ~{res['on_disk_gb']} GB")
    print(f"• VRAM Necessária:         ~{res['vram_needed_gb']} GB")
    print(f"• RAM do Sistema Mínima:   ~{res['system_ram_needed_gb']} GB")
    print(f"• Recomendação de GPU:     {res['gpu_type']}")
    print("=" * 60)
