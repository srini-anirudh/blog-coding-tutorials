"""vlmtrain: a staged multimodal training system for a vision-language model.

Modules, in the order the tutorial builds them:
    synth/, data       synthetic tasks with exact labels, one {"images", "messages"} format
    processor          chat template, image-token expansion, loss mask
    freeze, optim      freeze presets, per-module learning rates
    mixture, replay    task mixtures, text replay, the forgetting check
    resolution, packing  tiles and token budgets
    lora               LoRA placement on vision, projector and LM
    grounding, metrics, evaluate   points, boxes, masks and per-task evaluation
    train              the staged trainer
    c07_adapter, model C7's TinyVLM, or a small standalone model with the same contract
"""
