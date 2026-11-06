"""benchlab: the evaluation mechanisms of "Measuring What a VLM Can Do", small enough to read.

metrics  scoring rules: VQA accuracy, ANLS, relaxed accuracy, IoU, gIoU / cIoU, POPE, CHAIR
mc       multiple choice: option parsing, answer extractors, CircularEval
stats    noise floor, bootstrap intervals, multi-modal gain and leakage, Bradley-Terry ratings
runner   batched SmolVLM / SmolLM2 inference on MMStar and POPE (needs the Hub, a GPU is faster)
"""
