# Labs

One lab per theory post on [The Backward Pass](https://srini-anirudh.github.io/blogs/). A lab is the code companion to the post: it recomputes the post's numbers, makes its mechanisms happen on small models, and saves the real outputs in `results/`. Where a measurement disagrees with the post, the lab README says so.

Each post links to its lab near the end (the "Run the code" box), and each lab README links back to its post, so you can move between them in one click. The twelve tutorials (`c01` to `c12`) are the longer, tested builds; see the [top-level README](../README.md).

Run any lab from the repository root, after the setup in the top-level README:

```bash
pytest labs/<lab> -m "not slow and not gpu and not network"   # the fast tests
```

Each lab README lists its scripts, the hardware each result came from, and what was not measured.

## Vision-language models

| Post | Lab |
|---|---|
| [Aligning Vision-Language Models](https://srini-anirudh.github.io/blogs/aligning-vision-language-models/) | [`aligning-vision-language-models`](aligning-vision-language-models) |
| [From VLMs to VLAs](https://srini-anirudh.github.io/blogs/from-vlms-to-vlas/) | [`from-vlms-to-vlas`](from-vlms-to-vlas) |
| [How VLMs Read Documents](https://srini-anirudh.github.io/blogs/how-vlms-read-documents/) | [`how-vlms-read-documents`](how-vlms-read-documents) |
| [How Vision-Language Models Scale](https://srini-anirudh.github.io/blogs/how-vision-language-models-scale/) | [`how-vision-language-models-scale`](how-vision-language-models-scale) |
| [How the Open VLM Families Were Built](https://srini-anirudh.github.io/blogs/how-the-open-vlm-families-were-built/) | [`how-the-open-vlm-families-were-built`](how-the-open-vlm-families-were-built) |
| [Inside a Modern Vision Encoder](https://srini-anirudh.github.io/blogs/inside-a-modern-vision-encoder/) | [`inside-a-modern-vision-encoder`](inside-a-modern-vision-encoder) |
| [Measuring What a VLM Can Do](https://srini-anirudh.github.io/blogs/measuring-what-a-vlm-can-do/) | [`measuring-what-a-vlm-can-do`](measuring-what-a-vlm-can-do) |
| [Models That See and Draw](https://srini-anirudh.github.io/blogs/models-that-see-and-draw/) | [`models-that-see-and-draw`](models-that-see-and-draw) |
| [One Model, Many Visual Skills](https://srini-anirudh.github.io/blogs/one-model-many-visual-skills/) | [`one-model-many-visual-skills`](one-model-many-visual-skills) |
| [Rewarding What the Model Sees](https://srini-anirudh.github.io/blogs/rewarding-what-the-model-sees/) | [`rewarding-what-the-model-sees`](rewarding-what-the-model-sees) |
| [Searching With Pixels](https://srini-anirudh.github.io/blogs/searching-with-pixels/) | [`searching-with-pixels`](searching-with-pixels) |
| [Seeing in Time](https://srini-anirudh.github.io/blogs/seeing-in-time/) | [`seeing-in-time`](seeing-in-time) |
| [Teaching Language Models to See](https://srini-anirudh.github.io/blogs/teaching-language-models-to-see/) | [`teaching-language-models-to-see`](teaching-language-models-to-see) |
| [The Cost of a Picture](https://srini-anirudh.github.io/blogs/the-cost-of-a-picture/) | [`the-cost-of-a-picture`](the-cost-of-a-picture) |
| [The VLM Architecture Gallery](https://srini-anirudh.github.io/blogs/the-vlm-architecture-gallery/) | [`the-vlm-architecture-gallery`](the-vlm-architecture-gallery) |
| [Thinking With Images](https://srini-anirudh.github.io/blogs/thinking-with-images/) | [`thinking-with-images`](thinking-with-images) |
| [VLMs That Act](https://srini-anirudh.github.io/blogs/vlms-that-act/) | [`vlms-that-act`](vlms-that-act) |
| [What Happens Inside a VLM](https://srini-anirudh.github.io/blogs/what-happens-inside-a-vlm/) | [`what-happens-inside-a-vlm`](what-happens-inside-a-vlm) |
| [When Language Must Point](https://srini-anirudh.github.io/blogs/when-language-must-point/) | [`when-language-must-point`](when-language-must-point) |

## Deep Learning Foundations

| Post | Lab |
|---|---|
| [Choosing the Next Token](https://srini-anirudh.github.io/blogs/choosing-the-next-token/) | [`choosing-the-next-token`](choosing-the-next-token) |
| [How Attention Moves Information](https://srini-anirudh.github.io/blogs/how-attention-moves-information/) | [`how-attention-moves-information`](how-attention-moves-information) |
| [How Models Know Where They Are](https://srini-anirudh.github.io/blogs/how-models-know-where-they-are/) | [`how-models-know-where-they-are`](how-models-know-where-they-are) |
| [The Geometry of Normalization](https://srini-anirudh.github.io/blogs/the-geometry-of-normalization/) | [`the-geometry-of-normalization`](the-geometry-of-normalization) |
| [What Autograd Actually Does](https://srini-anirudh.github.io/blogs/what-autograd-actually-does/) | [`what-autograd-actually-does`](what-autograd-actually-does) |
| [What an Optimizer Actually Does](https://srini-anirudh.github.io/blogs/what-an-optimizer-actually-does/) | [`what-an-optimizer-actually-does`](what-an-optimizer-actually-does) |
| [Why Neural Networks Need Nonlinearity](https://srini-anirudh.github.io/blogs/why-neural-networks-need-nonlinearity/) | [`why-neural-networks-need-nonlinearity`](why-neural-networks-need-nonlinearity) |
| [Why Transformers Do Not Explode](https://srini-anirudh.github.io/blogs/why-transformers-do-not-explode/) | [`why-transformers-do-not-explode`](why-transformers-do-not-explode) |

## ML Systems

| Post | Lab |
|---|---|
| [How to Find What Is Slowing Your Model](https://srini-anirudh.github.io/blogs/how-to-find-what-is-slowing-your-model/) | [`how-to-find-what-is-slowing-your-model`](how-to-find-what-is-slowing-your-model) |
| [Inside a Training Step](https://srini-anirudh.github.io/blogs/inside-a-training-step/) | [`inside-a-training-step`](inside-a-training-step) |
| [Talk Is Not Cheap](https://srini-anirudh.github.io/blogs/talk-is-not-cheap/) | [`talk-is-not-cheap`](talk-is-not-cheap) |
| [The Mechanics of LLM Inference](https://srini-anirudh.github.io/blogs/the-mechanics-of-llm-inference/) | [`the-mechanics-of-llm-inference`](the-mechanics-of-llm-inference) |
| [The Parallelism Playbook](https://srini-anirudh.github.io/blogs/the-parallelism-playbook/) | [`the-parallelism-playbook`](the-parallelism-playbook) |
| [The Serving Playbook](https://srini-anirudh.github.io/blogs/the-serving-playbook/) | [`the-serving-playbook`](the-serving-playbook) |
| [Why Fast GPUs Still Wait for Memory](https://srini-anirudh.github.io/blogs/why-fast-gpus-still-wait-for-memory/) | [`why-fast-gpus-still-wait-for-memory`](why-fast-gpus-still-wait-for-memory) |
| [Why GPUs Are Built for Deep Learning](https://srini-anirudh.github.io/blogs/why-gpus-are-built-for-deep-learning/) | [`why-gpus-are-built-for-deep-learning`](why-gpus-are-built-for-deep-learning) |

## Tool Use & Agentic Models

| Post | Lab |
|---|---|
| [How Language Models Learn to Use Tools](https://srini-anirudh.github.io/blogs/how-language-models-learn-to-use-tools/) | [`how-language-models-learn-to-use-tools`](how-language-models-learn-to-use-tools) |
| [How Multi-Agent Systems Actually Coordinate](https://srini-anirudh.github.io/blogs/how-multi-agent-systems-actually-coordinate/) | [`how-multi-agent-systems-actually-coordinate`](how-multi-agent-systems-actually-coordinate) |
| [Inside an AI Agent Harness](https://srini-anirudh.github.io/blogs/inside-an-ai-agent-harness/) | [`inside-an-ai-agent-harness`](inside-an-ai-agent-harness) |
| [Memory Is More Than Context](https://srini-anirudh.github.io/blogs/memory-is-more-than-context/) | [`memory-is-more-than-context`](memory-is-more-than-context) |
| [Reinforcement Learning for Agents From First Principles](https://srini-anirudh.github.io/blogs/reinforcement-learning-for-agents-from-first-principles/) | [`reinforcement-learning-for-agents-from-first-principles`](reinforcement-learning-for-agents-from-first-principles) |
| [The Agent Evaluation Playbook](https://srini-anirudh.github.io/blogs/the-agent-evaluation-playbook/) | [`the-agent-evaluation-playbook`](the-agent-evaluation-playbook) |
| [The Making of an AI Agent](https://srini-anirudh.github.io/blogs/the-making-of-an-ai-agent/) | [`the-making-of-an-ai-agent`](the-making-of-an-ai-agent) |

## Post-training

| Post | Lab |
|---|---|
| [How LoRA Changes a Model Without Rewriting It](https://srini-anirudh.github.io/blogs/how-lora-changes-a-model-without-rewriting-it/) | [`how-lora-changes-a-model-without-rewriting-it`](how-lora-changes-a-model-without-rewriting-it) |
| [Mixing Data Without Losing Capabilities](https://srini-anirudh.github.io/blogs/mixing-data-without-losing-capabilities/) | [`mixing-data-without-losing-capabilities`](mixing-data-without-losing-capabilities) |
| [Reinforcement Learning from First Principles](https://srini-anirudh.github.io/blogs/reinforcement-learning-from-first-principles/) | [`reinforcement-learning-from-first-principles`](reinforcement-learning-from-first-principles) |
| [Teaching a Language Model to Follow Instructions](https://srini-anirudh.github.io/blogs/teaching-a-language-model-to-follow-instructions/) | [`teaching-a-language-model-to-follow-instructions`](teaching-a-language-model-to-follow-instructions) |
| [Teaching a Model What We Prefer](https://srini-anirudh.github.io/blogs/teaching-a-model-what-we-prefer/) | [`teaching-a-model-what-we-prefer`](teaching-a-model-what-we-prefer) |

## Architecture

| Post | Lab |
|---|---|
| [Across the CNNVerse](https://srini-anirudh.github.io/blogs/across-the-cnnverse/) | [`across-the-cnnverse`](across-the-cnnverse) |
| [An Image Is a Sentence](https://srini-anirudh.github.io/blogs/an-image-is-a-sentence/) | [`an-image-is-a-sentence`](an-image-is-a-sentence) |
| [Evolution of ML Architectures](https://srini-anirudh.github.io/blogs/evolution-of-ml-architectures/) | [`evolution-of-ml-architectures`](evolution-of-ml-architectures) |
| [One Transformer, Two Modalities](https://srini-anirudh.github.io/blogs/one-transformer-two-modalities/) | [`one-transformer-two-modalities`](one-transformer-two-modalities) |
| [What Changed Inside the Transformer?](https://srini-anirudh.github.io/blogs/what-changed-inside-the-transformer/) | [`what-changed-inside-the-transformer`](what-changed-inside-the-transformer) |

## Training Science

| Post | Lab |
|---|---|
| [How LLMs Are Pretrained](https://srini-anirudh.github.io/blogs/how-llms-are-pretrained/) | [`how-llms-are-pretrained`](how-llms-are-pretrained) |
| [Scaling Laws from First Principles](https://srini-anirudh.github.io/blogs/scaling-laws-from-first-principles/) | [`scaling-laws-from-first-principles`](scaling-laws-from-first-principles) |
| [The Puzzle of Overparameterization](https://srini-anirudh.github.io/blogs/the-puzzle-of-overparameterization/) | [`the-puzzle-of-overparameterization`](the-puzzle-of-overparameterization) |

## Reasoning Models

| Post | Lab |
|---|---|
| [How Models Improve Without Changing Their Weights](https://srini-anirudh.github.io/blogs/how-models-improve-without-changing-their-weights/) | [`how-models-improve-without-changing-their-weights`](how-models-improve-without-changing-their-weights) |
| [How Reinforcement Learning Teaches Models to Reason](https://srini-anirudh.github.io/blogs/how-reinforcement-learning-teaches-models-to-reason/) | [`how-reinforcement-learning-teaches-models-to-reason`](how-reinforcement-learning-teaches-models-to-reason) |
| [Thinking in Tokens](https://srini-anirudh.github.io/blogs/thinking-in-tokens/) | [`thinking-in-tokens`](thinking-in-tokens) |
