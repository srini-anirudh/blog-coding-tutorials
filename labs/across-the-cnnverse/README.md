# Lab: Across the CNNVerse

This is the code companion to the post
[Across the CNNVerse: A Visual Guide to CNN Architectures](https://srini-anirudh.github.io/blogs/across-the-cnnverse/).
The post makes many claims about convolution and about nine networks, from LeNet-5 to ConvNeXt. This lab builds each
piece in small PyTorch files and measures the claims you can check without ImageNet.

What you can run:

- Convolution three ways: as nested loops, as one matrix multiply (im2col) with stride, padding, dilation and
  groups, and transposed convolution as a scatter. Each is checked against `torch.nn.functional`. The post's cost
  tables are recomputed from the formulas.
- The receptive-field recursion, checked against the input region that gradients actually reach. Then the
  smaller *effective* receptive field.
- Translation equivariance, measured. It is exact for stride 1 and breaks for odd shifts after stride 2.
- The nine networks written from their papers' layer tables. A spec engine reads parameters, MACs and activations
  off the modules. It lands within 2.3% of every published figure.
- Gradient flow at initialisation in plain and residual stacks, and the degradation experiment that motivated
  ResNet, run small on CIFAR-10.
- Latency against MACs, single-operator throughput, and what fills memory during a training step, with and
  without activation checkpointing.

Nothing here is trained on ImageNet. Every ImageNet accuracy in the tables is the paper's number as the post quotes
it, and the lab does not reproduce it.

## Layout

```
labs/across-the-cnnverse/
├── cnnverse/
│   ├── conv.py        # loops, im2col, transposed scatter, output-size and cost formulas, classic kernels
│   ├── receptive.py   # the recursion, gradient footprints, effective receptive field
│   ├── zoo.py         # LeNet-5, AlexNet, VGG-16, GoogLeNet, ResNet-50, DenseNet-121, MobileNetV2,
│   │                  # EfficientNet-B0..B7, ConvNeXt-T, ResNeXt-50, SE-ResNet-50, CIFAR plain/residual nets
│   ├── profile.py     # spec engine: params, MACs, activations per layer (on the meta device); saved-tensor bytes
│   ├── data.py        # CIFAR-10 from the Hugging Face Hub as uint8 tensors; GPU-side crop and flip
│   └── report.py      # results files start with the hardware and the command
├── scripts/           # 01..07, listed below, and run_all.sh
├── results/           # real outputs of every script
└── tests/
```

## Setup and tests

From the repository root, after `pip install -e ".[dev]"`. The lab needs `torch`, and `datasets` for CIFAR-10.
It does not use torchvision.

```bash
python -m pytest labs/across-the-cnnverse -m "not slow and not gpu and not network"   # 37 tests, about 20 s
python -m pytest labs/across-the-cnnverse -m network    # downloads CIFAR-10 (about 120 MB) the first time
python -m pytest labs/across-the-cnnverse -m gpu        # needs CUDA; one test, not yet run
```

## Run everything

```bash
bash labs/across-the-cnnverse/scripts/run_all.sh
```

Or one at a time:

```bash
python labs/across-the-cnnverse/scripts/01_conv_from_scratch.py      # a few seconds
python labs/across-the-cnnverse/scripts/02_receptive_fields.py       # a few seconds
python labs/across-the-cnnverse/scripts/03_equivariance.py           # a few seconds
python labs/across-the-cnnverse/scripts/04_architecture_zoo.py       # about 20 s, no arithmetic (meta device)
python labs/across-the-cnnverse/scripts/05_residual_gradients.py     # about 10 s
python labs/across-the-cnnverse/scripts/07_time_and_space.py --reps 10                  # about 1 min on CPU
python labs/across-the-cnnverse/scripts/07_time_and_space.py --device cuda --reps 20    # GPU latency and memory
python labs/across-the-cnnverse/scripts/06_degradation.py --epochs 20 --depths 20 --tag _depth20_cuda   # GPU
python labs/across-the-cnnverse/scripts/06_degradation.py --epochs 20 --depths 56 --tag _depth56_cuda   # GPU
python labs/across-the-cnnverse/scripts/06_degradation.py --epochs 3 --depths 20 --subset 2000 --tag _cpu_quick
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| The operation (Figure 01) | `01_conv_from_scratch.py`, `cnnverse/conv.py` |
| What a filter sees (Figure 02) | `01_conv_from_scratch.py` |
| Stride, padding, dilation (Figure 03) | `01_conv_from_scratch.py` |
| Receptive fields (Figure 04) | `02_receptive_fields.py` |
| Why convolution wins (Figure 05) | `03_equivariance.py` |
| 1D, 2D, 3D (Figure 06) | `01_conv_from_scratch.py`, `03_equivariance.py` |
| The convolution family (Figure 07) | `01_conv_from_scratch.py`, `07_time_and_space.py` |
| Layer-by-layer explorer (Figure 09) | `04_architecture_zoo.py`, `cnnverse/zoo.py`, `cnnverse/profile.py` |
| LeNet-5, AlexNet, VGG, Inception (Figures 10, 11) | `04_architecture_zoo.py` |
| ResNet (Figure 12) | `05_residual_gradients.py`, `06_degradation.py` |
| DenseNet and ResNeXt, MobileNet and SENet (Figures 13, 14) | `04_architecture_zoo.py`, `07_time_and_space.py` |
| EfficientNet (Figure 15) | `04_architecture_zoo.py` |
| ConvNeXt (Figure 16) | `04_architecture_zoo.py` (cost only) |
| Time and space (Figure 18) | `07_time_and_space.py` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it.

CPU: AMD EPYC 7R13, 8 threads, torch 2.15.0. The machines were shared with other jobs, so CPU timings are
orders of magnitude, not benchmarks. Counts, shapes and errors do not depend on load.

GPU: NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2. Latency and memory runs use fp32 with TF32 off. The CIFAR
training runs use bf16 autocast. Each GPU job ran alone on its GPU.

### The operation and its cost (CPU; exact arithmetic, no timing)

| Check | Result |
|---|---|
| Loops vs `F.conv2d`, 6x6 input, 3x3 filter | max abs error 0 |
| im2col vs `F.conv2d`, 8 settings of k, stride, padding, dilation, groups | max abs error at most 5.3e-5 (fp32) |
| Transposed conv as scatter vs `F.conv_transpose2d` | max abs error 0 |
| k=3, s=2 transposed conv of a constant image | output takes the values 1, 2 and 4: the checkerboard |
| Kernels that sum to zero (Sobel, Laplacian) on a flat region | response 0; blurs that sum to one return 1 |
| 224 px through 16 unpadded 3x3 layers | 192 px |
| Three 3x3 vs one 7x7 at C=64 | 110,592 vs 200,704 weights, 45% fewer |
| Convolution family at 56x56, 128 channels | all 7 rows of the post's table reproduced exactly |
| Depthwise separable vs standard | 1/C_out + 1/k^2 = 0.1189, 8.4x cheaper |
| 3x3x64 from 2D at 56^2 to 3D at 16 frames | 48x the MACs |

### Receptive fields (CPU)

| Stacking strategy | Params at C=64 | Recursion | Measured from gradients |
|---|---|---|---|
| 3x3, stride 1, 8 layers | 294,912 | 17 | 17 |
| 3x3, stride 2 every second layer, 8 layers | 294,912 | 61 | 61 |
| 3x3, dilations 1, 2, 4, 8 twice | 294,912 | 61 | 61 |
| 7x7, stride 1, 3 layers | 602,112 | 19 | 19 |

The recursion gives AlexNet a theoretical receptive field of 195 px and VGG-16 one of 212 px. With random weights and
ReLU, 8 layers of 3x3 put 74% of the gradient inside the central 9x9 square, which is 28% of the 17x17 field. The
gradient falls off toward the edge, as Luo et al. (2017) describe.

### Equivariance (CPU)

| Operation | Relative error, shift 1 | Shift 2 | Shift 4 |
|---|---|---|---|
| Stride-1 conv | 0.000 | 0.000 | 0.000 |
| Stride-2 conv, output shifted by half | 1.488 | 0.000 | 0.000 |
| 2x2 max pool, output shifted by half | 0.801 | 0.000 | 0.000 |

A dense layer from a 224x224x3 image to 1000 units has 150.5 M parameters. Global average pooling after a conv
changes by less than 4e-9 under shifts. A 3-D conv is exactly equivariant to time shifts. A 2-D conv over frames
stacked as channels changes its output by 158% (relative) under the same time shift.

### The nine networks, computed from their layer specs (meta device: no arithmetic, no hardware dependence)

| Network | Params | Published | GMACs | Published | Activations / image | Classifier share of params | Top-1 (paper, not reproduced) |
|---|---|---|---|---|---|---|---|
| LeNet-5 | 61,706 | 61,706 | 0.0004 | 0.0004 | 0.01 M | 17.8% | MNIST |
| AlexNet | 60.97 M | 61.0 M | 0.724 | 0.72 | 0.78 M | 96.2% | 57.1 |
| VGG-16 | 138.36 M | 138.4 M | 15.47 | 15.5 | 15.09 M | 89.4% | 71.6 |
| GoogLeNet | 7.00 M | 7.0 M | 1.583 | 1.6 | 4.65 M | 14.6% | 69.8 |
| ResNet-50 | 25.56 M | 25.6 M | 4.089 | 4.1 | 11.32 M | 8.0% | 76.1 |
| DenseNet-121 | 7.98 M | 8.0 M | 2.834 | 2.9 | 7.28 M | 12.8% | 74.4 |
| MobileNetV2 | 3.50 M | 3.5 M | 0.301 | 0.30 | 6.68 M | 36.5% | 72.0 |
| EfficientNet-B0 | 5.29 M | 5.3 M | 0.386 | 0.39 | 6.75 M | 24.2% | 77.1 |
| ConvNeXt-T | 28.59 M | 28.6 M | 4.456 | 4.5 | 13.44 M | 2.7% | 82.1 |
| ResNeXt-50 32x4d | 25.03 M | 25.0 M | 4.230 | 4.2 | 14.60 M | 8.2% | 77.8 |
| SE-ResNet-50 | 28.09 M | 28.1 M | 4.092 | 4.1 | 11.33 M | 7.3% | 77.6 |

The largest gap is DenseNet-121's MACs, 2.3% below the post's 2.9 G. MACs count convolutions and linear layers only.
"Activations" counts the outputs of conv, linear and pooling layers. Other findings from `04_architecture_zoo.txt`:

| Claim in the post | Measured |
|---|---|
| LeNet-5's C3 table | the 1998 form (sparse C3, trainable subsampling) has 60,840 parameters; 61,706 is the common dense form |
| AlexNet's grouping saves params and compute | 60.97 M and 0.724 GMACs grouped; 62.38 M and 1.135 GMACs ungrouped |
| VGG-16's fc6 holds 102 M parameters | 102.8 M; the three FC layers hold 123.6 M |
| VGG-16 costs 3.8x ResNet-50's compute | 3.8x |
| SE adds about 2.5 M params and under 1% FLOPs to ResNet-50 | +2.53 M params, +0.06% MACs |
| ResNeXt-50 costs about the same as ResNet-50 | 0.98x params, 1.03x MACs |
| MobileNetV2 has 13x fewer FLOPs than ResNet-50 | 13.6x |
| 1x1 bottleneck in Inception's 5x5 branch (480 to 128) | 1,536,000 vs 117,760 weights, 13x fewer |
| The same, for whole modules | module 3a: 2.4x fewer MACs; module 4a: 4.2x fewer |
| EfficientNet's alpha beta^2 gamma^2 is about 2 | 1.92; but one compound step (phi = 1) costs 2.57x B0's MACs |
| EfficientNet-B7: 66 M params, 37 GMACs | 66.35 M, 37.75 GMACs |

### Gradient flow at initialisation (CPU)

Gradient norm at the first layer divided by the gradient norm at the last layer, 16 channels, mean of 3 seeds:

| Stack | Depth 8 | Depth 20 | Depth 56 | Depth 110 |
|---|---|---|---|---|
| Plain, no BN, PyTorch default init | 1.2e-3 | 5.1e-9 | 0 | 0 |
| Plain, no BN, He init | 0.19 | 0.28 | 0.15 | 0.09 |
| Plain, with BN | 0.85 | 5.2 | 2.2e3 | 4.6e6 |
| Residual, with BN | 0.56 | 0.66 | 1.22 | 1.80 |

### Degradation on CIFAR-10 (GPU)

Plain and residual nets of depth 20 and 56 (He et al. 2015, section 4.2) on all 50,000 CIFAR-10 training images.
The two kinds have exactly the same weights: the shortcut is parameter-free. Training: 20 epochs, batch 128,
SGD with momentum 0.9, learning rate 0.1 divided by 10 at 50% and 75% of training, pad-and-crop plus flip, one seed
per network. NVIDIA H100 80GB HBM3, bf16 autocast. Times are 70 to 195 s per network.

| Net | Params | Train error, last epoch (augmented) | Train error, clean 10k images | Test error |
|---|---|---|---|---|
| plain-20 | 269,722 | 0.150 | 0.142 | 0.163 |
| ResNet-20 | 269,722 | 0.103 | 0.096 | 0.127 |
| plain-56 | 853,018 | 0.535 | 0.540 | 0.536 |
| ResNet-56 | 853,018 | 0.147 | 0.134 | 0.172 |

The post's claim holds in direction. The deeper plain net has much higher *training* error than the shallower one
(0.540 against 0.142 on clean training images), so it is an optimisation failure, not overfitting. The residual nets
do not degrade in training error. They do not improve either: ResNet-56 has higher test error than ResNet-20 here,
because 20 epochs is short for 0.85 M parameters.

Caveats. The plain-56 gap is much larger than the 2-point gap in the paper's Figure 6. The reason is not known. This
run uses learning rate 0.1 from the first step with no warm-up, and plain-56 is still falling at epoch 20
(0.610 at epoch 10, 0.535 at epoch 20), so it may be slow rather than stuck. A lower learning rate or a longer run
was not tried. One seed per network. `results/06_degradation_cpu_quick.txt` is a smoke run (3 epochs on 2,000 images,
EPYC CPU): at that size ResNet-20 is ahead of plain-20 (0.729 against 0.785 training error), which shows nothing
about degradation.

### Time and space

Latency and throughput come from `07_time_and_space.py`. fp32, eval mode, median of 10 (CPU) or 20 (GPU) runs.

| Measurement | AMD EPYC 7R13, 8 threads | NVIDIA H100 80GB HBM3 |
|---|---|---|
| MobileNetV2 vs ResNet-50, MACs | 13.6x fewer | 13.6x fewer |
| MobileNetV2 vs ResNet-50, speed at batch 1 | 2.0x faster | 1.1x faster |
| MobileNetV2 vs ResNet-50, speed at batch 64 | not measured | 3.2x faster |
| VGG-16 vs ResNet-50, MACs / time at batch 1 | 3.8x / 6.2x | 3.8x / 0.34x (VGG-16 is faster) |
| DenseNet-121 vs ResNet-50, MACs / time at batch 1 | 0.69x / 1.65x | 0.69x / 2.55x |
| EfficientNet-B0 vs ResNet-50, MACs / time at batch 1 | 0.09x / 0.78x | 0.09x / 1.15x |
| Depthwise 3x3 vs standard 3x3 at 56x56x128: MACs / time | 128x fewer / 36x faster (batch 8) | 128x fewer / 5.2x faster (batch 64) |
| Achieved GMAC/s, standard 3x3 / depthwise | 71.6 / 20.3 | 20,914 / 851 |
| MAC per byte moved, standard 3x3 / 1x1 / grouped / depthwise | 140.8 / 16.0 / 4.5 / 1.1 | same |

MACs do not predict latency. At batch 1 on the GPU, MobileNetV2 and EfficientNet-B0 are no faster than ResNet-50
despite 11 to 14 times fewer MACs, and DenseNet-121 is 2.6 times slower with fewer MACs. Depthwise convolution
reaches 850 GMAC/s against 20,900 for a standard 3x3, which matches its 1.1 MAC per byte. At batch 1 the GPU is
mostly waiting on kernel launches, so batch-1 numbers say more about launch overhead than about arithmetic.
The CPU timings come from a shared machine and are noisy (a 3-repeat trial gave 2.7x for MobileNetV2 where the 10-repeat run gave 2.0x).

Training memory. Saved activations are the tensors autograd keeps for the backward pass, measured with saved-tensor
hooks on CPU at batch 2 and scaled to batch 32 (fp32):

| Network | Weights | Adam m and v | Saved activations at batch 32 | Saved / weights |
|---|---|---|---|---|
| VGG-16 | 0.52 GiB | 1.03 GiB | 2.18 GiB | 4.2x |
| ResNet-50 | 0.10 GiB | 0.19 GiB | 2.57 GiB | 27x |
| DenseNet-121 | 0.03 GiB | 0.06 GiB | 3.90 GiB | 131x |
| MobileNetV2 | 0.01 GiB | 0.03 GiB | 2.33 GiB | 179x |
| ConvNeXt-T | 0.11 GiB | 0.21 GiB | 3.25 GiB | 30x |

Activations dominate the weights for every network except VGG-16. VGG-16's 2.18 GiB of activations against 0.52 GiB
of weights is close to the post's "roughly 1.9 GB against 0.55 GB". The post counts layer outputs;
autograd also keeps ReLU and pooling inputs, so the two are not the same quantity. Peak allocated memory in a real training step (H100, batch 32, fp32, Adam):

| Network | Checkpointing | Peak | Weights + grads + Adam | Step time |
|---|---|---|---|---|
| VGG-16 | off / on (4 segments) | 5.92 / 5.35 GiB | 2.06 GiB | 66 / 81 ms |
| ResNet-50 | off / on (4 segments) | 2.92 / 2.25 GiB | 0.38 GiB | 41 / 145 ms |
| DenseNet-121 | off | 4.04 GiB | 0.12 GiB | 58 ms |
| MobileNetV2 | off | 2.43 GiB | 0.05 GiB | 22 ms |
| ConvNeXt-T | off | 3.65 GiB | 0.43 GiB | 67 ms |

Checkpointing in `checkpoint_sequential` cuts the activations autograd saves for ResNet-50 from 82 MiB per image to
22, 3 and 10 MiB with 2, 4 and 8 segments (CPU, saved bytes after the forward pass). The peak memory saving on the
GPU is smaller (2.92 to 2.25 GiB), because the peak also includes the recomputed segment and the framework's
own buffers. The ResNet-50 step is 3.6 times slower with checkpointing, more than the roughly 1.3 times that
recomputing one forward pass would suggest. This was not investigated, so read that time as unexplained.

## What was not measured

- The one `gpu` test (CUDA output equals CPU output for a small net) was written but not run: the GPU queue was full.
- No ImageNet training. Every top-1 accuracy is the paper's, as quoted in the post. That includes the ConvNeXt
  modernisation ladder (Figure 16): the lab computes the cost of ConvNeXt-T, not the accuracy of each step.
- Which single-axis scaling of EfficientNet is most accurate. The lab computes the cost of each option only.
- Trained first-layer filters (the Gabor-like AlexNet filters), loss-landscape smoothing by skips, and Figure 17's
  image pushed through nine networks with a fixed filter bank were not built.
- The degradation run is 20 epochs, not the paper's 64k iterations (about 164 epochs), with one seed per network.
- Phone NPU latency (the MobileNet claim about hardware it was designed for), int8 or 4-bit quantisation, and
  mixed-precision memory were not measured.
- CPU timings were taken on shared machines and are not benchmarks.
