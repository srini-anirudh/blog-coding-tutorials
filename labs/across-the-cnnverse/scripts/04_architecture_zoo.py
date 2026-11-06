"""Part two of the post: the nine networks, built layer by layer, with parameters and MACs read off the modules.

Writes results/04_architecture_zoo.txt and results/04_architecture_zoo.json. No weights are trained and no
arithmetic is done: profiling runs on the meta device. ImageNet accuracies are the papers' numbers as the post
quotes them; this lab does not reproduce them.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnnverse import zoo
from cnnverse.profile import profile, head_param_share, n_params
from cnnverse.report import RESULTS, Report


def by_resolution(prof):
    """Params and MACs grouped by the spatial size of each layer's output. Classifier layers go under 'head'."""
    agg = {}
    for _, kind, shape, p, m in prof.layers:
        if kind == "Linear" and len(shape) == 1:
            key = "head"
        elif kind == "Linear":
            key = shape[0]        # ConvNeXt's 1x1 convs are Linear layers on (H, W, C)
        else:
            key = shape[-1]
        a = agg.setdefault(key, [0, 0])
        a[0] += p
        a[1] += m
    return agg


def main():
    R = Report("04_architecture_zoo.txt")
    out = {}

    R("\n== Nine networks: computed from the layer specs vs published ==")
    R(f"{'network':>20} {'params':>11} {'published':>10} {'gap':>6} {'GMACs':>8} {'published':>10} {'gap':>6}"
      f" {'acts/img':>9} {'head share':>10} {'top-1 (paper)':>13}")
    for name, (build, inp, pub_p, pub_m, top1) in {**zoo.NINE, **zoo.EXTRAS}.items():
        model = build()
        prof = profile(model, inp, name)
        p_m, g = prof.params / 1e6, prof.macs / 1e9
        gp = f"{(p_m / pub_p - 1) * 100:+.1f}%" if pub_p else "-"
        gm = f"{(g / pub_m - 1) * 100:+.1f}%" if pub_m else "-"
        share = head_param_share(build())
        pub_ps = f"{pub_p:.4g}" if pub_p else "-"
        pub_ms = f"{pub_m:.4g}" if pub_m else "-"
        R(f"{name:>20} {p_m:>10.3f}M {pub_ps:>10} {gp:>6} {g:>8.3f} {pub_ms:>10} {gm:>6}"
          f" {prof.act_elems / 1e6:>8.2f}M {share:>10.1%} {top1 if top1 else '-':>13}")
        out[name] = {"input": inp, "params": prof.params, "macs": prof.macs, "activations_per_image": prof.act_elems,
                     "head_param_share": share, "published_params_M": pub_p, "published_GMACs": pub_m,
                     "paper_top1": top1, "by_resolution": {str(k): v for k, v in by_resolution(prof).items()}}
    R("acts/img counts the outputs of every conv, linear and pooling layer for one image.")

    lenet = out["LeNet-5"]["params"]
    R(f"\nLeNet-5: {lenet:,} parameters in the common form (dense C3);"
      f" {out['LeNet-5, 1998 form']['params']:,} with the 1998 sparse C3 table and trainable subsampling.")
    a, au = out["AlexNet"], out["AlexNet, ungrouped"]
    R(f"AlexNet: grouping conv2, conv4, conv5 in two: {a['params'] / 1e6:.2f} M params and {a['macs'] / 1e9:.3f} GMACs;"
      f" ungrouped {au['params'] / 1e6:.2f} M and {au['macs'] / 1e9:.3f} GMACs.")
    v = zoo.vgg16()
    fc6 = n_params(v[1][1])
    head = sum(n_params(m) for m in v[1] if m.__class__.__name__ == "Linear")
    r50 = out["ResNet-50"]
    R(f"VGG-16: fc6 holds {fc6 / 1e6:.1f} M parameters, the three FC layers {head / 1e6:.1f} M."
      f" VGG-16 / ResNet-50 compute = {out['VGG-16']['macs'] / r50['macs']:.1f}x.")

    R("\n== Where the cost sits: params and MACs by output resolution ==")
    for name in ("VGG-16", "ResNet-50", "ConvNeXt-T"):
        agg = out[name]["by_resolution"]
        tp, tm = sum(x[0] for x in agg.values()), sum(x[1] for x in agg.values())
        R(f"{name}:")
        for k, (p, m) in agg.items():
            label = f"{k}x{k}" if k != "head" else "head"
            R(f"   {label:>8}  params {p / tp:6.1%}   MACs {m / tm:6.1%}")

    R("\n== Inception: the 1x1 bottleneck ==")
    naive5 = 5 * 5 * 480 * 128
    shipped5 = 480 * 32 + 5 * 5 * 32 * 128
    R(f"5x5 branch, 480 -> 128 channels: {naive5:,} weights; with a 1x1 projection to 32 first: {shipped5:,}"
      f" ({naive5 / shipped5:.1f}x fewer)")
    for key in ("3a", "4a"):
        cfg = zoo.GOOGLENET_CFG[key]
        res = 28 if key.startswith("3") else 14
        pn = profile(zoo.Inception(*cfg, reduce=False), (cfg[0], res, res))
        ps = profile(zoo.Inception(*cfg), (cfg[0], res, res))
        c_in, b1, _, b3, _, b5, bp = cfg
        R(f"module {key} at {res}x{res}: naive {pn.params / 1e6:.2f} M params, {pn.macs / 1e6:.0f} M MACs,"
          f" {b1 + b3 + b5 + c_in} out channels; shipped {ps.params / 1e6:.2f} M params, {ps.macs / 1e6:.0f} M MACs,"
          f" {b1 + b3 + b5 + bp} out channels -> {pn.macs / ps.macs:.1f}x fewer MACs")
    R("the naive pool branch passes all input channels through, so naive outputs are wider as well as costlier.")

    R("\n== Residual family ==")
    se, rx = out["SE-ResNet-50"], out["ResNeXt-50 32x4d"]
    R(f"SE on ResNet-50: +{(se['params'] - r50['params']) / 1e6:.2f} M params, +{(se['macs'] / r50['macs'] - 1):.2%} MACs")
    R(f"ResNeXt-50 32x4d vs ResNet-50: params {rx['params'] / r50['params']:.3f}x, MACs {rx['macs'] / r50['macs']:.3f}x")
    R(f"MobileNetV2 vs ResNet-50: {r50['macs'] / out['MobileNetV2']['macs']:.1f}x fewer MACs,"
      f" {r50['params'] / out['MobileNetV2']['params']:.1f}x fewer params")
    R(f"ConvNeXt-T vs ResNet-50: params {out['ConvNeXt-T']['params'] / r50['params']:.2f}x,"
      f" MACs {out['ConvNeXt-T']['macs'] / r50['macs']:.2f}x")
    R(f"DenseNet-121 vs ResNet-50: {r50['params'] / out['DenseNet-121']['params']:.1f}x fewer params")

    R("\n== EfficientNet compound scaling ==")
    al, be, ga = 1.2, 1.1, 1.15
    R(f"alpha * beta^2 * gamma^2 = {al * be ** 2 * ga ** 2:.3f} (target 2)")
    base = profile(zoo.efficientnet(), (3, 224, 224)).macs
    R(f"{'phi':>3} {'depth':>6} {'width':>6} {'res':>4} {'GMACs':>7} {'ratio to B0':>12} {'(a b^2 g^2)^phi':>16}")
    for phi in range(0, 5):
        d, w, r = al ** phi, be ** phi, int(round(224 * ga ** phi))
        m = profile(zoo.efficientnet(w, d), (3, r, r)).macs
        R(f"{phi:>3} {d:>6.2f} {w:>6.2f} {r:>4} {m / 1e9:>7.2f} {m / base:>12.2f} {(al * be ** 2 * ga ** 2) ** phi:>16.2f}")
    blocks = lambda d: sum(math.ceil(n * d) for *_, n in zoo.EFFNET_B0)
    md = profile(zoo.efficientnet(1.0, al), (3, 224, 224)).macs
    mw = profile(zoo.efficientnet(be, 1.0), (3, 224, 224)).macs
    mr = profile(zoo.efficientnet(), (3, 258, 258)).macs
    R(f"one axis at a time, phi=1: depth 1.2 -> {md / base:.2f}x (blocks {blocks(1.0)} -> {blocks(al)}),"
      f" width 1.1 -> {mw / base:.2f}x (formula 1.21), resolution 258 -> {mr / base:.2f}x (formula 1.32)")
    sizes = [258]
    for _ in range(5):
        sizes.append((sizes[-1] - 1) // 2 + 1)
    R(f"depth drifts because repeats round up to whole blocks and B0's stages have 1 to 4 blocks. Resolution drifts"
      f" because every stride-2 layer rounds up: {' -> '.join(map(str, sizes))}, so the last map is"
      f" {sizes[-1]}x{sizes[-1]}, not {258 / 32:.1f}x{258 / 32:.1f}. Width rounds to multiples of 8 and drifts down.")
    R("\nReleased members, computed from their (width, depth, resolution):")
    R(f"{'':>3} {'width':>6} {'depth':>6} {'res':>4} {'params':>8} {'GMACs':>7}")
    for k, (w, d, r) in zoo.EFFNET_MEMBERS.items():
        p = profile(zoo.efficientnet(w, d), (3, r, r))
        R(f"{k:>3} {w:>6} {d:>6} {r:>4} {p.params / 1e6:>7.2f}M {p.macs / 1e9:>7.2f}")
        out[f"EfficientNet-{k}"] = {"params": p.params, "macs": p.macs, "width": w, "depth": d, "res": r}
    R("published: B0 5.3 M / 0.39 G, B4 19 M / 4.2 G, B7 66 M / 37 G (post's figures).")
    R("\nScaling one axis alone to about 2x B0's compute, with the compound phi=1 step for comparison:")
    for label, (w, d, r) in {"depth x2": (1.0, 2.0, 224), "width x1.41": (math.sqrt(2), 1.0, 224),
                             "resolution x1.41": (1.0, 1.0, 317), "compound phi=1": (1.1, 1.2, 258)}.items():
        p = profile(zoo.efficientnet(w, d), (3, r, r))
        R(f"   {label:>17}: {p.macs / base:.2f}x MACs, {p.params / 1e6:.2f} M params")
    R("which of these is most accurate needs ImageNet training; the lab does not measure it.")

    (RESULTS / "04_architecture_zoo.json").write_text(json.dumps(out, indent=1))
    R.close()


if __name__ == "__main__":
    main()
