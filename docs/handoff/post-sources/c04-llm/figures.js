(function () {
  "use strict";
  const { svg, lerp, stepper } = window.CT;
  const C = { ink: "#14171a", soft: "#474e53", muted: "#7b8388", rule: "#cac6bd", paper: "#f7f6f3", hi: "#fbfaf8", blue: "#0b5ea8", orange: "#d66f00", green: "#2f8f5b", red: "#b8432f", violet: "#7a4fa3", grid: "#e4e1da" };
  const DATA = /*DATA*/null;
  const $ = (id) => document.getElementById(id);
  const fmtN = (n) => n >= 1e9 ? (n / 1e9).toFixed(3) + "B" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n.toLocaleString("en-US");

  /* ---------- Figure 1: one token through one block, with shapes ---------- */
  (function flowFigure() {
    const fig = $("fig-flow");
    if (!fig) return;
    const VW = 960, VH = 430;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "One token flowing through a decoder block, with the shape of every tensor" }, fig.querySelector(".ct-figure__stage"));
    const world = svg("g", null, root);
    const T = 11, CELL = 11;
    const label = (g, x, y, text, o) => svg("text", Object.assign({ x, y, "text-anchor": "middle", "font-size": 15, fill: C.ink, text }, o || {}), g);
    const shape = (g, x, y, text) => label(g, x, y, text, { "font-size": 15, fill: C.orange, "font-weight": 600 });
    // A tensor drawn as T rows. Row 10 is the token we follow, always orange.
    function block(g, x, y, cols, color, depth) {
      for (let d = (depth || 1) - 1; d > 0; d--) {
        svg("rect", { x: x + d * 7, y: y - d * 7, width: cols * CELL - 1, height: T * CELL - 1, rx: 2, fill: C.hi, stroke: color, "stroke-opacity": 0.55 }, g);
      }
      let seed = x * 13 + y * 7 + cols;
      const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      for (let r = 0; r < T; r++) for (let c = 0; c < cols; c++) {
        const tracked = r === T - 1;
        svg("rect", { x: x + c * CELL, y: y + r * CELL, width: CELL - 1.5, height: CELL - 1.5, rx: 1, fill: tracked ? C.orange : color, "fill-opacity": (tracked ? 0.55 : 0.16) + 0.4 * rand() }, g);
      }
    }
    const arrow = (g, x1, x2, y) => svg("path", { d: `M${x1} ${y} L${x2} ${y}`, stroke: C.muted, "stroke-width": 1.6, fill: "none", "marker-end": "url(#flow-arr)" }, g);
    const defs = svg("defs", null, root);
    const m = svg("marker", { id: "flow-arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
    svg("path", { d: "M0 0 L10 5 L0 10 z", fill: C.muted }, m);

    const layers = [];
    const layer = () => { const g = svg("g", { opacity: 0 }, world); layers.push(g); return g; };
    const Y = 140; // top of tensor rows

    // 0: ids
    let g = layer();
    label(g, 130, 76, "input_ids", { class: "disp", "font-size": 22 });
    "The cat sat".split("").forEach((ch, i) => {
      const tracked = i === T - 1;
      svg("rect", { x: 112, y: Y + i * CELL * 1.06 - 2, width: 38, height: CELL, rx: 1.5, fill: tracked ? C.orange : C.hi, "fill-opacity": tracked ? 0.75 : 1, stroke: C.rule }, g);
      svg("text", { x: 131, y: Y + i * CELL * 1.06 + 7, "text-anchor": "middle", "font-size": 9.5, fill: C.ink, text: ch === " " ? "32" : String(ch.charCodeAt(0)) }, g);
    });
    shape(g, 130, Y + 158, "(1, 11)");
    label(g, 130, Y + 180, "one byte, one token", { "font-size": 13, fill: C.muted });
    // 1: embeddings
    g = layer();
    arrow(g, 170, 268, Y + 60);
    label(g, 370, 76, "embed_tokens", { class: "disp", "font-size": 22 });
    block(g, 280, Y, 16, C.ink);
    shape(g, 370, Y + 158, "(1, 11, 256)");
    label(g, 370, Y + 180, "rms 0.020: very small", { "font-size": 13, fill: C.muted });
    // 2: RMSNorm
    g = layer();
    arrow(g, 470, 568, Y + 60);
    label(g, 670, 76, "input_layernorm", { class: "disp", "font-size": 22 });
    block(g, 580, Y, 16, C.blue);
    shape(g, 670, Y + 158, "(1, 11, 256)");
    label(g, 670, Y + 180, "rms 0.988: rescaled, same shape", { "font-size": 13, fill: C.muted });
    // 3: q k v
    g = layer();
    arrow(g, 770, 868, Y + 60);
    label(g, 928, 76, "q", { class: "disp", "font-size": 22, fill: C.blue });
    block(g, 880, Y, 8, C.blue, 8);
    shape(g, 950, Y + 158, "(1, 8, 11, 32)");
    label(g, 1098, 76, "k", { class: "disp", "font-size": 22, fill: C.violet });
    block(g, 1050, Y, 8, C.violet, 2);
    shape(g, 1098, Y + 158, "(1, 2, 11, 32)");
    label(g, 1238, 76, "v", { class: "disp", "font-size": 22, fill: C.green });
    block(g, 1190, Y, 8, C.green, 2);
    shape(g, 1238, Y + 158, "(1, 2, 11, 32)");
    label(g, 1070, Y + 184, "eight query heads, only two key and value heads", { "font-size": 13, fill: C.muted });
    // 4: RoPE
    g = layer();
    label(g, 1070, 46, "RoPE rotates q and k, by position. v is left alone.", { "font-size": 14, fill: C.soft });
    [[948, C.blue], [1098, C.violet]].forEach(([x, col]) => {
      svg("circle", { cx: x, cy: Y - 26, r: 14, fill: "none", stroke: col, "stroke-width": 1.4 }, g);
      svg("path", { d: `M${x} ${Y - 26} L${x + 10} ${Y - 36}`, stroke: col, "stroke-width": 2 }, g);
    });
    // 5: attention scores with mask
    g = layer();
    arrow(g, 1300, 1398, Y + 60);
    label(g, 1490, 76, "softmax(q kᵀ / √32)", { class: "disp", "font-size": 19 });
    const SZ = 11;
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const allowed = j <= i, tracked = i === T - 1;
      svg("rect", { x: 1430 + j * SZ, y: Y + i * SZ, width: SZ - 1.5, height: SZ - 1.5, rx: 1, fill: !allowed ? C.hi : tracked ? C.orange : C.blue, stroke: allowed ? "none" : C.grid, "fill-opacity": allowed ? 0.3 + 0.5 * (((i * 7 + j * 3) % 5) / 5) : 1 }, g);
    }
    shape(g, 1490, Y + 158, "(1, 8, 11, 11)");
    label(g, 1490, Y + 180, "the last token sees all 11", { "font-size": 13, fill: C.muted });
    // 6: merge + o_proj
    g = layer();
    arrow(g, 1570, 1668, Y + 60);
    label(g, 1770, 76, "o_proj(merge_heads)", { class: "disp", "font-size": 19 });
    block(g, 1680, Y, 16, C.blue);
    shape(g, 1770, Y + 158, "(1, 11, 256)");
    label(g, 1770, Y + 180, "back to the input shape", { "font-size": 13, fill: C.muted });
    // 7: first residual
    g = layer();
    arrow(g, 1870, 1930, Y + 60);
    svg("circle", { cx: 1952, cy: Y + 60, r: 18, fill: C.hi, stroke: C.ink }, g);
    label(g, 1952, Y + 66, "+", { "font-size": 22 });
    label(g, 1952, 76, "x + attention", { class: "disp", "font-size": 19 });
    shape(g, 1952, Y + 158, "(1, 11, 256)");
    label(g, 1952, Y + 180, "the residual stream is not normalised", { "font-size": 13, fill: C.muted });
    // 8: SwiGLU expansion
    g = layer();
    arrow(g, 1975, 2038, Y + 60);
    label(g, 2130, 66, "silu(gate) × up", { class: "disp", "font-size": 19 });
    block(g, 2050, Y, 18, C.green);
    shape(g, 2150, Y + 158, "(1, 11, 704)");
    label(g, 2150, Y + 180, "gate and up widen 256 to 704, then multiply", { "font-size": 13, fill: C.muted });
    // 9: down + residual
    g = layer();
    arrow(g, 2256, 2318, Y + 60);
    label(g, 2420, 76, "down_proj, then +", { class: "disp", "font-size": 19 });
    block(g, 2330, Y, 16, C.ink);
    shape(g, 2420, Y + 158, "(1, 11, 256)");
    label(g, 2420, Y + 180, "block output has the block input's shape", { "font-size": 13, fill: C.muted });
    // 10: stack and head
    g = layer();
    arrow(g, 2520, 2588, Y + 60);
    label(g, 2650, 66, "× 6 blocks", { class: "disp", "font-size": 19 });
    for (let d = 5; d >= 0; d--) svg("rect", { x: 2600 + d * 9, y: Y - d * 7 + 20, width: 92, height: 92, rx: 3, fill: d === 0 ? C.hi : C.paper, stroke: C.soft }, g);
    label(g, 2646, Y + 72, "block", { "font-size": 13, fill: C.soft });
    shape(g, 2650, Y + 158, "(1, 11, 256)");
    // 11: logits
    g = layer();
    arrow(g, 2720, 2788, Y + 60);
    label(g, 2895, 66, "norm, then lm_head", { class: "disp", "font-size": 19 });
    block(g, 2800, Y, 18, C.violet);
    shape(g, 2895, Y + 158, "(1, 11, 256)");
    label(g, 2895, Y + 180, "orange row: scores for the byte after 't'", { "font-size": 13, fill: C.muted });

    const cams = [{ x: 330, w: 680 }, { x: 330, w: 680 }, { x: 520, w: 760 }, { x: 1040, w: 780 }, { x: 1040, w: 780 }, { x: 1400, w: 700 }, { x: 1680, w: 700 }, { x: 1880, w: 700 }, { x: 2130, w: 700 }, { x: 2400, w: 700 }, { x: 2620, w: 640 }, { x: 2800, w: 680 }];
    const frames = [
      { note: "We follow the last token, the byte <code>t</code> at position 10. It starts as one integer in a (1, 11) tensor. Orange marks its row in every tensor below." },
      { note: "<code>embed_tokens</code> looks the id up in a (256, 256) matrix. The row is now 256 numbers with an RMS of 0.020." },
      { note: "<code>input_layernorm</code> rescales each row to RMS near 1. Real run: 0.988. Epsilon explains the gap from 1, as step 3 shows." },
      { note: "Three projections make queries, keys and values. Queries get 8 heads of 32 numbers. Keys and values get only 2 heads. This is the GQA saving." },
      { note: "RoPE rotates each channel pair of q and k by an angle set by the position. Position 10 turns further than position 0." },
      { note: "Each query scores the keys at and before it. The mask blanks the future. Our token is last, so its row is full: 11 weights summing to 1." },
      { note: "Weights average the values, heads merge back to 256 numbers, and <code>o_proj</code> mixes them. The shape equals the input again." },
      { note: "The branch is added to the stream. At initialisation it is small: the branch RMS is 0.014 against a stream of 0.020." },
      { note: "A second norm, then SwiGLU. <code>gate_proj</code> and <code>up_proj</code> widen each token to 704 numbers. The silu gate multiplies the up values channel by channel." },
      { note: "<code>down_proj</code> returns to 256 and the branch is added. A block maps (1, 11, 256) to (1, 11, 256), which is why blocks stack." },
      { note: "Five more identical blocks run. Only the layer index differs, and that only picks the cache slot." },
      { note: "A final norm, then <code>lm_head</code> scores every byte for every position. We keep only the orange row to pick the next token." },
    ];
    stepper(fig, {
      frames, tween: 1000, hold: 2100,
      render(i, t, from) {
        const a = cams[from], b = cams[i];
        const w = lerp(a.w, b.w, t), h = (w * VH) / VW;
        root.setAttribute("viewBox", `${lerp(a.x, b.x, t) - w / 2} ${215 - h / 2} ${w} ${h}`);
        layers.forEach((L, k) => L.setAttribute("opacity", lerp(k <= from ? 1 : 0, k <= i ? 1 : 0, t).toFixed(3)));
      },
    });
  })();

  /* ---------- Figure 2: RMSNorm, input scale vs output scale ---------- */
  (function rmsFigure() {
    const fig = $("fig-rms");
    if (!fig) return;
    const W = 960, H = 280, N = 32;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "One token vector before and after RMSNorm" }, fig.querySelector(".ct-figure__stage"));
    let seed = 11;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const base = Array.from({ length: N }, () => { const u = rand(), v = rand(); return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(6.283 * v); });
    const scaleB = Math.sqrt(base.reduce((s, x) => s + x * x, 0) / N);
    const unit = base.map((x) => x / scaleB);
    const axes = [{ x: 30, title: "input x" }, { x: 500, title: "RMSNorm(x)" }];
    const bars = axes.map((ax) => {
      svg("text", { x: ax.x, y: 24, "font-size": 15, fill: C.ink, class: "disp", text: ax.title }, root);
      svg("line", { x1: ax.x, x2: ax.x + 430, y1: 150, y2: 150, stroke: C.rule }, root);
      return { lab: svg("text", { x: ax.x + 430, y: 24, "text-anchor": "end", "font-size": 13, fill: C.soft }, root), rects: unit.map((_, i) => svg("rect", { x: ax.x + i * 13.4, width: 10.5, rx: 2, fill: ax.x < 100 ? C.ink : C.blue }, root)) };
    });
    const inp = $("rms-scale"), eps = $("rms-eps");
    function draw() {
      const rmsIn = Math.pow(10, +inp.value / 10), e = Math.pow(10, +eps.value);
      const rmsOut = rmsIn / Math.sqrt(rmsIn * rmsIn + e);
      $("rms-scale-out").textContent = rmsIn >= 1 ? rmsIn.toFixed(1) : rmsIn.toFixed(4);
      $("rms-eps-out").textContent = "1e" + eps.value;
      // The input panel autoscales, so its axis label carries the size.
      const inPeak = Math.max(...unit.map(Math.abs)) * rmsIn;
      bars[0].lab.textContent = "tallest bar = " + (inPeak >= 1 ? inPeak.toFixed(1) : inPeak.toFixed(4));
      bars[1].lab.textContent = "fixed axis: ±3";
      unit.forEach((u, i) => {
        const h0 = (u / Math.max(...unit.map(Math.abs))) * 110;
        bars[0].rects[i].setAttribute("y", h0 > 0 ? 150 - h0 : 150); bars[0].rects[i].setAttribute("height", Math.abs(h0));
        const h1 = ((u * rmsOut) / 3) * 110;
        bars[1].rects[i].setAttribute("y", h1 > 0 ? 150 - h1 : 150); bars[1].rects[i].setAttribute("height", Math.abs(h1));
      });
      $("rms-in").textContent = rmsIn >= 1 ? rmsIn.toFixed(2) : rmsIn.toFixed(4);
      $("rms-out").textContent = rmsOut.toFixed(4);
      $("rms-ratio").textContent = (100 * (1 - rmsOut)).toFixed(1) + "%";
    }
    [inp, eps].forEach((n) => n.addEventListener("input", draw));
    draw();
  })();

  /* ---------- Figure 3: RoPE clocks ---------- */
  (function ropeFigure() {
    const fig = $("fig-rope");
    if (!fig) return;
    const W = 960, H = 330;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Two channel pairs drawn as clocks. Query and key arrows rotate with position." }, fig.querySelector(".ct-figure__stage"));
    const freqs = [{ f: 1.0, cx: 250, name: "channel pair 0", sub: "speed 1.000, wavelength 6.3 tokens" }, { f: Math.pow(10000, -6 / 32), cx: 710, name: "channel pair 3", sub: "speed 0.178, wavelength 35 tokens" }];
    const R = 96, CY = 168;
    const parts = freqs.map((c) => {
      svg("circle", { cx: c.cx, cy: CY, r: R, fill: C.hi, stroke: C.rule }, root);
      svg("text", { x: c.cx, y: 30, "text-anchor": "middle", "font-size": 15, class: "disp", fill: C.ink, text: c.name }, root);
      svg("text", { x: c.cx, y: 49, "text-anchor": "middle", "font-size": 12, fill: C.muted, text: c.sub }, root);
      const arc = svg("path", { fill: C.orange, "fill-opacity": 0.2, stroke: "none" }, root);
      const q = svg("line", { x1: c.cx, y1: CY, stroke: C.blue, "stroke-width": 3, "stroke-linecap": "round" }, root);
      const k = svg("line", { x1: c.cx, y1: CY, stroke: C.violet, "stroke-width": 3, "stroke-linecap": "round" }, root);
      const qt = svg("text", { "font-size": 14, fill: C.blue, "font-weight": 600, "text-anchor": "middle", text: "q" }, root);
      const kt = svg("text", { "font-size": 14, fill: C.violet, "font-weight": 600, "text-anchor": "middle", text: "k" }, root);
      const lab = svg("text", { x: c.cx, y: CY + R + 28, "text-anchor": "middle", "font-size": 14, fill: C.soft }, root);
      return { c, arc, q, k, qt, kt, lab };
    });
    const pos = svg("text", { x: W / 2, y: H - 8, "text-anchor": "middle", "font-size": 15, fill: C.ink }, root);
    const Q0 = 0.7, K0 = 2.5;
    const frames = DATA.rope.map((r) => ({ m: r.m, n: r.n, note: r.note }));
    stepper(fig, {
      frames, tween: 1300, hold: 2200,
      render(i, t, from) {
        const a = frames[from], b = frames[i];
        const m = lerp(a.m, b.m, t), n = lerp(a.n, b.n, t);
        parts.forEach((p) => {
          const aq = Q0 + m * p.c.f, ak = K0 + n * p.c.f;
          const px = (ang, r) => [p.c.cx + r * Math.cos(ang), CY - r * Math.sin(ang)];
          const [qx, qy] = px(aq, R - 8), [kx, ky] = px(ak, R - 8);
          p.q.setAttribute("x2", qx); p.q.setAttribute("y2", qy);
          p.k.setAttribute("x2", kx); p.k.setAttribute("y2", ky);
          const [qtx, qty] = px(aq, R + 14), [ktx, kty] = px(ak, R + 14);
          p.qt.setAttribute("x", qtx); p.qt.setAttribute("y", qty + 5);
          p.kt.setAttribute("x", ktx); p.kt.setAttribute("y", kty + 5);
          // Signed gap between the arrows, folded to (-pi, pi].
          let gap = ((aq - ak) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
          const steps = 24, rr = 34, pts = [`M${p.c.cx} ${CY}`];
          for (let s = 0; s <= steps; s++) { const [x, y] = px(ak + (gap * s) / steps, rr); pts.push(`L${x.toFixed(1)} ${y.toFixed(1)}`); }
          p.arc.setAttribute("d", pts.join(" ") + " Z");
          p.lab.textContent = "angle between q and k: " + (Math.abs(gap) * 57.2958).toFixed(0) + "°";
        });
        pos.textContent = `q at position ${Math.round(m)}, k at position ${Math.round(n)}, offset ${Math.round(m) - Math.round(n)}`;
      },
    });
  })();

  /* ---------- Figure 4: GQA head grouping ---------- */
  (function gqaFigure() {
    const fig = $("fig-gqa");
    if (!fig) return;
    const W = 960, H = 310, Hq = 8;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Eight query heads share key and value heads" }, fig.querySelector(".ct-figure__stage"));
    const qx = (h) => 90 + h * 112;
    svg("text", { x: 20, y: 56, "font-size": 15, fill: C.blue, "font-weight": 600, text: "q" }, root);
    svg("text", { x: 20, y: 246, "font-size": 15, fill: C.violet, "font-weight": 600, text: "k, v" }, root);
    const lines = Array.from({ length: Hq }, () => svg("path", { fill: "none", stroke: C.muted, "stroke-width": 1.5 }, root));
    for (let h = 0; h < Hq; h++) {
      svg("rect", { x: qx(h) - 40, y: 34, width: 80, height: 40, rx: 3, fill: C.blue, "fill-opacity": 0.2, stroke: C.blue }, root);
      svg("text", { x: qx(h), y: 59, "text-anchor": "middle", "font-size": 13, fill: C.ink, text: "head " + h }, root);
    }
    const kv = Array.from({ length: Hq }, (_, h) => ({ r: svg("rect", { y: 220, height: 40, rx: 3, fill: C.violet, "fill-opacity": 0.2, stroke: C.violet }, root), t: svg("text", { y: 245, "text-anchor": "middle", "font-size": 13, fill: C.ink, text: "kv " + h }, root) }));
    const read = svg("text", { x: W / 2, y: 294, "text-anchor": "middle", "font-size": 14, fill: C.soft }, root);
    const stages = [
      { kv: 8, name: "MHA", note: "<strong>MHA, 8 KV heads.</strong> Every query head owns a key and value head. Cache: 12,288 bytes per token in float32." },
      { kv: 4, name: "GQA", note: "<strong>GQA, 4 KV heads.</strong> Pairs of query heads share one. The cache halves to 6,144 bytes per token." },
      { kv: 2, name: "GQA", note: "<strong>GQA, 2 KV heads.</strong> The tinyllm default. Four query heads read each KV head through <code>h // 4</code>. Cache: 3,072 bytes per token." },
      { kv: 1, name: "MQA", note: "<strong>MQA, 1 KV head.</strong> All queries read one key and value head. Cache: 1,536 bytes per token. Query heads still differ, because <code>q_proj</code> stays full size." },
    ];
    const layout = (n) => Array.from({ length: Hq }, (_, h) => { const g = Math.min(n - 1, h); const span = Hq / n; return { x: 90 + (g + 0.5) * span * 112 - 56, w: Math.max(80, span * 112 - 32) }; });
    const parentOf = (h, n) => Math.floor(h / (Hq / n));
    const opacity = (g, n) => (g < n ? 1 : 0);
    stepper(fig, {
      frames: stages, tween: 1100, hold: 2300,
      render(i, t, from) {
        const a = stages[from].kv, b = stages[i].kv;
        const la = layout(a), lb = layout(b);
        kv.forEach((k, g) => {
          const x = lerp(la[g].x, lb[g].x, t), w = lerp(la[g].w, lb[g].w, t), o = lerp(opacity(g, a), opacity(g, b), t);
          k.r.setAttribute("x", x - w / 2); k.r.setAttribute("width", w); k.r.setAttribute("opacity", o);
          k.t.setAttribute("x", x); k.t.setAttribute("opacity", o);
        });
        lines.forEach((ln, h) => {
          const xa = la[parentOf(h, a)].x, xb = lb[parentOf(h, b)].x, x = lerp(xa, xb, t);
          ln.setAttribute("d", `M${qx(h)} 74 C ${qx(h)} 140, ${x} 150, ${x} 220`);
        });
        const n = t < 0.5 ? a : b;
        read.textContent = `k_proj and v_proj: (${n * 32}, 256) each. Cached keys: (B, ${n}, T, 32).`;
      },
    });
  })();

  /* ---------- Figure 5: parameter-count calculator ---------- */
  (function paramFigure() {
    const fig = $("fig-params");
    if (!fig) return;
    const VS = [256, 1024, 32000, 49152, 128256];
    const HS = [1, 2, 4, 8, 9, 12, 16, 32];
    const PRE = { tiny: { v: 256, d: 256, l: 6, h: 8, kv: 2, f: 704, tied: true }, smol: { v: 49152, d: 576, l: 30, h: 9, kv: 3, f: 1536, tied: true }, llama: { v: 128256, d: 4096, l: 32, h: 32, kv: 8, f: 14336, tied: false } };
    const COL = { embeddings: C.blue, attention: C.orange, mlp: C.green, norms: C.muted, lm_head: C.violet };
    const NAME = { embeddings: "Embeddings", attention: "Attention", mlp: "SwiGLU", norms: "RMSNorm", lm_head: "LM head" };
    const W = 960, H = 74;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Stacked bar of parameters by component" }, $("params-stage"));
    const tip = svg("text", { x: 0, y: 66, "font-size": 13, fill: C.soft }, root);
    const segs = Object.keys(COL).map((k) => ({ k, r: svg("rect", { y: 10, height: 24, fill: COL[k] }, root) }));
    const el = { v: $("p-v"), d: $("p-d"), l: $("p-l"), h: $("p-h"), kv: $("p-kv"), f: $("p-f"), tied: $("p-tied") };
    let last = null;
    function divisors(h) { const o = []; for (let i = 1; i <= h; i++) if (h % i === 0) o.push(i); return o; }
    function read() {
      const h = HS[+el.h.value], ds = divisors(h);
      el.kv.max = ds.length - 1;
      if (+el.kv.value > ds.length - 1) el.kv.value = ds.length - 1;
      return { v: VS[+el.v.value], d: +el.d.value, l: +el.l.value, h, kv: ds[+el.kv.value], f: +el.f.value, tied: el.tied.checked };
    }
    function draw() {
      const c = read();
      $("p-v-out").textContent = c.v.toLocaleString("en-US");
      $("p-d-out").textContent = c.d;
      $("p-l-out").textContent = c.l;
      $("p-h-out").textContent = c.h;
      $("p-kv-out").textContent = c.kv;
      $("p-f-out").textContent = c.f.toLocaleString("en-US");
      const bad = c.d % c.h !== 0 ? `ValueError: hidden_size=${c.d} is not divisible by num_heads=${c.h}` : (c.d / c.h) % 2 !== 0 ? "ValueError: RoPE rotates pairs of channels, so head_dim must be even" : "";
      $("p-error").textContent = bad;
      $("p-error").hidden = !bad;
      if (bad) { segs.forEach((s) => s.r.setAttribute("width", 0)); ["p-total", "p-kvmem", "p-hd"].forEach((id) => { $(id).textContent = "n/a"; }); tip.textContent = ""; return; }
      const hd = c.d / c.h;
      const counts = { embeddings: c.v * c.d, attention: c.l * (c.d * c.h * hd + 2 * c.d * c.kv * hd + c.h * hd * c.d), mlp: c.l * 3 * c.d * c.f, norms: c.l * 2 * c.d + c.d, lm_head: c.tied ? 0 : c.v * c.d };
      const total = Object.values(counts).reduce((a, b) => a + b, 0);
      let x = 0;
      const gap = 2, usable = W - gap * 4;
      segs.forEach((s) => {
        const w = (counts[s.k] / total) * usable;
        s.r.setAttribute("x", x); s.r.setAttribute("width", Math.max(0, w));
        x += w + (w > 0 ? gap : 0);
      });
      last = { counts, total };
      $("p-legend").innerHTML = Object.keys(COL).map((k) => `<span class="legend-item"><i style="background:${COL[k]}"></i>${NAME[k]} <b>${fmtN(counts[k])}</b> <em>${((100 * counts[k]) / total).toFixed(1)}%</em></span>`).join("");
      $("p-total").textContent = fmtN(total);
      $("p-hd").textContent = hd;
      const gib = (2 * c.l * c.kv * hd * 4096 * 2) / 2 ** 30;
      $("p-kvmem").textContent = gib >= 0.1 ? gib.toFixed(2) + " GiB" : ((gib * 1024).toFixed(1) + " MiB");
      tip.textContent = "Hover or tap a segment for its exact count.";
    }
    segs.forEach((s) => {
      const show = () => { if (last) tip.textContent = `${NAME[s.k]}: ${last.counts[s.k].toLocaleString("en-US")} of ${last.total.toLocaleString("en-US")}`; };
      s.r.addEventListener("pointerenter", show);
      s.r.addEventListener("pointerdown", show);
    });
    Object.values(el).forEach((n) => n.addEventListener("input", draw));
    fig.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const p = PRE[b.dataset.preset];
      el.v.value = VS.indexOf(p.v); el.d.value = p.d; el.l.value = p.l; el.h.value = HS.indexOf(p.h);
      el.kv.max = divisors(p.h).length - 1;
      el.kv.value = divisors(p.h).indexOf(p.kv); el.f.value = p.f; el.tied.checked = p.tied;
      fig.querySelectorAll("[data-preset]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      draw();
    }));
    draw();
  })();

  /* ---------- Figure 6: sampling filters on real logits ---------- */
  (function samplingFigure() {
    const fig = $("fig-sampling");
    if (!fig) return;
    const logits = DATA.sampling.logits, N = 14;
    const label = (i) => { if (i === 32) return "space"; if (i === 10) return "\\n"; return String.fromCharCode(i); };
    const order = logits.map((_, i) => i).sort((a, b) => logits[b] - logits[a]);
    const W = 960, rowH = 22, H = N * rowH + 30;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Next-byte probabilities after temperature, top-k and top-p" }, $("sampling-stage"));
    const X0 = 96, BW = W - X0 - 90;
    svg("line", { x1: X0, x2: X0, y1: 4, y2: N * rowH + 4, stroke: C.rule }, root);
    const rows = order.slice(0, N).map((id, r) => {
      const y = 6 + r * rowH;
      svg("text", { x: X0 - 10, y: y + 13, "text-anchor": "end", "font-size": 13, fill: C.ink, text: label(id) }, root);
      return { id, bar: svg("rect", { x: X0, y, height: 15, rx: 3, fill: C.blue }, root), val: svg("text", { y: y + 13, "font-size": 12.5, fill: C.soft }, root) };
    });
    const rest = svg("text", { x: X0, y: H - 6, "font-size": 12.5, fill: C.muted }, root);
    const tIn = $("s-t"), kIn = $("s-k"), pIn = $("s-p");
    function draw() {
      const T = +tIn.value / 100, k = +kIn.value, p = +pIn.value / 100;
      $("s-t-out").textContent = T.toFixed(2); $("s-k-out").textContent = k === 256 ? "off" : k; $("s-p-out").textContent = p >= 1 ? "off" : p.toFixed(2);
      // Same order as sample_next_token: temperature, then top-k, then top-p.
      const scaled = logits.map((x) => x / T);
      const idx = logits.map((_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
      const keep = new Set(idx.slice(0, k));
      const sumK = idx.slice(0, k).reduce((s, i) => s + Math.exp(scaled[i] - scaled[idx[0]]), 0);
      let before = 0;
      if (p < 1) idx.slice(0, k).forEach((i) => { const q = Math.exp(scaled[i] - scaled[idx[0]]) / sumK; if (before >= p) keep.delete(i); before += q; });
      const sumAll = [...keep].reduce((s, i) => s + Math.exp(scaled[i] - scaled[idx[0]]), 0);
      const prob = (i) => keep.has(i) ? Math.exp(scaled[i] - scaled[idx[0]]) / sumAll : 0;
      let entropy = 0;
      keep.forEach((i) => { const q = prob(i); if (q > 0) entropy -= q * Math.log(q); });
      rows.forEach((r) => {
        const q = prob(r.id);
        r.bar.setAttribute("width", q * BW);
        r.bar.setAttribute("fill", keep.has(r.id) ? C.blue : C.rule);
        r.val.setAttribute("x", X0 + q * BW + 6);
        r.val.textContent = keep.has(r.id) ? q.toFixed(3) : "filtered";
      });
      let shown = 0; rows.forEach((r) => { shown += prob(r.id); });
      rest.textContent = `The ${256 - N} other bytes hold ${(100 * (1 - shown)).toFixed(2)}% of the mass.`;
      $("s-kept").textContent = keep.size;
      $("s-top").textContent = prob(idx[0]).toFixed(3);
      $("s-ent").textContent = entropy.toFixed(2) + " nats";
    }
    [tIn, kIn, pIn].forEach((n) => n.addEventListener("input", draw));
    draw();
  })();

  /* ---------- Figure 7: uncached vs cached decoding ---------- */
  (function cacheFigure() {
    const fig = $("fig-cache");
    if (!fig) return;
    const W = 960, H = 340, P = 5, STEPS = 8, MAXT = P + STEPS, S = 38;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Uncached decoding recomputes every token. Cached decoding computes one token and grows the key tensor." }, fig.querySelector(".ct-figure__stage"));
    const rowsY = { off: 62, on: 190 };
    svg("text", { x: 20, y: 34, "font-size": 15, class: "disp", fill: C.ink, text: "No cache: re-run the whole sequence" }, root);
    svg("text", { x: 20, y: 162, "font-size": 15, class: "disp", fill: C.ink, text: "Cache: run one token, reuse the rest" }, root);
    const mk = (y) => Array.from({ length: MAXT }, (_, i) => svg("rect", { x: 20 + i * (S + 4), y, width: S, height: S, rx: 3 }, root));
    const off = mk(rowsY.off), on = mk(rowsY.on);
    const offWork = svg("text", { x: 20, y: 124, "font-size": 13, fill: C.soft }, root);
    const onWork = svg("text", { x: 20, y: 252, "font-size": 13, fill: C.soft }, root);
    svg("text", { x: 20, y: 286, "font-size": 13, fill: C.muted, text: "cached keys, one layer" }, root);
    const kBar = Array.from({ length: MAXT }, (_, i) => svg("rect", { x: 20 + i * (S + 4), y: 296, width: S, height: 16, rx: 2, fill: C.violet }, root));
    const kShape = svg("text", { x: W - 20, y: 286, "text-anchor": "end", "font-size": 14, "font-weight": 600, fill: C.orange }, root);
    const frames = [{ note: "<strong>Prefill.</strong> Both paths run the 5-token prompt once. The cache now holds 5 keys and 5 values in every layer." }];
    for (let s = 1; s <= STEPS; s++) frames.push({ note: `<strong>Decode step ${s}.</strong> Without a cache the model re-runs ${P + s} tokens to get one new row of logits. With a cache it runs 1 token, against ${P + s} keys.` });
    const sum = (s) => { let t = 0; for (let i = 0; i <= s; i++) t += P + i; return t; };
    stepper(fig, {
      frames, tween: 800, hold: 1500,
      render(i, t, from) {
        const grow = from <= i ? t : t; // the newest cell fades in on every transition
        for (let c = 0; c < MAXT; c++) {
          const present = c < P + i;
          const newest = c === P + i - 1 && i > 0;
          const o = !present ? 0 : newest ? grow : 1;
          off[c].setAttribute("opacity", o);
          off[c].setAttribute("fill", C.blue);
          off[c].setAttribute("fill-opacity", present ? 0.6 : 0);
          on[c].setAttribute("opacity", o);
          const live = i === 0 ? c < P : c === P + i - 1;
          on[c].setAttribute("fill", live ? C.blue : C.rule);
          on[c].setAttribute("fill-opacity", live ? 0.6 : 0.45);
          kBar[c].setAttribute("opacity", o);
          kBar[c].setAttribute("fill-opacity", 0.55);
        }
        offWork.textContent = `tokens processed so far: ${sum(i)}`;
        onWork.textContent = `tokens processed so far: ${P + i}`;
        kShape.textContent = `(1, 2, ${P + i}, 32)`;
      },
    });
  })();

  /* ---------- Line chart helper for the measured figures ---------- */
  function lineChart(fig, opts) {
    const W = 960, H = 340, L = 66, R = 96, Tp = 18, B = 46;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria }, fig.querySelector(".ct-figure__stage"));
    const xs = opts.x, xMin = xs[0], xMax = xs[xs.length - 1];
    const ymax = opts.yMax, ymin = opts.yMin || 0;
    const px = (x) => L + ((x - xMin) / (xMax - xMin)) * (W - L - R);
    const py = (y) => H - B - ((y - ymin) / (ymax - ymin)) * (H - Tp - B);
    opts.yTicks.forEach((v) => {
      svg("line", { x1: L, x2: W - R, y1: py(v), y2: py(v), stroke: C.grid }, root);
      svg("text", { x: L - 8, y: py(v) + 4, "text-anchor": "end", "font-size": 12, fill: C.muted, text: v }, root);
    });
    opts.xTicks.forEach((v) => svg("text", { x: px(v), y: H - B + 20, "text-anchor": "middle", "font-size": 12, fill: C.muted, text: v }, root));
    svg("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 12.5, fill: C.soft, text: opts.xLabel }, root);
    svg("text", { x: 14, y: (Tp + H - B) / 2, "text-anchor": "middle", "font-size": 12.5, fill: C.soft, transform: `rotate(-90 14 ${(Tp + H - B) / 2})`, text: opts.yLabel }, root);
    const dots = opts.series.map((s) => {
      svg("path", { d: s.y.map((v, i) => [v, i]).filter(([v]) => v != null).map(([v, i], n) => (n ? "L" : "M") + px(xs[i]).toFixed(1) + " " + py(v).toFixed(1)).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }, root);
      const lastV = s.y[s.y.length - 1];
      svg("circle", { cx: px(xMax), cy: py(lastV), r: 4, fill: s.color, stroke: C.paper, "stroke-width": 2 }, root);
      svg("text", { x: px(xMax) + 9, y: py(lastV) + (s.dy || 0) + 4, "font-size": 12.5, fill: C.ink, text: s.name }, root);
      return svg("circle", { r: 5, fill: s.color, stroke: C.paper, "stroke-width": 2, opacity: 0 }, root);
    });
    const cross = svg("line", { y1: Tp, y2: H - B, stroke: C.muted, opacity: 0 }, root);
    const tipBox = svg("rect", { width: 196, height: 20 + 18 * opts.series.length, rx: 3, fill: C.hi, stroke: C.rule, opacity: 0 }, root);
    const tipHead = svg("text", { "font-size": 12, fill: C.ink, "font-weight": 600, opacity: 0 }, root);
    const tipRows = opts.series.map(() => svg("text", { "font-size": 12, fill: C.soft, opacity: 0 }, root));
    const hit = svg("rect", { x: L, y: Tp, width: W - L - R, height: H - Tp - B, fill: "transparent" }, root);
    const hide = () => [cross, tipBox, tipHead, ...tipRows, ...dots].forEach((n) => n.setAttribute("opacity", 0));
    hit.addEventListener("pointermove", (e) => {
      const b = root.getBoundingClientRect();
      const mx = ((e.clientX - b.left) / b.width) * W;
      let best = 0;
      xs.forEach((x, i) => { if (Math.abs(px(x) - mx) < Math.abs(px(xs[best]) - mx)) best = i; });
      const cx = px(xs[best]);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("opacity", 1);
      const bx = cx + 206 > W ? cx - 206 : cx + 10;
      tipBox.setAttribute("x", bx); tipBox.setAttribute("y", Tp + 4); tipBox.setAttribute("opacity", 1);
      tipHead.setAttribute("x", bx + 8); tipHead.setAttribute("y", Tp + 19); tipHead.setAttribute("opacity", 1);
      tipHead.textContent = opts.xName + " " + xs[best];
      opts.series.forEach((s, k) => {
        const yv = s.y[best];
        dots[k].setAttribute("cx", cx); dots[k].setAttribute("cy", yv == null ? -99 : py(yv)); dots[k].setAttribute("opacity", yv == null ? 0 : 1);
        tipRows[k].setAttribute("x", bx + 8); tipRows[k].setAttribute("y", Tp + 37 + k * 18); tipRows[k].setAttribute("opacity", 1);
        tipRows[k].textContent = s.name + ": " + (yv == null ? "not measured" : yv.toFixed(opts.digits));
      });
    });
    hit.addEventListener("pointerleave", hide);
  }

  (function latencyFigure() {
    const fig = $("fig-latency");
    if (!fig) return;
    const d = DATA.latency;
    const top = Math.ceil(Math.max(...d.off) / 10) * 10;
    const ticks = [0, 1, 2, 3, 4].map((i) => Math.round((top * i) / 4));
    lineChart(fig, { aria: "Measured per-step decode latency with and without a KV cache", x: d.T, xTicks: d.T.filter((v) => v !== 17 && v !== 48), yMax: top, yTicks: ticks, xLabel: "Sequence length T at this step", yLabel: "Milliseconds per step", xName: "T", digits: 2, series: [{ name: "No cache", y: d.off, color: C.orange }, { name: "Cache", y: d.on, color: C.blue }] });
  })();

  (function trainingFigure() {
    const fig = $("fig-training");
    if (!fig) return;
    const d = DATA.training;
    lineChart(fig, { aria: "Measured training and validation loss on Tiny Shakespeare", x: d.step, xTicks: d.step, yMax: 6, yMin: 0, yTicks: [0, 1, 2, 3, 4, 5, 6], xLabel: "Training step", yLabel: "Loss (nats per byte)", xName: "Step", digits: 3, series: [{ name: "Train", y: d.train, color: C.orange, dy: 8 }, { name: "Validation", y: d.val, color: C.blue, dy: -8 }] });
  })();

  /* ---------- Figure 10: mixture of experts ---------- */
  (function moeFigure() {
    const fig = $("fig-moe");
    if (!fig) return;
    const W = 960, H = 340, N = 6, E = 4;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "A router sends each token to its top two experts" }, fig.querySelector(".ct-figure__stage"));
    // Illustrative router outputs for six tokens over four experts.
    const P = [[.55, .25, .15, .05], [.1, .6, .2, .1], [.45, .1, .35, .1], [.05, .15, .3, .5], [.2, .5, .05, .25], [.1, .05, .25, .6]];
    const top = P.map((row) => row.map((p, e) => [p, e]).sort((a, b) => b[0] - a[0]).slice(0, 2));
    const tokY = (t) => 58 + t * 44, cellX = (e) => 130 + e * 54, expY = (e) => 40 + e * 66;
    svg("text", { x: 44, y: 34, "text-anchor": "middle", "font-size": 13, fill: C.muted, text: "tokens" }, root);
    svg("text", { x: 211, y: 34, "text-anchor": "middle", "font-size": 13, fill: C.muted, text: "router softmax (N, E)" }, root);
    for (let t = 0; t < N; t++) {
      svg("rect", { x: 18, y: tokY(t) - 15, width: 52, height: 30, rx: 3, fill: C.ink, "fill-opacity": 0.14, stroke: C.rule }, root);
      svg("text", { x: 44, y: tokY(t) + 5, "text-anchor": "middle", "font-size": 13, fill: C.ink, text: "t" + t }, root);
    }
    const cells = [];
    for (let t = 0; t < N; t++) for (let e = 0; e < E; e++) {
      const r = svg("rect", { x: cellX(e) - 24, y: tokY(t) - 15, width: 48, height: 30, rx: 3, fill: C.blue }, root);
      const tx = svg("text", { x: cellX(e), y: tokY(t) + 5, "text-anchor": "middle", "font-size": 12.5, fill: C.ink }, root);
      cells.push({ t, e, r, tx, chosen: top[t].some(([, i]) => i === e) });
    }
    for (let e = 0; e < E; e++) svg("text", { x: cellX(e), y: 330 - 12, "text-anchor": "middle", "font-size": 12, fill: C.muted, text: "E" + e }, root);
    const lineG = svg("g", null, root);
    const links = [];
    for (let t = 0; t < N; t++) top[t].forEach(([p, e]) => {
      const w = p / (top[t][0][0] + top[t][1][0]);
      links.push({ t, e, w, path: svg("path", { d: `M 300 ${tokY(t)} C 420 ${tokY(t)}, 470 ${expY(e) + 24}, 590 ${expY(e) + 24}`, fill: "none", stroke: C.orange, "stroke-width": 1 + 3.5 * w, "stroke-opacity": 0.75 }, lineG) });
    });
    const boxes = Array.from({ length: E }, (_, e) => {
      const g = svg("g", null, root);
      svg("rect", { x: 590, y: expY(e), width: 170, height: 48, rx: 4, fill: C.green, "fill-opacity": 0.16, stroke: C.green }, g);
      svg("text", { x: 675, y: expY(e) + 21, "text-anchor": "middle", "font-size": 13.5, fill: C.ink, text: "SwiGLU expert " + e }, g);
      return { g, load: svg("text", { x: 675, y: expY(e) + 39, "text-anchor": "middle", "font-size": 12, fill: C.soft }, g) };
    });
    const combine = svg("g", null, root);
    svg("path", { d: "M760 150 L 806 150", stroke: C.muted, "stroke-width": 1.6, fill: "none" }, combine);
    svg("circle", { cx: 826, cy: 150, r: 20, fill: C.hi, stroke: C.ink }, combine);
    svg("text", { x: 826, y: 156, "text-anchor": "middle", "font-size": 17, fill: C.ink, text: "Σ" }, combine);
    svg("text", { x: 884, y: 142, "text-anchor": "middle", "font-size": 13, fill: C.ink, text: "weighted" }, combine);
    svg("text", { x: 884, y: 160, "text-anchor": "middle", "font-size": 13, fill: C.ink, text: "sum" }, combine);
    const counts = Array(E).fill(0); top.forEach((r) => r.forEach(([, e]) => counts[e]++));
    const frames = [
      { note: "The router is one linear layer. It gives each token a probability for each expert. All tokens in a batch are pooled, so N = B &times; T." },
      { note: "<code>topk</code> keeps the two largest scores per token. The other experts will not see this token at all." },
      { note: "The kept scores are divided by their sum, so each token's two weights add to 1. Numbers on the links show this as line width." },
      { note: "Tokens are dispatched. Only 12 of the 24 possible expert evaluations run, which is why MoE is sparse." },
      { note: "Each expert's output is weighted and summed back to (N, D). Experts with no tokens cost nothing in this batch." },
      { note: "<code>load_balancing_loss</code> watches the load on each expert. If the router collapses onto a few experts, the loss rises above 1 and pushes the router back." },
    ];
    stepper(fig, {
      frames, tween: 1000, hold: 2300,
      render(i, t, from) {
        cells.forEach((c) => {
          const state = (f) => {
            if (f === 0) return { op: 0.08 + 0.75 * P[c.t][c.e], val: P[c.t][c.e] };
            if (!c.chosen) return { op: 0.03, val: null };
            const pair = top[c.t];
            const w = f >= 2 ? P[c.t][c.e] / (pair[0][0] + pair[1][0]) : P[c.t][c.e];
            return { op: 0.15 + 0.7 * w, val: w };
          };
          const a = state(from), b = state(i), now = t < 0.5 ? a : b;
          c.r.setAttribute("fill-opacity", lerp(a.op, b.op, t).toFixed(3));
          c.tx.textContent = now.val === null ? "" : now.val.toFixed(2);
          c.tx.setAttribute("fill", now.val !== null && now.val > 0.45 ? "#fff" : C.ink);
        });
        const on = (f, min) => (f >= min ? 1 : 0);
        links.forEach((l) => l.path.setAttribute("opacity", lerp(on(from, 3), on(i, 3), t).toFixed(3)));
        boxes.forEach((b, e) => {
          b.g.setAttribute("opacity", lerp(0.25 + 0.75 * on(from, 3), 0.25 + 0.75 * on(i, 3), t).toFixed(3));
          b.load.textContent = (t < 0.5 ? from : i) >= 3 ? `${counts[e]} tokens` : "";
        });
        combine.setAttribute("opacity", lerp(on(from, 4), on(i, 4), t).toFixed(3));
      },
    });
  })();
})();
