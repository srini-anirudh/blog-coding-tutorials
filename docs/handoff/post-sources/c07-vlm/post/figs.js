(function () {
  "use strict";
  const { svg, lerp, stepper } = window.CT;
  const DATA = window.C7_DATA;
  const C = { ink: "#14171a", soft: "#474e53", muted: "#7b8388", rule: "#cac6bd", paper: "#f7f6f3", hi: "#fbfaf8", blue: "#0b5ea8", orange: "#d66f00", green: "#2f8f5b", red: "#b8432f", grey: "#d9d6cf" };
  const $ = (id) => document.getElementById(id);
  function tag(parent, x, y, text, o) { return svg("text", Object.assign({ x, y, "text-anchor": "middle", "font-size": 14, fill: C.ink, text }, o || {}), parent); }
  function fadeLayer(world, on) { const g = svg("g", { opacity: 0 }, world); g._on = on; return g; }

  /* ---------- Figure 1: picture to sequence ---------- */
  (function pipeline() {
    const fig = $("fig-pipeline");
    if (!fig) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 460;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "A picture becomes 196 patch tokens, then 49 folded tokens, then 49 projected tokens placed into the text sequence" }, stage);
    const world = svg("g", null, root);
    const PX = 40, PY = 76, CELL = 15, G = 14;
    const pic = fadeLayer(world, (f) => f <= 2);
    svg("rect", { x: PX, y: PY, width: CELL * G, height: CELL * G, fill: "#f6f5f2", stroke: C.rule }, pic);
    const blobs = [[70, 76, 40], [150, 152, 32]];
    blobs.forEach(([cx, cy, r]) => svg("circle", { cx: PX + (cx * CELL * G) / 224, cy: PY + (cy * CELL * G) / 224, r: (r * CELL * G) / 224, fill: "#dc2828" }, pic));
    function overlap(i, j) {
      let hit = 0;
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
        const x = (j + (a + 0.5) / 4) * 16, y = (i + (b + 0.5) / 4) * 16;
        if (blobs.some(([cx, cy, r]) => (x - cx) ** 2 + (y - cy) ** 2 < r * r)) hit++;
      }
      return hit / 16;
    }
    const blockHit = {};
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const k = (i >> 1) + "," + (j >> 1); blockHit[k] = (blockHit[k] || 0) + overlap(i, j) / 4; }
    const SX = 56, SY = 372, SS = 13.45;
    const kinds = [];
    for (let t = 0; t < 63; t++) kinds.push(t >= 3 && t <= 51 ? "img" : "txt");
    const seq = fadeLayer(world, (f) => f >= 5);
    const slots = kinds.map((k, t) => svg("rect", { x: SX + t * SS, y: SY, width: SS - 2, height: 26, rx: 2, fill: k === "img" ? C.hi : C.blue, "fill-opacity": k === "img" ? 1 : 0.65, stroke: k === "img" ? C.orange : "none", "stroke-dasharray": k === "img" ? "2 2" : "none" }, seq));
    tag(seq, SX + 1.5 * SS, SY + 46, "user\\n", { "font-size": 13, fill: C.blue });
    tag(seq, SX + 27.5 * SS, SY + 46, "<image> × 49, positions 3 to 51", { "font-size": 13, fill: C.orange });
    tag(seq, SX + 57 * SS, SY + 46, "What do you see? … assistant\\n", { "font-size": 13, fill: C.blue });
    tag(seq, SX + 31.5 * SS, SY - 16, "input_ids (1, 63), then token embeddings (1, 63, 576)", { "font-size": 14, fill: C.soft });
    const cells = [];
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) cells.push({ i, j, I: i >> 1, J: j >> 1, hit: overlap(i, j), block: blockHit[(i >> 1) + "," + (j >> 1)], el: svg("rect", { rx: 1 }, world) });
    const FX = 310, FY = 76, TX = 610, TY = 88;
    function geom(f, c) {
      if (f <= 1) return { x: PX + c.j * CELL, y: PY + c.i * CELL, s: CELL };
      if (f === 2) return { x: FX + c.j * 15.5, y: FY + c.i * 15.5, s: 13.5 };
      if (f === 3) return { x: FX + c.J * 33 + (c.j & 1) * 14, y: FY + c.I * 33 + (c.i & 1) * 14, s: 13 };
      if (f <= 5) return { x: TX + c.J * 28, y: TY + c.I * 28, s: 24 };
      const k = c.I * 7 + c.J;
      return { x: SX + (3 + k) * SS, y: SY, s: SS - 2, h: 26 };
    }
    function style(f, c) {
      if (f === 0) return { fill: "none", op: 0, stroke: "none" };
      if (f === 1) return { fill: "none", op: 1, stroke: "rgba(20,23,26,.28)" };
      if (f <= 3) return { fill: c.hit > 0.15 ? C.red : C.muted, op: c.hit > 0.15 ? 0.3 + 0.6 * c.hit : 0.28, stroke: "none" };
      return { fill: C.orange, op: 0.3 + 0.7 * c.block, stroke: "none" };
    }
    const labs = [];
    function lab(on, x, y, text, o) { const L = fadeLayer(world, on); tag(L, x, y, text, o); labs.push(L); return L; }
    lab((f) => f <= 2, PX + 105, PY - 30, "One picture, resized to 224 × 224", { "font-size": 15 });
    lab((f) => f >= 1 && f <= 2, PX + 105, PY + 240, "pixel_values (1, 3, 224, 224)", { "font-size": 14, fill: C.orange });
    lab((f) => f === 2, FX + 108, FY - 30, "Encoder output: 14 × 14 tokens", { "font-size": 15 });
    lab((f) => f === 2, FX + 108, FY + 240, "(1, 196, 768)", { "font-size": 14, fill: C.orange });
    lab((f) => f === 3, FX + 108, FY - 30, "Pixel shuffle folds each 2 × 2 block", { "font-size": 15 });
    lab((f) => f === 3, FX + 108, FY + 250, "(1, 49, 3072): 4 × 768 per block", { "font-size": 14, fill: C.orange });
    lab((f) => f === 4 || f === 5, TX + 96, TY - 42, "The projector maps each block to d_lm", { "font-size": 15 });
    lab((f) => f === 4 || f === 5, TX + 96, TY + 222, "(1, 49, 576)", { "font-size": 14, fill: C.orange });
    lab((f) => f >= 6, 480, 110, "inputs_embeds (1, 63, 576): 14 text rows and 49 image rows", { "font-size": 17 });
    lab((f) => f === 6, 480, 145, "masked_scatter fills the placeholders in reading order. The language model cannot tell them apart from text.", { "font-size": 13, fill: C.soft });
    const lm = fadeLayer(world, (f) => f === 7);
    svg("rect", { x: 320, y: 186, width: 320, height: 66, rx: 4, fill: C.hi, stroke: C.ink }, lm);
    tag(lm, 480, 214, "SmolLM2-135M (30 layers)", { "font-size": 15 });
    tag(lm, 480, 236, "logits (1, 63, 49216)", { "font-size": 14, fill: C.orange });
    svg("path", { d: "M480 364 L480 256", stroke: C.ink, "stroke-width": 1.4, fill: "none" }, lm);
    svg("path", { d: "M474 262 L480 252 L486 262 z", fill: C.ink }, lm);
    const cams = [
      { x: 230, y: 195, w: 620 }, { x: 230, y: 195, w: 620 }, { x: 310, y: 195, w: 790 }, { x: 420, y: 195, w: 790 },
      { x: 600, y: 195, w: 790 }, { x: 480, y: 230, w: 960 }, { x: 480, y: 230, w: 960 }, { x: 480, y: 230, w: 960 },
    ];
    const allLayers = [pic, seq, lm, ...labs];
    const frames = [
      { note: "We start with one picture. The processor has already resized and normalised it, so the encoder sees 3 × 224 × 224 floats in [-1, 1]." },
      { note: "SigLIP cuts the crop into 16 px patches. That gives a 14 × 14 grid, so 196 patches." },
      { note: "The encoder returns one 768-number vector per patch: shape (1, 196, 768). Spatial order is kept." },
      { note: "Pixel shuffle concatenates each 2 × 2 block. The token count falls from 196 to 49, and each token grows to 3072 numbers. No information is dropped." },
      { note: "The projector is the only new part. It maps 3072 numbers to 576, the size SmolLM2 expects." },
      { note: "The prompt is tokenised with one placeholder per image token. Placeholder embeddings are meaningless. They only hold the seats." },
      { note: "The 49 projected tokens overwrite the 49 placeholders in reading order. This is the whole multimodal trick in a decoder-only VLM." },
      { note: "The language model now runs as usual on embeddings instead of ids. It returns one distribution over the vocabulary per position." },
    ];
    stepper(fig, {
      frames, tween: 1100, hold: 2200,
      render(i, t, from) {
        const a = cams[from], b = cams[i];
        const w = lerp(a.w, b.w, t), h = (w * VH) / VW;
        root.setAttribute("viewBox", `${lerp(a.x, b.x, t) - w / 2} ${lerp(a.y, b.y, t) - h / 2} ${w} ${h}`);
        allLayers.forEach((L) => L.setAttribute("opacity", lerp(L._on(from) ? 1 : 0, L._on(i) ? 1 : 0, t)));
        cells.forEach((c) => {
          const stag = i === 6 && from !== 6 ? Math.min(1, Math.max(0, t * 1.7 - ((c.I * 7 + c.J) / 49) * 0.7)) : t;
          const g0 = geom(from, c), g1 = geom(i, c);
          const s = lerp(g0.s, g1.s, stag), hh = lerp(g0.h || g0.s, g1.h || g1.s, stag);
          c.el.setAttribute("x", lerp(g0.x, g1.x, stag)); c.el.setAttribute("y", lerp(g0.y, g1.y, stag));
          c.el.setAttribute("width", s); c.el.setAttribute("height", hh);
          const st = t < 0.5 ? style(from, c) : style(i, c);
          const s0 = style(from, c), s1 = style(i, c);
          c.el.setAttribute("fill", st.fill); c.el.setAttribute("stroke", st.stroke);
          c.el.setAttribute("opacity", lerp(s0.op, s1.op, t));
        });
      },
    });
  })();

  /* ---------- Figure 2: scattering a batch ---------- */
  (function scatter() {
    const fig = $("fig-scatter");
    if (!fig) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const root = svg("svg", { viewBox: "0 0 860 360", role: "img", "aria-label": "Three crops are flattened and scattered into two samples in reading order" }, stage);
    const cropColors = [C.orange, C.green, "#7a4fb5"];
    const rows = [
      ["t", "t", 0, 0, 0, 0, "t", 1, 1, 1, 1, "t", "t", "t"],
      ["t", "t", "t", 2, 2, 2, 2, "t", "t", "t", "p", "p", "p", "p"],
    ];
    const X0 = 170, Y0 = 200, W = 44, H = 38;
    const rowCells = [];
    rows.forEach((r, b) => {
      tag(root, 88, Y0 + b * (H + 22) + 24, `sample ${b}`, { "font-size": 14, fill: C.soft });
      r.forEach((k, t) => {
        const x = X0 + t * W, y = Y0 + b * (H + 22);
        const el = svg("rect", { x, y, width: W - 4, height: H, rx: 3, fill: k === "t" ? C.blue : k === "p" ? C.grey : C.hi, "fill-opacity": k === "t" ? 0.6 : 1, stroke: typeof k === "number" ? C.orange : "none", "stroke-dasharray": typeof k === "number" ? "3 2" : "none" }, root);
        const text = svg("text", { x: x + (W - 4) / 2, y: y + 24, "text-anchor": "middle", "font-size": 12, fill: typeof k === "number" ? C.orange : k === "p" ? C.muted : "#fff", text: typeof k === "number" ? "img" : k === "p" ? "pad" : "txt" }, root);
        rowCells.push({ b, t, k, x, y, el, text });
      });
    });
    tag(root, 430, Y0 + 2 * (H + 22) + 8, "input_ids and inputs_embeds, B × N (2 × 14)", { "font-size": 14, fill: C.soft });
    const queue = [];
    [0, 1, 2].forEach((c) => {
      for (let n = 0; n < 4; n++) {
        const g = { c, n, el: svg("rect", { width: W - 4, height: H, rx: 3, fill: cropColors[c] }, root), text: svg("text", { "text-anchor": "middle", "font-size": 11, fill: "#fff", text: `${c}.${n}` }, root) };
        queue.push(g);
      }
    });
    const qlabel = tag(root, 430, 28, "", { "font-size": 15 });
    const bug = tag(root, 430, 166, "", { "font-size": 14, fill: C.red });
    function slotsOf() { return rowCells.filter((c) => typeof c.k === "number"); }
    const slotList = slotsOf();
    const frames = [
      { note: "Two samples are padded to one length. Sample 0 holds two pictures. Sample 1 holds one. Each crop owns 4 placeholder seats here. The real model uses 49." },
      { note: "The encoder and projector see crops, not samples. Three crops arrive as a tensor of shape (3, 4, d)." },
      { note: "Flatten to (12, d). Nothing here records which sample a vector belongs to. Order is the only link." },
      { note: "masked_scatter walks the batch in reading order. Crop 0 fills the first 4 seats." },
      { note: "Crop 1 fills the next 4 seats in the same sample." },
      { note: "Crop 2 moves on to sample 1. Padding is never touched." },
      { note: "Now reverse pixel_values. The counts still match, so nothing raises. Sample 0 now reads crop 2 and crop 1. Sample 1 reads crop 0. The check we wrote cannot catch this." },
    ];
    function qpos(f, g, idx) {
      if (f === 0) return { x: 200 + g.c * 210 + g.n * 10, y: 60, o: 0 };
      if (f === 1) return { x: 96 + g.c * 240 + (g.n % 2) * (W - 2), y: 44 + (g.n >> 1) * (H + 2), o: 1 };
      const fillUpTo = f === 2 ? -1 : f === 3 ? 4 : f === 4 ? 8 : 12;
      const order = f === 6 ? (11 - idx + 0) : idx;
      if (f === 6) {
        const mapped = (2 - g.c) * 4 + g.n;
        const s = slotList[mapped];
        return { x: s.x, y: s.y, o: 1 };
      }
      if (idx < fillUpTo) { const s = slotList[idx]; return { x: s.x, y: s.y, o: 1 }; }
      return { x: 170 + idx * W - (idx > 11 ? 0 : 0), y: 70, o: 1 + 0 * order };
    }
    stepper(fig, {
      frames, tween: 1000, hold: 2100,
      render(i, t, from) {
        const f = t < 0.5 ? from : i;
        qlabel.textContent = f === 0 ? "" : f === 1 ? "image_features (3, 4, d): crops 0, 1, 2" : f === 6 ? "pixel_values reversed by mistake" : "image_features.reshape(-1, d): 12 vectors, in order";
        qlabel.setAttribute("fill", f === 6 ? C.red : C.ink);
        bug.textContent = f === 6 ? "No error: 12 seats, 12 vectors. Each sample now reads the wrong picture." : "";
        queue.forEach((g, idx) => {
          const a = qpos(from, g, idx), b = qpos(i, g, idx);
          const x = lerp(a.x, b.x, t), y = lerp(a.y, b.y, t);
          g.el.setAttribute("x", x); g.el.setAttribute("y", y);
          g.el.setAttribute("opacity", lerp(a.o, b.o, t));
          g.text.setAttribute("x", x + (W - 4) / 2); g.text.setAttribute("y", y + 24);
          g.text.setAttribute("opacity", lerp(a.o, b.o, t));
        });
      },
    });
  })();

  /* ---------- Figure 3: loss-mask painter ---------- */
  (function labels() {
    const fig = $("fig-labels");
    if (!fig) return;
    const toks = [
      ["<|im_start|>", 1, "s"], ["user", 1, "u"], ["\\n", 1, "u"], ["<image> × 49", 49, "i"], ["What", 1, "u"], [" do", 1, "u"], [" you", 1, "u"], [" see", 1, "u"], ["?", 1, "u"], ["<|im_end|>", 1, "s"], ["\\n", 1, "u"],
      ["<|im_start|>", 1, "s"], ["ass", 1, "u"], ["istant", 1, "u"], ["\\n", 1, "u"], ["Two", 1, "a"], [" red", 1, "a"], [" circles", 1, "a"], [".", 1, "a"], ["<|im_end|>", 1, "e"], ["\\n", 1, "u"],
    ];
    const box = $("labels-chips");
    const state = toks.map(() => false);
    const chips = toks.map(([text, n, kind], k) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "tok-chip tok-" + kind; b.textContent = text;
      b.addEventListener("click", () => { state[k] = !state[k]; preset(null); draw(); });
      box.appendChild(b);
      return b;
    });
    const presets = {
      correct: { f: (k) => toks[k][2] === "a" || toks[k][2] === "e", note: "Correct. Five targets: the reply and its stop token. Every other position has label -100 and adds nothing to the loss." },
      nostop: { f: (k) => toks[k][2] === "a", note: "The stop token is missing. The caption loss looks fine, but the model never learns where to stop. Generation runs on until max_new_tokens." },
      all: { f: () => true, note: DATA.badNote },
      question: { f: (k) => k > 3, note: "The question is supervised too. The projector spends gradient on predicting a fixed set of prompts, which carries little information about the picture." },
    };
    const buttons = [...fig.querySelectorAll("[data-labels]")];
    function preset(name) {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.labels === name)));
      if (name) { const p = presets[name]; toks.forEach((_, k) => (state[k] = p.f(k))); fig.querySelector(".ct-figure__note").textContent = p.note; }
      else fig.querySelector(".ct-figure__note").textContent = "Custom mask. Click any token to include or exclude it.";
    }
    function draw() {
      let tgt = 0, imgT = 0, first = null, firstPrev = null, pos = 0, prevText = null;
      toks.forEach(([text, n, kind], k) => {
        chips[k].classList.toggle("is-on", state[k]);
        chips[k].setAttribute("aria-pressed", String(state[k]));
        if (state[k]) { tgt += n; if (kind === "i") imgT += n; if (first === null && k > 0) { first = text; firstPrev = prevText; } }
        pos += n; prevText = text;
      });
      $("labels-count").textContent = tgt + " of 69";
      $("labels-img").textContent = tgt ? Math.round((100 * imgT) / tgt) + "%" : "0%";
      $("labels-first").textContent = first ? `${JSON.stringify(firstPrev)} → ${JSON.stringify(first)}` : "none";
    }
    buttons.forEach((b) => b.addEventListener("click", () => { preset(b.dataset.labels); draw(); }));
    preset("correct"); draw();
  })();

  /* ---------- Figure 4: mask and position builder ---------- */
  (function maskFig() {
    const fig = $("fig-mask");
    if (!fig) return;
    const padIn = $("mask-pad"), imgIn = $("mask-img"), bidir = $("mask-bidir"), causalIn = $("mask-causal");
    const S = 330, PAD = 40;
    const root = svg("svg", { viewBox: `0 0 ${S + PAD + 8} ${S + PAD + 8}`, role: "img", "aria-label": "Attention mask grid" }, $("mask-stage"));
    root.style.maxWidth = "420px"; root.style.margin = "0 auto";
    function draw() {
      const p = +padIn.value, side = +imgIn.value, m = side * side, n = p + 2 + m + 2;
      $("mask-pad-out").textContent = p; $("mask-img-out").textContent = m;
      const kind = [];
      for (let t = 0; t < p; t++) kind.push("pad");
      kind.push("txt", "txt");
      for (let t = 0; t < m; t++) kind.push("img");
      kind.push("txt", "txt");
      root.innerHTML = "";
      const cell = S / n;
      const col = { pad: C.muted, txt: C.blue, img: C.orange };
      let allowed = 0;
      const lines = [];
      for (let i = 0; i < n; i++) {
        let line = "";
        svg("rect", { x: PAD - 8, y: PAD + i * cell + 1, width: 6, height: cell - 2, fill: col[kind[i]] }, root);
        svg("rect", { x: PAD + i * cell + 1, y: PAD - 8, width: cell - 2, height: 6, fill: col[kind[i]] }, root);
        for (let j = 0; j < n; j++) {
          let ok = kind[j] !== "pad" && kind[i] !== "pad" && (causalIn.checked ? j <= i : true);
          if (bidir.checked && kind[i] === "img" && kind[j] === "img") ok = true;
          if (kind[i] === "pad") ok = false;
          if (ok) allowed++;
          line += ok ? "#" : ".";
          svg("rect", { x: PAD + j * cell + 1, y: PAD + i * cell + 1, width: cell - 2, height: cell - 2, rx: 1.5, fill: ok ? (kind[i] === "img" && kind[j] === "img" ? C.orange : C.blue) : C.hi, "fill-opacity": ok ? 0.72 : 1, stroke: ok ? "none" : C.rule }, root);
        }
        lines.push(kind[i].padStart(4) + " " + line);
      }
      svg("text", { x: PAD + S / 2, y: 18, "text-anchor": "middle", "font-size": 12, fill: C.muted, text: "key j" }, root);
      svg("text", { x: 14, y: PAD + S / 2, "text-anchor": "middle", "font-size": 12, fill: C.muted, transform: `rotate(-90 14 ${PAD + S / 2})`, text: "query i" }, root);
      const oneD = kind.map((k, t) => (k === "pad" ? 1 : t - p));
      let next = 0; const tt = [], hh = [], ww = [];
      let seen = 0;
      kind.forEach((k) => {
        if (k === "pad") { tt.push("-"); hh.push("-"); ww.push("-"); return; }
        if (k === "txt") { tt.push(next); hh.push(next); ww.push(next); next++; return; }
        tt.push(next); hh.push(next + Math.floor(seen / side)); ww.push(next + (seen % side)); seen++;
        if (seen === m) next += side;
      });
      $("mask-out").textContent = lines.join("\n") + "\n\n1D positions  " + oneD.join(" ") + "\n3D t          " + tt.join(" ") + "\n3D h          " + hh.join(" ") + "\n3D w          " + ww.join(" ");
      $("mask-allowed").textContent = allowed;
      $("mask-frac").textContent = Math.round((100 * allowed) / (n * n)) + "%";
      $("mask-after").textContent = oneD[n - 1] + " vs " + ww[n - 1];
    }
    [padIn, imgIn, bidir, causalIn].forEach((x) => x.addEventListener("input", draw));
    draw();
  })();

  /* ---------- Figure 5: resolution to token count ---------- */
  (function resolution() {
    const fig = $("fig-resolution");
    if (!fig) return;
    const wIn = $("res-w"), hIn = $("res-h"), tilesIn = $("res-tiles"), shuffleIn = $("res-shuffle");
    const stratBtns = [...fig.querySelectorAll("[data-strategy]")];
    let strategy = "tile";
    const SIZE = 224;
    function grids(maxTiles) {
      const out = [];
      for (let c = 1; c <= maxTiles; c++) for (let r = 1; r <= maxTiles; r++) if (c * r <= maxTiles) out.push([c, r]);
      return out.sort((a, b) => a[0] * a[1] - b[0] * b[1] || a[0] - b[0]);
    }
    function selectGrid(w, h, maxTiles) {
      const aspect = w / h;
      let best = [1, 1], bestDiff = Infinity;
      grids(maxTiles).forEach(([c, r]) => {
        const diff = Math.abs(aspect - c / r);
        if (diff < bestDiff) { best = [c, r]; bestDiff = diff; }
        else if (diff === bestDiff && w * h > 0.5 * SIZE * SIZE * c * r) best = [c, r];
      });
      return best;
    }
    window.__selectGrid = selectGrid;
    const VW = 860, VH = 300;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "The picture and the crops it becomes" }, $("res-stage"));
    function scene(parent, x, y, w, h, vb) {
      const s = svg("svg", { x, y, width: w, height: h, viewBox: vb.join(" "), preserveAspectRatio: "none" }, parent);
      const [W, H] = [+wIn.value, +hIn.value];
      svg("rect", { x: 0, y: 0, width: W, height: H, fill: "#f6f5f2" }, s);
      svg("ellipse", { cx: W * 0.3, cy: H * 0.45, rx: Math.min(W, H) * 0.22, ry: Math.min(W, H) * 0.22, fill: "#dc2828" }, s);
      svg("rect", { x: W * 0.62, y: H * 0.5, width: Math.min(W, H) * 0.3, height: Math.min(W, H) * 0.3, fill: "#2850dc" }, s);
      return s;
    }
    function draw() {
      const W = +wIn.value, H = +hIn.value, maxT = +tilesIn.value, s = shuffleIn.checked ? 2 : 1;
      $("res-w-out").textContent = W; $("res-h-out").textContent = H; $("res-tiles-out").textContent = maxT;
      stratBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.strategy === strategy)));
      tilesIn.disabled = strategy !== "tile";
      root.innerHTML = "";
      const scale = Math.min(300 / W, 230 / H), pw = W * scale, ph = H * scale, px = 20, py = 40;
      tag(root, px + 150, 22, `input picture, ${W} × ${H}`, { "font-size": 14, fill: C.soft });
      scene(root, px, py, pw, ph, [0, 0, W, H]);
      svg("rect", { x: px, y: py, width: pw, height: ph, fill: "none", stroke: C.ink }, root);
      const X = 370;
      let crops = 1, cols = 1, rows = 1, caption = "";
      const side = 104;
      if (strategy === "resize") {
        scene(root, X, py, 200, 200, [0, 0, W, H]);
        svg("rect", { x: X, y: py, width: 200, height: 200, fill: "none", stroke: C.ink }, root);
        const stretch = Math.max(W / H, H / W);
        caption = stretch > 1.02 ? `1 crop, shapes stretched ${stretch.toFixed(2)}×` : "1 crop, no distortion";
      } else if (strategy === "pad") {
        const side2 = Math.max(W, H);
        const canvas = svg("svg", { x: X, y: py, width: 200, height: 200, viewBox: `0 0 ${side2} ${side2}` }, root);
        svg("rect", { x: 0, y: 0, width: side2, height: side2, fill: C.grey }, canvas);
        scene(canvas, (side2 - W) / 2, (side2 - H) / 2, W, H, [0, 0, W, H]);
        svg("rect", { x: X, y: py, width: 200, height: 200, fill: "none", stroke: C.ink }, root);
        caption = `1 crop, ${Math.round((100 * Math.min(W, H)) / side2)}% of its height carries content`;
      } else {
        [cols, rows] = selectGrid(W, H, maxT);
        const tile = Math.min(side, 230 / rows, 300 / cols);
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          scene(root, X + c * (tile + 5), py + r * (tile + 5), tile, tile, [(c * W) / cols, (r * H) / rows, W / cols, H / rows]);
          svg("rect", { x: X + c * (tile + 5), y: py + r * (tile + 5), width: tile, height: tile, fill: "none", stroke: C.ink }, root);
          svg("line", { x1: px + (c * pw) / cols, y1: py, x2: px + (c * pw) / cols, y2: py + ph, stroke: C.orange, "stroke-opacity": c ? 0.8 : 0 }, root);
          svg("line", { x1: px, y1: py + (r * ph) / rows, x2: px + pw, y2: py + (r * ph) / rows, stroke: C.orange, "stroke-opacity": r ? 0.8 : 0 }, root);
        }
        crops = cols * rows;
        if (crops > 1) {
          crops += 1;
          const tx = X + cols * (tile + 5) + 18;
          scene(root, tx, py, 80, 80, [0, 0, W, H]);
          svg("rect", { x: tx, y: py, width: 80, height: 80, fill: "none", stroke: C.green, "stroke-width": 1.6 }, root);
          tag(root, tx + 40, py + 98, "thumbnail", { "font-size": 12, fill: C.green });
        }
        caption = `${cols} × ${rows} grid${crops > 1 ? " + thumbnail" : ""}`;
      }
      tag(root, X + 150, 22, "crops the encoder sees, 224 × 224 each", { "font-size": 14, fill: C.soft });
      tag(root, 430, 290, caption, { "font-size": 14, fill: C.orange });
      const perCrop = 196 / (s * s), tokens = crops * perCrop, seq = tokens + 14;
      $("res-crops").textContent = crops;
      $("res-per").textContent = perCrop;
      $("res-tokens").textContent = tokens;
      $("res-kv").textContent = ((seq * 46080) / 1e6).toFixed(1) + " MB";
    }
    [wIn, hIn, tilesIn, shuffleIn].forEach((x) => x.addEventListener("input", draw));
    stratBtns.forEach((b) => b.addEventListener("click", () => { strategy = b.dataset.strategy; draw(); }));
    draw();
  })();

  /* ---------- Figure 6: generation ---------- */
  (function generation() {
    const fig = $("fig-generate");
    if (!fig) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const root = svg("svg", { viewBox: "0 0 860 300", role: "img", "aria-label": "Prefill runs the whole prompt once, then each decode step feeds one new token" }, stage);
    const words = DATA.caption;
    const X0 = 40, Y = 92, TW = 7.4;
    const prompt = [];
    for (let t = 0; t < 63; t++) prompt.push(svg("rect", { x: X0 + t * TW, y: Y, width: TW - 1, height: 34, fill: t >= 3 && t <= 51 ? C.orange : C.blue, "fill-opacity": 0.7 }, root));
    tag(root, X0 + 31 * TW, Y - 14, "prompt: 49 image rows + 14 text rows", { "font-size": 13, fill: C.soft });
    const newT = words.map((w, k) => {
      const x = X0 + 63 * TW + 8 + k * 62;
      return { g: svg("g", { opacity: 0 }, root), x, w };
    });
    newT.forEach((n) => {
      svg("rect", { x: n.x, y: Y, width: 58, height: 34, rx: 3, fill: C.green, "fill-opacity": 0.85 }, n.g);
      svg("text", { x: n.x + 29, y: Y + 22, "text-anchor": "middle", "font-size": 11.5, fill: "#fff", text: n.w }, n.g);
    });
    const runBar = svg("rect", { x: X0, y: Y - 6, width: 0, height: 46, fill: "none", stroke: C.ink, "stroke-width": 1.6, rx: 3 }, root);
    const title = tag(root, 430, 30, "", { "font-size": 16 });
    const CY = 200;
    tag(root, 130, CY + 16, "KV cache, all 30 layers", { "font-size": 13, fill: C.soft });
    svg("rect", { x: 240, y: CY, width: 560, height: 24, fill: C.hi, stroke: C.rule }, root);
    const bar = svg("rect", { x: 240, y: CY, width: 0, height: 24, fill: C.blue, "fill-opacity": 0.7 }, root);
    const mb = tag(root, 520, CY + 48, "", { "font-size": 14, fill: C.orange });
    const vis = tag(root, 430, 276, "", { "font-size": 13, fill: C.soft });
    const frames = [
      { note: "The processor builds a 63-token prompt. Nothing has run yet." },
      { note: "Prefill: the encoder and projector run once, then the language model runs over all 63 rows and stores keys and values." },
    ].concat(words.map((w, k) => ({ note: k === 0 ? `Decode step 1. The newest row of logits picks "${w}". Only one new row goes through the model. The image is never encoded again.` : `Decode step ${k + 1}: "${w}". The cache grows by one position per step.` })));
    const kbPerTok = 46080 / 1e6;
    stepper(fig, {
      frames, tween: 800, hold: 1700,
      render(i, t, from) {
        const f = t < 0.5 ? from : i;
        const count = (s) => (s === 0 ? 0 : 63 + (s - 1));
        const toks = lerp(count(from), count(i), t);
        bar.setAttribute("width", (560 * toks) / (63 + words.length));
        mb.textContent = `${toks.toFixed(0)} positions × 46,080 bytes = ${((toks * kbPerTok)).toFixed(2)} MB`;
        title.textContent = f === 0 ? "Before generation" : f === 1 ? "Prefill: 63 positions in one forward pass" : `Decode: 1 position, ${63 + f - 2} cached`;
        vis.textContent = f === 0 ? "" : f === 1 ? "SigLIP runs here, and only here." : "SigLIP does not run. pixel_values is not used again.";
        newT.forEach((n, k) => n.g.setAttribute("opacity", lerp(from >= k + 2 ? 1 : 0, i >= k + 2 ? 1 : 0, t)));
        if (f === 0) { runBar.setAttribute("width", 0); }
        else if (f === 1) { runBar.setAttribute("x", X0 - 3); runBar.setAttribute("width", 63 * TW + 5); }
        else { const n = newT[f - 2]; runBar.setAttribute("x", n.x - 3); runBar.setAttribute("width", 64); }
      },
    });
  })();

  /* ---------- Figure 7: alignment run ---------- */
  (function alignment() {
    const fig = $("fig-align");
    if (!fig) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const root = svg("svg", { viewBox: "0 0 900 420", role: "img", "aria-label": "Loss curve and captions during projector training" }, stage);
    const snaps = DATA.snapshots, hist = DATA.history;
    const LX = 56, LY = 40, LW = 330, LH = 190;
    const maxL = Math.max(...hist.map((h) => h.loss)), maxS = hist[hist.length - 1].step + 1;
    const px = (s) => LX + (s / maxS) * LW, py = (l) => LY + LH - (l / maxL) * LH;
    svg("line", { x1: LX, y1: LY + LH, x2: LX + LW, y2: LY + LH, stroke: C.ink }, root);
    svg("line", { x1: LX, y1: LY, x2: LX, y2: LY + LH, stroke: C.ink }, root);
    [0, 1, 2, 3, 4].forEach((v) => { if (v <= maxL) { svg("text", { x: LX - 8, y: py(v) + 4, "text-anchor": "end", "font-size": 11, fill: C.muted, text: v }, root); svg("line", { x1: LX, y1: py(v), x2: LX + LW, y2: py(v), stroke: C.rule, "stroke-dasharray": "2 3" }, root); } });
    tag(root, LX + LW / 2, 24, "Training loss, batch of 16 (MPS)", { "font-size": 13, fill: C.soft });
    tag(root, LX + LW / 2, LY + LH + 30, "step", { "font-size": 12, fill: C.muted });
    const curve = svg("path", { fill: "none", stroke: C.blue, "stroke-width": 1.6 }, root);
    const dot = svg("circle", { r: 5, fill: C.orange }, root);
    const stepLabel = tag(root, LX + LW / 2, LY + LH + 56, "", { "font-size": 15 });
    const bars = ["count", "color", "shape", "all"].map((k, n) => {
      const y = 322 + n * 22;
      svg("text", { x: LX + 40, y: y + 12, "text-anchor": "end", "font-size": 12, fill: C.soft, text: k === "all" ? "all three" : k === "color" ? "colour" : k }, root);
      svg("rect", { x: LX + 48, y, width: 240, height: 15, fill: C.hi, stroke: C.rule }, root);
      return { k, bar: svg("rect", { x: LX + 48, y, width: 0, height: 15, fill: k === "all" ? C.green : C.blue, "fill-opacity": 0.75 }, root), val: svg("text", { x: LX + 296, y: y + 12, "font-size": 12, fill: C.ink }, root) };
    });
    const CX = 450;
    tag(root, CX + 215, 24, "Greedy captions for the same held-out pictures", { "font-size": 13, fill: C.soft });
    const rows = DATA.images.map((im, n) => {
      const y = 40 + n * 60;
      const ns = "http://www.w3.org/1999/xlink";
      const img = svg("image", { x: CX, y, width: 52, height: 52 }, root);
      img.setAttributeNS(ns, "href", im.src); img.setAttribute("href", im.src);
      svg("text", { x: CX + 62, y: y + 18, "font-size": 11.5, fill: C.muted, text: "target: " + im.target }, root);
      return svg("text", { x: CX + 62, y: y + 40, "font-size": 13, fill: C.ink }, root);
    });
    const frames = snaps.map((s) => ({
      note: s.step === 0
        ? "Step 0. The projector is random. SmolLM2 receives noise where the image should be and writes generic text. It knows nothing of our caption format."
        : `Step ${s.step}. All-three-correct on ${s.step === snaps[snaps.length - 1].step ? DATA.nFinal : 32} held-out pictures: ${(s.score.all * 100).toFixed(0)}%.`,
    }));
    const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
    stepper(fig, {
      frames, tween: 900, hold: 2100,
      render(i, t, from) {
        const a = snaps[from], b = snaps[i];
        const s = lerp(a.step, b.step, t);
        let d = "";
        hist.forEach((h) => { if (h.step <= s) d += (d ? "L" : "M") + px(h.step).toFixed(1) + " " + py(Math.min(h.loss, maxL)).toFixed(1); });
        curve.setAttribute("d", d);
        const last = hist.filter((h) => h.step <= s).pop() || hist[0];
        dot.setAttribute("cx", px(last.step)); dot.setAttribute("cy", py(last.loss));
        stepLabel.textContent = `step ${Math.round(s)}, loss ${last.loss.toFixed(3)}`;
        bars.forEach((B) => {
          const v = lerp(a.score[B.k], b.score[B.k], t);
          B.bar.setAttribute("width", 240 * v);
          B.val.textContent = (v * 100).toFixed(0) + "%";
        });
        const shown = t < 0.5 ? a : b;
        rows.forEach((el, n) => { el.textContent = clip(shown.captions[n] || "", 46); });
      },
    });
  })();
})();
