(function () {
  "use strict";
  const { svg, lerp, stepper } = window.CT;
  const D = window.C9_DATA;
  const C = { ink: "#14171a", soft: "#474e53", muted: "#7b8388", rule: "#cac6bd", paper: "#f7f6f3", hi: "#fbfaf8", blue: "#0b5ea8", orange: "#d66f00", green: "#2f8f5b", red: "#b8432f", grey: "#dedad2" };
  const SHAPE_FILL = { red: "#d63031", green: "#279a48", blue: "#2860c8", yellow: "#eebe14" };
  const pct = (x) => Math.round(x * 100) + "%";
  const txt = (g, x, y, t, o) => svg("text", Object.assign({ x, y, "text-anchor": "middle", "font-size": 13, fill: C.ink, text: t }, o || {}), g);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  /* ---------- Figure 1: hallucination-negative generator ---------- */
  (function negatives() {
    const fig = document.getElementById("fig-neg");
    if (!fig) return;
    const sceneHost = fig.querySelector(".neg-scene");
    const COLORS = ["red", "green", "blue", "yellow"], SHAPES = ["circle", "square", "triangle"];
    const WORDS = ["CAT", "DOG", "SUN", "MAP", "BOX", "CUP", "PEN", "KEY", "BUS", "HAT"];
    const OFF = ["purple", "orange", "pink"];
    let seed = 11, kind = "color", mode = "balanced";
    let s = 1;
    const rnd = () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
    const pick = (a) => a[Math.floor(rnd() * a.length) % a.length];
    function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

    function sampleScene() {
      const cells = [];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cells.push([r, c]);
      shuffle(cells);
      const wc = cells.pop();
      const word = pick(WORDS);
      const n = 2 + Math.floor(rnd() * 5);
      const ids = [];
      SHAPES.forEach((sh) => COLORS.forEach((co) => ids.push([sh, co])));
      shuffle(ids);
      const objs = [];
      for (let i = 0; i < n; i++) {
        const [r, c] = cells[i];
        const side = 80 + Math.floor(rnd() * 40);
        const x = c * 170 + 8 + Math.floor(rnd() * (170 - side - 16));
        const y = r * 170 + 8 + Math.floor(rnd() * (170 - side - 16));
        objs.push({ shape: ids[i][0], color: ids[i][1], x, y, side });
      }
      return { objs, word, wc };
    }
    function question(sc) {
      const objs = sc.objs;
      const countShape = (sh) => objs.filter((o) => o.shape === sh).length;
      if (kind === "color") {
        const u = objs.filter((o) => countShape(o.shape) === 1);
        if (!u.length) return null;
        const o = pick(u);
        return { prompt: `What color is the ${o.shape}?`, ans: o.color, t: (v) => `The ${o.shape} is ${v}.`, focus: [o] };
      }
      if (kind === "count") {
        const sh = pick([...new Set(objs.map((o) => o.shape))]);
        return { prompt: `How many ${sh}s are there?`, ans: String(countShape(sh)), t: (v) => `The number of ${sh}s is ${v}.`, focus: objs.filter((o) => o.shape === sh) };
      }
      if (kind === "object") {
        const u = objs.filter((o) => objs.filter((p) => p.color === o.color).length === 1);
        if (!u.length) return null;
        const o = pick(u);
        return { prompt: `What shape is the ${o.color} object?`, ans: o.shape, t: (v) => `The ${o.color} object is a ${v}.`, focus: [o] };
      }
      if (kind === "ocr") return { prompt: "What word is written in the image?", ans: sc.word, t: (v) => `The word is ${v}.`, focus: [] };
      const ps = [];
      objs.forEach((a) => objs.forEach((b) => { if (a !== b && Math.abs(a.x + a.side / 2 - b.x - b.side / 2) > 60) ps.push([a, b]); }));
      if (!ps.length) return null;
      const [a, b] = pick(ps);
      const side = a.x + a.side / 2 < b.x + b.side / 2 ? "left" : "right";
      return { prompt: `Is the ${a.color} ${a.shape} to the left or to the right of the ${b.color} ${b.shape}?`, ans: side, t: (v) => `The ${a.color} ${a.shape} is to the ${v} of the ${b.color} ${b.shape}.`, focus: [a, b] };
    }
    function wrong(ans) {
      if (kind === "spatial") return ans === "left" ? "right" : "left";
      const sets = { color: COLORS, count: ["1", "2", "3"], object: SHAPES, ocr: WORDS };
      if (mode === "balanced") return pick(sets[kind].filter((v) => v !== ans));
      if (kind === "color") return pick(OFF);
      if (kind === "count") return String(+ans + 1);
      if (kind === "object") return pick(["star", "hexagon"]);
      const i = Math.floor(rnd() * ans.length);
      return ans.slice(0, i) + pick(["Q", "X", "Z", "J"]) + ans.slice(i + 1);
    }
    const LEAK = {
      color: { balanced: "Both colours come from the four the scenes use. Text alone cannot tell them apart.", biased: "The rejected colour is never drawn. Any model can reject it without looking." },
      count: { balanced: "True and wrong counts come from the same set {1, 2, 3}, so neither number is more likely to be chosen.", biased: "The rejected count is always one too many. The smaller number always wins." },
      object: { balanced: "Both shapes appear in scenes equally often.", biased: "No image contains a star or a hexagon. Those words alone reveal the label." },
      ocr: { balanced: "Both are real words from the same list, so the pair must be read from the image.", biased: "The rejected word is a misspelling. A language model already dislikes it." },
      spatial: { balanced: "There are only two values, so flipping is symmetric.", biased: "Spatial flips cannot be biased this way. This is why the text probe stays near chance on this kind." },
    };
    function draw() {
      s = (seed * 48271) % 2147483647 || 1;
      let sc, q;
      for (let k = 0; k < 40 && !q; k++) { sc = sampleScene(); q = question(sc); }
      sceneHost.innerHTML = "";
      const root = svg("svg", { viewBox: "0 0 512 512", role: "img", "aria-label": "A synthetic scene with coloured shapes and a printed word" }, sceneHost);
      svg("rect", { x: 0, y: 0, width: 512, height: 512, fill: "#fff", stroke: C.rule }, root);
      sc.objs.forEach((o) => {
        const fill = SHAPE_FILL[o.color], f = q.focus.includes(o);
        if (o.shape === "circle") svg("circle", { cx: o.x + o.side / 2, cy: o.y + o.side / 2, r: o.side / 2, fill }, root);
        else if (o.shape === "square") svg("rect", { x: o.x, y: o.y, width: o.side, height: o.side, fill }, root);
        else svg("polygon", { points: `${o.x + o.side / 2},${o.y} ${o.x + o.side},${o.y + o.side} ${o.x},${o.y + o.side}`, fill }, root);
        if (f) svg("rect", { x: o.x - 6, y: o.y - 6, width: o.side + 12, height: o.side + 12, fill: "none", stroke: C.ink, "stroke-width": 2, "stroke-dasharray": "6 4" }, root);
      });
      const wx = sc.wc[1] * 170 + 85, wy = sc.wc[0] * 170 + 85;
      svg("text", { x: wx, y: wy + 20, "text-anchor": "middle", "font-size": 58, "font-weight": 700, fill: "#000", text: sc.word, style: "font-family:Arial,Helvetica,sans-serif" }, root);
      const bad = wrong(q.ans);
      const mark = (t, v, cls) => t.replace(v, `<mark class="${cls}">${v}</mark>`);
      fig.querySelector(".neg-prompt").textContent = q.prompt;
      fig.querySelector(".neg-chosen").innerHTML = mark(q.t(q.ans), q.ans, "neg-true");
      fig.querySelector(".neg-rejected").innerHTML = mark(q.t(bad), bad, "neg-false");
      fig.querySelector(".neg-leak").textContent = LEAK[kind][mode];
      fig.querySelector(".neg-json").textContent = JSON.stringify({ kind, chosen: q.ans, rejected: bad, mode });
    }
    fig.querySelectorAll("[data-kind]").forEach((b) => b.addEventListener("click", () => {
      kind = b.dataset.kind;
      fig.querySelectorAll("[data-kind]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw();
    }));
    fig.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
      mode = b.dataset.mode;
      fig.querySelectorAll("[data-mode]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw();
    }));
    fig.querySelector("[data-new]").addEventListener("click", () => { seed = (seed * 31 + 7) % 99991; draw(); });
    draw();
  })();

  /* ---------- Figure 2: one preference pair as tensors ---------- */
  (function batchFig() {
    const fig = document.getElementById("fig-batch");
    if (!fig) return;
    const W = 960, H = 400;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "One preference pair: input ids, mask and labels, then per-token log-probabilities" }, fig.querySelector(".ct-figure__stage"));
    const world = svg("g", null, root);
    const tb = D.dpoBal && D.dpoBal.tb ? D.dpoBal.tb[0] : null;
    const chosenT = tb ? tb.chosen.tokens : [" The", " circle", " is", " yellow", ".", "<eou>"];
    const rejT = tb ? tb.rejected.tokens : [" The", " circle", " is", " blue", ".", "<eou>"];
    const chosenL = tb ? tb.chosen.logp : chosenT.map(() => 0);
    const rejL = tb ? tb.rejected.logp : rejT.map(() => 0);
    const short = (t) => t.replace("<end_of_utterance>", "<eou>").trim() || "␣";
    const prefix = ["<|im_start|>", "User", ":", "<fake>", "<global>"];
    const middle = ["<fake>"].concat((tb ? tb.prompt : "What color is the circle?").match(/\w+|\?/g).slice(0, 6), ["<eou>", "\\n", "Ass", "istant", ":"]);
    const cw = 34, ch = 30, x0 = 74;
    let cursor = x0;
    const cells = [];
    function cell(g, x, y, w, label, kind) {
      const r = svg("rect", { x, y, width: w - 2, height: ch - 2, rx: 3, fill: C.hi, stroke: C.rule }, g);
      const t = svg("text", { x: x + (w - 2) / 2, y: y + 19, "text-anchor": "middle", "font-size": 9, fill: C.soft, text: label }, g);
      return { r, t, kind, x, y, w };
    }
    const layerIds = svg("g", null, world);
    const rows = [{ y: 110, a: chosenT, name: "chosen", lp: chosenL }, { y: 146, a: rejT, name: "rejected", lp: rejL }];
    const ansX = [];
    rows.forEach((row, ri) => {
      cursor = x0;
      txt(layerIds, 36, row.y + 19, row.name, { "font-size": 10.5, fill: ri === 0 ? C.green : C.red });
      prefix.forEach((t) => { cells.push(cell(layerIds, cursor, row.y, cw, t.length > 6 ? t.slice(0, 5) + "…" : t, "prefix")); cursor += cw; });
      cells.push(cell(layerIds, cursor, row.y, 64, "<image>×64", "image")); cursor += 64;
      middle.forEach((t) => { cells.push(cell(layerIds, cursor, row.y, cw, t.length > 6 ? t.slice(0, 5) + "…" : t, "prefix")); cursor += cw; });
      ansX[0] = cursor;
      row.cells = [];
      row.a.forEach((t) => { const c = cell(layerIds, cursor, row.y, cw + 12, short(t), "answer"); row.cells.push(c); cells.push(c); cursor += cw + 12; });
    });
    txt(layerIds, x0 + 300, 92, "input_ids (2, T): prompt ids + answer ids + right padding", { fill: C.soft, "font-size": 12 });
    const shift = svg("g", { opacity: 0 }, world);
    const ax = rows[0].cells[0].x;
    txt(shift, ax + 130, 72, "labels: −100 everywhere except the answer", { fill: C.orange, "font-size": 12 });
    const imgLayer = svg("g", { opacity: 0 }, world);
    const ic = cells.find((c) => c.kind === "image");
    txt(imgLayer, ic.x + 32, 212, "feats[image_index]", { fill: C.blue, "font-size": 12 });
    txt(imgLayer, ic.x + 32, 232, "(B, 64, 576)", { fill: C.blue, "font-size": 12 });
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) svg("rect", { x: ic.x + 4 + c * 7, y: 240 + r * 7, width: 6, height: 6, fill: C.blue, "fill-opacity": 0.2 + 0.7 * ((r * 7 + c * 13) % 10) / 10 }, imgLayer);
    txt(imgLayer, ic.x + 32, 316, "one encoder pass", { fill: C.soft, "font-size": 11 });
    txt(imgLayer, ic.x + 32, 332, "for both rows", { fill: C.soft, "font-size": 11 });
    const lpLayer = svg("g", { opacity: 0 }, world);
    rows.forEach((row, ri) => {
      row.cells.forEach((c, i) => {
        const y = 200 + ri * 58;
        const v = row.lp[i];
        const h = Math.min(30, Math.abs(v) * 4 + 1);
        svg("rect", { x: c.x + 6, y, width: c.w - 14, height: h, fill: ri === 0 ? C.green : C.red, "fill-opacity": 0.55 }, lpLayer);
        svg("text", { x: c.x + c.w / 2 - 1, y: y - 3, "text-anchor": "middle", "font-size": 9.5, fill: C.ink, text: v.toFixed(2) }, lpLayer);
      });
    });
    const sumC = chosenL.reduce((a, b) => a + b, 0), sumR = rejL.reduce((a, b) => a + b, 0);
    const sumX = rows[0].cells[0].x + rows[0].cells.length * (cw + 12) + 8;
    txt(lpLayer, ax - 44, 214, "log p", { fill: C.green, "font-size": 11 });
    txt(lpLayer, ax - 44, 272, "log p", { fill: C.red, "font-size": 11 });
    const sumLayer = svg("g", { opacity: 0 }, world);
    txt(sumLayer, sumX + 44, 214, "Σ = " + sumC.toFixed(2), { fill: C.green, "font-size": 15, "font-weight": 600 });
    txt(sumLayer, sumX + 44, 272, "Σ = " + sumR.toFixed(2), { fill: C.red, "font-size": 15, "font-weight": 600 });
    txt(sumLayer, sumX + 44, 318, "margin " + (sumC - sumR >= 0 ? "+" : "") + (sumC - sumR).toFixed(2), { fill: C.ink, "font-size": 13 });

    const frames = [
      { note: "One pair becomes two rows that share a prompt. They differ only after <code>Assistant:</code>." },
      { note: "Everything before the answer gets label <code>-100</code>, shown here in grey. Only the answer tokens carry a label, shown in orange." },
      { note: "Both rows look at one image. The encoder runs once and writes 64 feature vectors of size 576 into the placeholders of each row." },
      { note: "Zoom in. Each answer token has one log-probability, read from the logits at the position before it." },
      { note: "The shared tokens score about the same. The slot token decides the pair. These are real numbers from the pretrained model." },
      { note: "Summing gives the sequence log-probability, the only quantity DPO needs. A positive margin means the model already prefers the true answer." },
    ];
    const cams = [{ x: 480, y: 180, w: 960 }, { x: 480, y: 180, w: 960 }, { x: 300, y: 220, w: 760 }, { x: ax + 170, y: 220, w: 560 }, { x: ax + 170, y: 220, w: 560 }, { x: ax + 220, y: 220, w: 640 }];
    function paint(f) {
      cells.forEach((c) => {
        let fill = C.hi, stroke = C.rule, tc = C.soft;
        if (f >= 1) {
          if (c.kind === "answer") { fill = "rgba(214,111,0,0.16)"; stroke = C.orange; tc = C.ink; }
          else { fill = "#eceae4"; tc = C.muted; }
        }
        if (f >= 2 && c.kind === "image") { fill = "rgba(11,94,168,0.16)"; stroke = C.blue; tc = C.blue; }
        c.r.setAttribute("fill", fill); c.r.setAttribute("stroke", stroke); c.t.setAttribute("fill", tc);
      });
    }
    stepper(fig, {
      frames, tween: 1000, hold: 2300,
      render(i, t, from) {
        const a = cams[from], b = cams[i], w = lerp(a.w, b.w, t), h = (w * H) / W;
        root.setAttribute("viewBox", `${lerp(a.x, b.x, t) - w / 2} ${lerp(a.y, b.y, t) - h / 2} ${w} ${h}`);
        paint(t < 0.5 ? from : i);
        const op = (k) => lerp(from >= k ? 1 : 0, i >= k ? 1 : 0, t);
        shift.setAttribute("opacity", op(1));
        imgLayer.setAttribute("opacity", lerp(from === 2 ? 1 : 0, i === 2 ? 1 : 0, t));
        lpLayer.setAttribute("opacity", op(3));
        sumLayer.setAttribute("opacity", op(5));
      },
    });
  })();

  /* ---------- Figure 3: DPO on one real pair, then the training curve ---------- */
  (function dpoFig() {
    const fig = document.getElementById("fig-dpo");
    if (!fig || !D.dpoBal || !D.dpoBal.tb) return;
    const W = 960, H = 390, root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "DPO on one pair and the training curve" }, fig.querySelector(".ct-figure__stage"));
    const sum = (a) => a.reduce((x, y) => x + y, 0);
    // Pick the worked pair with the largest change in margin: the clearest example to read.
    let best = 0, bestD = -1e9;
    D.dpoBal.tb.forEach((p, i) => {
      const q = D.dpoBal.ta[i];
      const d = (sum(q.chosen.logp) - sum(q.rejected.logp)) - (sum(p.chosen.logp) - sum(p.rejected.logp));
      if (d > bestD) { bestD = d; best = i; }
    });
    const tb = D.dpoBal.tb[best], ta = D.dpoBal.ta[best];
    const rc = sum(tb.chosen.logp), rr = sum(tb.rejected.logp), pc = sum(ta.chosen.logp), pr = sum(ta.rejected.logp);
    const beta = 0.1, m = beta * ((pc - rc) - (pr - rr)), loss = Math.log1p(Math.exp(-m));
    const clean = (a) => a.join("").replace("<end_of_utterance>", "").trim();
    txt(root, W / 2, 22, tb.prompt, { "font-size": 14, fill: C.soft });
    const rowsY = [70, 124];
    const lo = Math.min(rc, rr, pc, pr) - 1, hi = Math.max(rc, rr, pc, pr, 0) + 0.2;
    const X = (v) => 240 + ((v - lo) / (hi - lo)) * 400;
    const labels = [["chosen: " + clean(ta.chosen.tokens), C.green], ["rejected: " + clean(ta.rejected.tokens), C.red]];
    labels.forEach(([t, c], k) => {
      const s = t.length > 34 ? t.slice(0, 33) + "…" : t;
      txt(root, 228, rowsY[k] + 5, s, { "text-anchor": "end", "font-size": 10.5, fill: c });
      svg("line", { x1: 240, x2: 640, y1: rowsY[k], y2: rowsY[k], stroke: C.rule }, root);
    });
    const mk = (y, col, filled) => svg("circle", { cx: 0, cy: y, r: 8, fill: filled ? col : C.paper, stroke: col, "stroke-width": 2 }, root);
    const dot = { rc: mk(rowsY[0], C.green, false), rr: mk(rowsY[1], C.red, false), pc: mk(rowsY[0], C.green, true), pr: mk(rowsY[1], C.red, true) };
    const lab = (y) => svg("text", { x: 0, y, "text-anchor": "middle", "font-size": 11, fill: C.ink }, root);
    const lrc = lab(rowsY[0] - 14), lrr = lab(rowsY[1] - 14), lpc = lab(rowsY[0] + 24), lpr = lab(rowsY[1] + 24);
    const side = svg("g", null, root);
    const rt = (y, t, o) => svg("text", Object.assign({ x: 670, y, "font-size": 13, fill: C.ink, text: t }, o || {}), side);
    const s1 = rt(62, "reference: hollow"), s2 = rt(82, "policy: filled");
    const r1 = rt(112, "chosen ratio  " + (pc - rc).toFixed(2)), r2 = rt(132, "rejected ratio  " + (pr - rr).toFixed(2));
    const r3 = rt(158, "β × difference  " + m.toFixed(3), { "font-weight": 600 }), r4 = rt(180, "loss  " + loss.toFixed(3) + "  (from 0.693)", { "font-weight": 600, fill: C.blue });
    // training curves
    const log = D.dpoBal.log;
    const cx0 = 90, cx1 = 900, cy0 = 370, cy1 = 220;
    const maxLoss = 0.72, minLoss = Math.min(...log.map((e) => e.loss), 0.3);
    const cX = (s) => cx0 + (s / (log.length - 1)) * (cx1 - cx0), cY = (v) => cy0 - ((v - minLoss) / (maxLoss - minLoss)) * (cy0 - cy1);
    svg("line", { x1: cx0, x2: cx1, y1: cY(Math.LN2), y2: cY(Math.LN2), stroke: C.rule, "stroke-dasharray": "4 4" }, root);
    txt(root, cx0 - 8, cY(Math.LN2) + 4, "ln 2", { "text-anchor": "end", "font-size": 11, fill: C.muted });
    txt(root, cx0 - 8, cy0, minLoss.toFixed(2), { "text-anchor": "end", "font-size": 11, fill: C.muted });
    svg("line", { x1: cx0, x2: cx1, y1: cy0, y2: cy0, stroke: C.rule }, root);
    txt(root, (cx0 + cx1) / 2, 388, "training step (Apple M5, MPS). Thin line: batch loss. Thick line: moving average.", { "font-size": 11, fill: C.muted });
    const smooth = log.map((e, i) => { const w = log.slice(Math.max(0, i - 9), i + 1); return w.reduce((a, x) => a + x.loss, 0) / w.length; });
    const pRaw = svg("path", { fill: "none", stroke: C.blue, "stroke-opacity": 0.25, "stroke-width": 1 }, root);
    const pSm = svg("path", { fill: "none", stroke: C.blue, "stroke-width": 2.2 }, root);
    const dRaw = log.map((e, i) => (i ? "L" : "M") + cX(i).toFixed(1) + " " + cY(clamp(e.loss, minLoss, maxLoss + 0.02)).toFixed(1)).join(" ");
    const dSm = smooth.map((v, i) => (i ? "L" : "M") + cX(i).toFixed(1) + " " + cY(v).toFixed(1)).join(" ");
    pRaw.setAttribute("d", dRaw); pSm.setAttribute("d", dSm);
    [pRaw, pSm].forEach((p) => { const L = p.getTotalLength(); p.dataset.len = L; p.setAttribute("stroke-dasharray", L + " " + L); });

    const frames = [
      { note: "Before training the policy equals the reference, so each hollow dot and its filled dot coincide. The margin is zero and the loss is ln 2." },
      { note: "After training, the filled chosen dot has moved. A positive log-ratio means the policy raised this answer above the reference." },
      { note: "The filled rejected dot moves as well. DPO only cares about the difference between the two moves." },
      { note: "Multiply the difference by β and pass it through <code>-logsigmoid</code>. This is the whole loss for the pair." },
      { note: "Averaged over batches, the loss falls below ln 2 over the 150 steps. These are real numbers from this run." },
    ];
    stepper(fig, {
      frames, tween: 1100, hold: 2400,
      render(i, t, from) {
        const prog = (fr) => { const f = fr >= 2 ? 1 : fr >= 1 ? 0.5 : 0; return f; };
        const pa = prog(from), pb = prog(i);
        const cgo = clamp(lerp(pa, pb, t) * 2, 0, 1), rgo = clamp(lerp(pa, pb, t) * 2 - 1, 0, 1);
        const ccx = lerp(X(rc), X(pc), cgo), rcx = lerp(X(rr), X(pr), rgo);
        dot.rc.setAttribute("cx", X(rc)); dot.rr.setAttribute("cx", X(rr)); dot.pc.setAttribute("cx", ccx); dot.pr.setAttribute("cx", rcx);
        lrc.setAttribute("x", X(rc)); lrc.textContent = rc.toFixed(2);
        lrr.setAttribute("x", X(rr)); lrr.textContent = rr.toFixed(2);
        lpc.setAttribute("x", ccx); lpc.textContent = lerp(rc, pc, cgo).toFixed(2);
        lpr.setAttribute("x", rcx); lpr.textContent = lerp(rr, pr, rgo).toFixed(2);
        const v = (k) => lerp(from >= k ? 1 : 0, i >= k ? 1 : 0, t);
        r1.setAttribute("opacity", v(1)); r2.setAttribute("opacity", v(2)); r3.setAttribute("opacity", v(3)); r4.setAttribute("opacity", v(3));
        const cv = v(4);
        [pRaw, pSm].forEach((p) => { const L = +p.dataset.len; p.setAttribute("stroke-dashoffset", L * (1 - cv)); });
      },
    });
  })();

  /* ---------- Figure 4: the blind check ---------- */
  (function blindFig() {
    const fig = document.getElementById("fig-blind");
    if (!fig || !D.probe) return;
    const W = 960, H = 380, root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Preference accuracy with and without the image" }, fig.querySelector(".ct-figure__stage"));
    const KINDS = ["color", "count", "object", "ocr", "spatial"];
    const groups = KINDS.concat(["mean"]);
    const x0 = 80, x1 = 930, y0 = 320, y1 = 50;
    const Y = (v) => y0 - v * (y0 - y1);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      svg("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: v === 0.5 ? C.orange : C.grey, "stroke-dasharray": v === 0.5 ? "5 4" : "" }, root);
      txt(root, x0 - 8, Y(v) + 4, pct(v), { "text-anchor": "end", "font-size": 11, fill: v === 0.5 ? C.orange : C.muted });
    });
    const gw = (x1 - x0) / groups.length, bw = gw * 0.34;
    const bars = groups.map((g, k) => {
      const cx = x0 + gw * (k + 0.5);
      txt(root, cx, y0 + 18, g, { "font-size": 12, fill: C.soft });
      const a = svg("rect", { x: cx - bw - 2, y: y0, width: bw, height: 0, fill: C.blue, "fill-opacity": 0.75 }, root);
      const b = svg("rect", { x: cx + 2, y: y0, width: bw, height: 0, fill: C.orange, "fill-opacity": 0.75 }, root);
      const ta = txt(root, cx - bw / 2 - 2, y0, "", { "font-size": 10.5 });
      const tb = txt(root, cx + bw / 2 + 2, y0, "", { "font-size": 10.5 });
      return { a, b, ta, tb };
    });
    const leg = [svg("rect", { x: x0, y: 16, width: 12, height: 12, fill: C.blue, "fill-opacity": 0.75 }, root), svg("rect", { x: 440, y: 16, width: 12, height: 12, fill: C.orange, "fill-opacity": 0.75 }, root)];
    const la = svg("text", { x: x0 + 18, y: 27, "font-size": 12, fill: C.ink }, root), lb = svg("text", { x: 458, y: 27, "font-size": 12, fill: C.ink }, root);
    const gv = (o) => groups.map((g) => (g === "mean" ? (o.mean != null ? o.mean : o.test) : o[g]));
    const P = D.probe;
    const probeV = (m) => groups.map((g) => (g === "mean" ? P[m].test : P[m].per_kind[g]));
    const stages = [];
    stages.push({ A: probeV("balanced"), B: probeV("biased"), la: "Text-only probe on balanced pairs", lb: "Text-only probe on biased pairs", note: "A logistic regression over words, with no pixels. On balanced pairs it stays near 50%. On biased pairs it reads the label from the words. Spatial stays near chance in both, because a flip cannot be biased." });
    if (D.dpoBal) {
      stages.push({ A: gv(D.dpoBal.before.balanced.image), B: gv(D.dpoBal.before.balanced.blind), la: "Pretrained VLM, real image", lb: "Pretrained VLM, grey image", note: "The pretrained VLM on held-out balanced pairs. With a grey image any scorer averages 50% on an exchangeable set, so the gap above the orange line is evidence that it read the pixels." });
      stages.push({ A: gv(D.dpoBal.after.balanced.image), B: gv(D.dpoBal.after.balanced.blind), la: "After DPO on balanced pairs, real image", lb: "After DPO on balanced pairs, grey image", note: "After DPO on balanced negatives, on the same held-out pairs." });
    }
    if (D.dpoBias) {
      stages.push({ A: gv(D.dpoBias.before.biased.blind), B: gv(D.dpoBias.after.biased.blind), la: "Grey image, biased pairs, before DPO on biased pairs", lb: "Grey image, biased pairs, after DPO on biased pairs", note: "The shortcut, shown directly. Biased pairs with a grey image. The model has no visual evidence at all, so everything above 50% is a language prior." });
      stages.push({ A: gv(D.dpoBal.after.balanced.image), B: gv(D.dpoBias.after.balanced.image), la: "Trained on balanced, tested on balanced, real image", lb: "Trained on biased, tested on balanced, real image", note: "Now test both models on the balanced pairs, where a shortcut cannot help." });
    }
    stepper(fig, {
      frames: stages.map((s) => ({ note: s.note })), tween: 1000, hold: 2900,
      render(i, t, from) {
        const a = stages[from], b = stages[i];
        la.textContent = (t < 0.5 ? a : b).la; lb.textContent = (t < 0.5 ? a : b).lb;
        bars.forEach((bar, k) => {
          const va = lerp(a.A[k] || 0, b.A[k] || 0, t), vb = lerp(a.B[k] || 0, b.B[k] || 0, t);
          bar.a.setAttribute("y", Y(va)); bar.a.setAttribute("height", y0 - Y(va));
          bar.b.setAttribute("y", Y(vb)); bar.b.setAttribute("height", y0 - Y(vb));
          bar.ta.setAttribute("y", Y(va) - 4); bar.ta.textContent = pct(va);
          bar.tb.setAttribute("y", Y(vb) - 4); bar.tb.textContent = pct(vb);
        });
      },
    });
  })();

  /* ---------- Figure 5: IoU vs reward explorer ---------- */
  (function iouFig() {
    const fig = document.getElementById("fig-iou");
    if (!fig) return;
    const host = fig.querySelector(".ct-figure__stage");
    const S = 512, root = svg("svg", { viewBox: `-2 -2 ${S + 4} ${S + 4}`, role: "application", "aria-label": "Drag the predicted box. Drag its corner handle to resize." }, host);
    root.style.touchAction = "none";
    root.style.maxWidth = "420px";
    root.style.margin = "0 auto";
    root.style.display = "block";
    svg("rect", { x: 0, y: 0, width: S, height: S, fill: "#fff", stroke: C.rule }, root);
    const gold = { x0: 170, y0: 190, x1: 300, y1: 320 };
    svg("polygon", { points: `${(gold.x0 + gold.x1) / 2},${gold.y0} ${gold.x1},${gold.y1} ${gold.x0},${gold.y1}`, fill: SHAPE_FILL.blue, "fill-opacity": 0.85 }, root);
    svg("rect", { x: gold.x0, y: gold.y0, width: gold.x1 - gold.x0, height: gold.y1 - gold.y0, fill: "none", stroke: C.green, "stroke-width": 3 }, root);
    svg("circle", { cx: (gold.x0 + gold.x1) / 2, cy: (gold.y0 + gold.y1) / 2, r: 4, fill: C.green }, root);
    const inter = svg("rect", { fill: C.orange, "fill-opacity": 0.28 }, root);
    const pred = svg("rect", { fill: "rgba(11,94,168,0.06)", stroke: C.blue, "stroke-width": 3, style: "cursor:move" }, root);
    const handle = svg("rect", { width: 22, height: 22, fill: C.blue, style: "cursor:nwse-resize", tabindex: 0, role: "slider", "aria-label": "Resize predicted box" }, root);
    let box = { x0: 210, y0: 150, x1: 380, y1: 330 };
    const out = (id) => fig.querySelector("#" + id);
    function reward(b) {
      const ix = Math.max(0, Math.min(b.x1, gold.x1) - Math.max(b.x0, gold.x0)), iy = Math.max(0, Math.min(b.y1, gold.y1) - Math.max(b.y0, gold.y0));
      const I = ix * iy, A = (b.x1 - b.x0) * (b.y1 - b.y0), G = (gold.x1 - gold.x0) * (gold.y1 - gold.y0);
      const cx = (gold.x0 + gold.x1) / 2, cy = (gold.y0 + gold.y1) / 2;
      return { iou: I / (A + G - I), dice: (2 * I) / (A + G), center: b.x0 <= cx && cx <= b.x1 && b.y0 <= cy && cy <= b.y1 ? 1 : 0, I };
    }
    function draw() {
      const b = box, r = reward(b);
      pred.setAttribute("x", b.x0); pred.setAttribute("y", b.y0); pred.setAttribute("width", b.x1 - b.x0); pred.setAttribute("height", b.y1 - b.y0);
      handle.setAttribute("x", b.x1 - 11); handle.setAttribute("y", b.y1 - 11);
      const ix0 = Math.max(b.x0, gold.x0), iy0 = Math.max(b.y0, gold.y0), ix1 = Math.min(b.x1, gold.x1), iy1 = Math.min(b.y1, gold.y1);
      if (ix1 > ix0 && iy1 > iy0) { inter.setAttribute("x", ix0); inter.setAttribute("y", iy0); inter.setAttribute("width", ix1 - ix0); inter.setAttribute("height", iy1 - iy0); }
      else { inter.setAttribute("width", 0); inter.setAttribute("height", 0); }
      out("iou-iou").textContent = r.iou.toFixed(3);
      out("iou-dice").textContent = r.dice.toFixed(3);
      out("iou-center").textContent = r.center.toFixed(0);
      out("iou-text").textContent = `answer: ${Math.round(b.x0)} ${Math.round(b.y0)} ${Math.round(b.x1)} ${Math.round(b.y1)}   gold: ${gold.x0} ${gold.y0} ${gold.x1} ${gold.y1}`;
      out("iou-verdict").textContent = r.center === 1 && r.iou < 0.25 ? "The weak reward pays in full for a box that does not localise anything. This is the hack." : r.iou > 0.8 ? "Both rewards agree: this box is good." : r.center === 0 && r.iou > 0.2 ? "IoU gives partial credit. The weak reward gives none, because the gold centre is outside." : "Move the box and watch the two rewards disagree.";
    }
    function pt(e) { const m = root.getScreenCTM().inverse(); const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m); return p; }
    let mode = null, last = null;
    function down(kind) { return (e) => { e.preventDefault(); mode = kind; last = pt(e); root.setPointerCapture(e.pointerId); }; }
    pred.addEventListener("pointerdown", down("move"));
    handle.addEventListener("pointerdown", down("size"));
    root.addEventListener("pointermove", (e) => {
      if (!mode) return;
      const p = pt(e), dx = p.x - last.x, dy = p.y - last.y;
      last = p;
      if (mode === "move") {
        const w = box.x1 - box.x0, h = box.y1 - box.y0;
        box.x0 = clamp(box.x0 + dx, 0, S - w); box.y0 = clamp(box.y0 + dy, 0, S - h); box.x1 = box.x0 + w; box.y1 = box.y0 + h;
      } else {
        box.x1 = clamp(box.x1 + dx, box.x0 + 12, S); box.y1 = clamp(box.y1 + dy, box.y0 + 12, S);
      }
      draw();
    });
    ["pointerup", "pointercancel"].forEach((n) => root.addEventListener(n, () => { mode = null; }));
    handle.addEventListener("keydown", (e) => {
      const d = { ArrowRight: [8, 0], ArrowLeft: [-8, 0], ArrowDown: [0, 8], ArrowUp: [0, -8] }[e.key];
      if (!d) return;
      e.preventDefault();
      box.x1 = clamp(box.x1 + d[0], box.x0 + 12, S); box.y1 = clamp(box.y1 + d[1], box.y0 + 12, S);
      draw();
    });
    fig.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      box = { tight: { x0: 170, y0: 190, x1: 300, y1: 320 }, whole: { x0: 0, y0: 0, x1: 512, y1: 512 }, tiny: { x0: 226, y0: 246, x1: 244, y1: 264 }, off: { x0: 20, y0: 20, x1: 150, y1: 150 } }[b.dataset.preset];
      box = Object.assign({}, box);
      draw();
    }));
    draw();
  })();

  /* ---------- Figure 6: one GRPO group ---------- */
  (function grpoFig() {
    const fig = document.getElementById("fig-grpo");
    if (!fig || !D.rollouts) return;
    const G0 = D.rollouts.find((g) => { const m = g.rewards.reduce((a, b) => a + b, 0) / g.rewards.length; return m > 0 && m < 1; }) || D.rollouts[0];
    const G = G0.rewards.length, W = 960, H = 420;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "One image, eight rollouts, rewards and relative advantages" }, fig.querySelector(".ct-figure__stage"));
    const img = svg("g", null, root);
    svg("rect", { x: 30, y: 90, width: 190, height: 190, fill: "#fff", stroke: C.rule }, img);
    const sc = 190 / 512;
    G0.objects.forEach(([shape, color, b]) => {
      const x = 30 + b[0] * sc, y = 90 + b[1] * sc, w = (b[2] - b[0]) * sc, f = SHAPE_FILL[color];
      if (shape === "circle") svg("circle", { cx: x + w / 2, cy: y + w / 2, r: w / 2, fill: f }, img);
      else if (shape === "square") svg("rect", { x, y, width: w, height: w, fill: f }, img);
      else svg("polygon", { points: `${x + w / 2},${y} ${x + w},${y + w} ${x},${y + w}`, fill: f }, img);
    });
    txt(root, 125, 72, "image + prompt", { "font-size": 13, fill: C.soft });
    const prompt = G0.prompt.length > 34 ? G0.prompt.slice(0, 34) : G0.prompt;
    txt(root, 125, 304, prompt, { "font-size": 10.5, fill: C.soft });
    txt(root, 125, 322, "Answer with a number.", { "font-size": 10.5, fill: C.soft });
    txt(root, 125, 344, "truth = " + G0.truth, { "font-size": 13, fill: C.green, "font-weight": 600 });
    const rowH = 38, ry = (i) => 60 + i * rowH;
    const X = { text: 330, r: 470, a: 650 };
    const hd = [txt(root, X.text, 40, "G sampled answers", { fill: C.soft }), txt(root, X.r, 40, "reward", { fill: C.soft }), txt(root, X.a, 40, "advantage", { fill: C.soft }), txt(root, 860, 40, "update", { fill: C.soft })];
    const mean = G0.rewards.reduce((a, b) => a + b, 0) / G;
    const std = Math.sqrt(G0.rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / (G - 1));
    const rows = G0.texts.map((t, i) => {
      const y = ry(i), g = svg("g", { opacity: 0 }, root);
      svg("rect", { x: X.text - 60, y: y - 15, width: 120, height: 28, rx: 3, fill: C.hi, stroke: C.rule }, g);
      txt(g, X.text, y + 4, JSON.stringify(t.trim()), { "font-size": 13 });
      const rg = svg("g", { opacity: 0 }, root);
      const ok = G0.rewards[i] > 0.5;
      svg("circle", { cx: X.r, cy: y, r: 11, fill: ok ? C.green : C.hi, stroke: ok ? C.green : C.rule }, rg);
      txt(rg, X.r, y + 4, G0.rewards[i].toFixed(0), { "font-size": 12, fill: ok ? "#fff" : C.soft });
      const a = G0.advantages[i], ag = svg("g", { opacity: 0 }, root);
      const w = Math.abs(a) * 40;
      svg("rect", { x: a >= 0 ? X.a : X.a - w, y: y - 9, width: w, height: 18, fill: a >= 0 ? C.green : C.red, "fill-opacity": 0.6 }, ag);
      txt(ag, a >= 0 ? X.a + w + 26 : X.a + 26, y + 4, (a >= 0 ? "+" : "") + a.toFixed(2), { "font-size": 11.5, fill: C.ink });
      const ug = svg("g", { opacity: 0 }, root);
      txt(ug, 860, y + 5, a > 0 ? "log π ↑" : a < 0 ? "log π ↓" : "no change", { "font-size": 12.5, fill: a > 0 ? C.green : a < 0 ? C.red : C.muted, "font-weight": 600 });
      return { g, rg, ag, ug };
    });
    svg("line", { x1: X.a, x2: X.a, y1: 44, y2: ry(G - 1) + 18, stroke: C.rule }, root);
    const stat = svg("g", { opacity: 0 }, root);
    txt(stat, X.r, ry(G - 1) + 46, `mean r = ${mean.toFixed(3)}   std r = ${std.toFixed(3)}`, { "font-size": 12.5, fill: C.orange });
    const flat = svg("g", { opacity: 0 }, root);
    const F = D.rollouts.find((g) => g.rewards.every((v) => v === g.rewards[0]));
    if (F) txt(flat, 620, ry(G - 1) + 66, `Another image in the same run: all ${G} answers were ${JSON.stringify(F.texts[0].trim())}, truth ${F.truth}. Every advantage is 0.`, { "font-size": 12, fill: C.ink });
    const frames = [
      { note: "One image and one prompt. Eight samples from the policy at temperature 1. No top-k and no top-p, because we want the policy's own distribution." },
      { note: "The reward is read off the scene. Exact match against a count we know, so no human and no learned judge." },
      { note: "Advantage is the reward minus the group mean, divided by the group standard deviation. Only the spread within this image matters." },
      { note: "Tokens of a right answer get pushed up. Tokens of a wrong answer get pushed down, by less, because right answers are rare here." },
      { note: "A group where every answer is wrong in the same way teaches nothing. This is a real failure mode of hard tasks: no spread, no gradient." },
    ];
    stepper(fig, {
      frames, tween: 1000, hold: 2500,
      render(i, t, from) {
        const v = (k) => lerp(from >= k ? 1 : 0, i >= k ? 1 : 0, t);
        rows.forEach((r, n) => {
          const stagger = (x) => clamp(x * (G + 2) - n * 0.9, 0, 1);
          r.g.setAttribute("opacity", from === 0 && i === 0 ? stagger(t) : 1);
          r.rg.setAttribute("opacity", from < 1 && i === 1 ? stagger(t) : v(1));
          r.ag.setAttribute("opacity", from < 2 && i === 2 ? stagger(t) : v(2));
          r.ug.setAttribute("opacity", from < 3 && i === 3 ? stagger(t) : v(3));
        });
        stat.setAttribute("opacity", v(2));
        flat.setAttribute("opacity", v(4));
      },
    });
  })();

  /* ---------- Figure 7: freeze ablation ---------- */
  (function freezeFig() {
    const fig = document.getElementById("fig-freeze");
    if (!fig || !D.grpo || !D.grpo.llm) return;
    const W = 960, H = 400, root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Which parts of the VLM train in each configuration, and the result" }, fig.querySelector(".ct-figure__stage"));
    const parts = [{ k: "vision", n: "Vision encoder", p: "86.4M" }, { k: "connector", n: "Connector", p: "7.1M" }, { k: "llm", n: "Decoder", p: "134.6M" }];
    const blocks = parts.map((p, i) => {
      const x = 30, y = 70 + i * 90;
      const r = svg("rect", { x, y, width: 280, height: 66, rx: 5, fill: C.hi, stroke: C.rule, "stroke-width": 1.5 }, root);
      txt(root, x + 14, y + 25, p.n, { "text-anchor": "start", "font-size": 15, "font-weight": 600 });
      txt(root, x + 14, y + 48, p.p + " weights", { "text-anchor": "start", "font-size": 11, fill: C.muted });
      const st = svg("text", { x: x + 268, y: y + 38, "text-anchor": "end", "font-size": 12, fill: C.muted }, root);
      return { r, st, k: p.k };
    });
    svg("path", { d: "M170 136 V 160 M170 226 V 250", stroke: C.muted, "stroke-width": 1.5, fill: "none" }, root);
    const title = svg("text", { x: 30, y: 40, "font-size": 18, "font-weight": 700, class: "disp", fill: C.ink }, root);
    const modes = [["llm", "LLM only"], ["projector", "Projector + LLM"], ["all", "All three"]];
    const M = modes.map(([k]) => D.grpo[k]).filter(Boolean);
    const base = D.grpo.llm.before;
    const met = [["Counting from pixels", (r) => r.count], ["Counting from text", (r) => r.text_count], ["Other visual skills", (r) => r.other.mean]];
    const x0 = 360, x1 = 930, Y0 = 340, Y1 = 90;
    const Y = (v) => Y0 - v * (Y0 - Y1);
    [0, 0.5, 1].forEach((v) => { svg("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: C.grey }, root); txt(root, x0 - 6, Y(v) + 4, pct(v), { "text-anchor": "end", "font-size": 11, fill: C.muted }); });
    const gw = (x1 - x0) / 3, bw = 40;
    const bars = met.map(([n], k) => {
      const cx = x0 + gw * (k + 0.5);
      txt(root, cx, Y0 + 20, n, { "font-size": 12, fill: C.soft });
      const b0 = svg("rect", { x: cx - bw - 3, width: bw, fill: C.muted, "fill-opacity": 0.45 }, root);
      const b1 = svg("rect", { x: cx + 3, width: bw, fill: C.blue, "fill-opacity": 0.8 }, root);
      const t0 = txt(root, cx - bw / 2 - 3, 0, "", { "font-size": 11 });
      const t1 = txt(root, cx + bw / 2 + 3, 0, "", { "font-size": 11 });
      return { b0, b1, t0, t1, f: met[k][1] };
    });
    txt(root, x0, 60, "grey: pretrained", { "text-anchor": "start", "font-size": 11.5, fill: C.muted });
    txt(root, x0 + 130, 60, "blue: after 60 GRPO steps", { "text-anchor": "start", "font-size": 11.5, fill: C.blue });
    const fmt = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : "frozen");
    const frames = M.map((r, i) => {
      const [k, n] = modes[i];
      return { note: `<b>${n}</b>: ${fmt(r.trainable.total)} trainable values. Counting from pixels ${pct(base.count)} → ${pct(r.after.count)}. Counting from text ${pct(base.text_count)} → ${pct(r.after.text_count)}. Run time ${Math.round(r.seconds)} s on MPS.`, title: n, r };
    });
    stepper(fig, {
      frames, tween: 1000, hold: 3200,
      render(i, t, from) {
        const a = frames[from].r, b = frames[i].r;
        title.textContent = frames[t < 0.5 ? from : i].title;
        blocks.forEach((bl) => {
          const on = (t < 0.5 ? a : b).trainable[bl.k] > 0;
          bl.r.setAttribute("fill", on ? "rgba(47,143,91,0.14)" : C.hi);
          bl.r.setAttribute("stroke", on ? C.green : C.rule);
          bl.st.textContent = on ? "trains: " + fmt((t < 0.5 ? a : b).trainable[bl.k]) : "frozen";
          bl.st.setAttribute("fill", on ? C.green : C.muted);
        });
        bars.forEach((B) => {
          const v0 = B.f(base), v = lerp(B.f(a.after), B.f(b.after), t);
          B.b0.setAttribute("y", Y(v0)); B.b0.setAttribute("height", Y0 - Y(v0));
          B.b1.setAttribute("y", Y(v)); B.b1.setAttribute("height", Y0 - Y(v));
          B.t0.setAttribute("y", Y(v0) - 4); B.t0.textContent = pct(v0);
          B.t1.setAttribute("y", Y(v) - 4); B.t1.textContent = pct(v);
        });
      },
    });
  })();

  /* ---------- Figure 8: reward hacking curves ---------- */
  (function hackFig() {
    const fig = document.getElementById("fig-hack");
    if (!fig || !D.hack || !D.hack.recall) return;
    const W = 960, H = 410, root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Weak reward and honest reward during a hackable run" }, fig.querySelector(".ct-figure__stage"));
    const smoothLog = (log, k) => log.map((e, i) => { const w = log.slice(Math.max(0, i - 7), i + 1); return w.reduce((a, x) => a + x[k], 0) / w.length; });
    const R = D.hack.recall.log, F = D.hack.f1 ? D.hack.f1.log : null;
    const N = R.length;
    const panels = [{ x: 60, title: "Reward signals (recall vs F1)", keys: [["reward", "weak: recall", C.orange], ["strict", "honest: F1", C.blue]], lo: 0, hi: 1 },
      { x: 520, title: "What the model says", keys: [["n_colors", "colours named", C.ink], ["hallucinated", "colours not in the image", C.red]], lo: 0, hi: null }];
    const pw = 390, py0 = 300, py1 = 60;
    panels.forEach((p) => {
      p.hi = p.hi == null ? Math.max(2.5, ...p.keys.map(([k]) => Math.max(...smoothLog(R, k).concat(F ? smoothLog(F, k) : [])))) * 1.1 : p.hi;
      txt(root, p.x, 34, p.title, { "text-anchor": "start", "font-size": 13.5, "font-weight": 600 });
      svg("line", { x1: p.x, x2: p.x + pw, y1: py0, y2: py0, stroke: C.rule }, root);
      svg("line", { x1: p.x, x2: p.x, y1: py1, y2: py0, stroke: C.rule }, root);
      [p.lo, (p.lo + p.hi) / 2, p.hi].forEach((v) => txt(root, p.x - 6, py0 - ((v - p.lo) / (p.hi - p.lo)) * (py0 - py1) + 4, v.toFixed(p.hi > 1 ? 1 : 1), { "text-anchor": "end", "font-size": 10.5, fill: C.muted }));
      p.X = (s) => p.x + (s / (N - 1)) * pw;
      p.Y = (v) => py0 - ((v - p.lo) / (p.hi - p.lo)) * (py0 - py1);
      p.paths = p.keys.map(([k, name, col], j) => {
        const sm = smoothLog(R, k);
        const d = sm.map((v, i) => (i ? "L" : "M") + p.X(i).toFixed(1) + " " + p.Y(v).toFixed(1)).join(" ");
        const path = svg("path", { d, fill: "none", stroke: col, "stroke-width": 2.2 }, root);
        const L = path.getTotalLength();
        path.setAttribute("stroke-dasharray", L + " " + L);
        const ly = 48 + j * 0;
        txt(root, p.x + 8 + j * 190, ly, name, { "text-anchor": "start", "font-size": 11, fill: col });
        let fx = null;
        if (F) {
          const smf = smoothLog(F, k);
          fx = svg("path", { d: smf.map((v, i) => (i ? "L" : "M") + p.X(i).toFixed(1) + " " + p.Y(v).toFixed(1)).join(" "), fill: "none", stroke: col, "stroke-width": 1.5, "stroke-dasharray": "5 4", opacity: 0 }, root);
        }
        return { path, L, fx };
      });
      p.cursor = svg("line", { y1: py1, y2: py0, stroke: C.muted, "stroke-dasharray": "2 3" }, root);
      txt(root, p.x + pw / 2, py0 + 20, "GRPO step", { "font-size": 11, fill: C.muted });
    });
    const sample = svg("text", { x: W / 2, y: 356, "text-anchor": "middle", "font-size": 13, fill: C.ink }, root);
    const sampleL = svg("text", { x: W / 2, y: 336, "text-anchor": "middle", "font-size": 11, fill: C.muted }, root);
    const fixedLeg = svg("text", { x: W / 2, y: 386, "text-anchor": "middle", "font-size": 12, fill: C.soft, opacity: 0, text: "Dashed lines: the same run with the reward changed to F1." }, root);
    const stops = [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => Math.round(f * (N - 1)));
    const frames = stops.map((s) => ({ note: `Step ${s}. Weak reward ${smoothLog(R, "reward")[s].toFixed(2)}, honest F1 ${smoothLog(R, "strict")[s].toFixed(2)}, colours named ${smoothLog(R, "n_colors")[s].toFixed(2)}, colours not in the image ${smoothLog(R, "hallucinated")[s].toFixed(2)}.`, s }));
    if (F) frames.push({ note: "Dashed lines add the same run with F1 as the reward. Compare the two runs in the table below rather than a single curve: each line is a short, noisy run.", s: N - 1, fixed: true });
    stepper(fig, {
      frames, tween: 1200, hold: 2200,
      render(i, t, from) {
        const s = lerp(frames[from].s, frames[i].s, t);
        panels.forEach((p) => {
          p.paths.forEach((q) => {
            q.path.setAttribute("stroke-dashoffset", q.L * (1 - s / (N - 1)));
            if (q.fx) q.fx.setAttribute("opacity", lerp(frames[from].fixed ? 1 : 0, frames[i].fixed ? 1 : 0, t));
          });
          p.cursor.setAttribute("x1", p.X(s)); p.cursor.setAttribute("x2", p.X(s));
        });
        fixedLeg.setAttribute("opacity", lerp(frames[from].fixed ? 1 : 0, frames[i].fixed ? 1 : 0, t));
        const shown = R[Math.round(s)];
        sampleL.textContent = `One rollout sampled at step ${Math.round(s)}:`;
        const text = JSON.stringify((shown.s || "").trim());
        sample.textContent = text.length > 110 ? text.slice(0, 110) + "…" : text;
      },
    });
  })();
})();
