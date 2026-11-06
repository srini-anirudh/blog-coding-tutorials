(function () {
  "use strict";
  const { svg, lerp, stepper } = window.CT;
  const D = window.C12_DATA;
  const C = { ink: "#14171a", soft: "#474e53", muted: "#7b8388", rule: "#cac6bd", ruleSoft: "#dedad2", paper: "#f7f6f3", hi: "#fbfaf8", blue: "#0b5ea8", orange: "#d66f00", green: "#2f8f5b", red: "#b8432f", purple: "#7b4fa8" };
  const SERIES = [C.blue, C.orange, C.green, C.purple];
  const DASH = ["", "7 4", "2 3", "10 3 2 3"];
  const pct = (v) => Math.round(100 * v) + "%";
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const clamp01 = (t) => Math.max(0, Math.min(1, t));
  function T(p, x, y, text, o) { return svg("text", Object.assign({ x, y, "font-size": 13, fill: C.ink, text }, o || {}), p); }
  function B(p, x, y, w, h, o) { return svg("rect", Object.assign({ x, y, width: w, height: h, rx: 3, fill: C.hi, stroke: C.rule }, o || {}), p); }

  /* ---------- reusable line chart: one axis, legend, direct labels, crosshair tooltip ---------- */
  function lineChart(host, spec) {
    host.innerHTML = "";
    const W = 860, H = 340, L = 56, R = 150, TOP = 16, BOT = 44;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.label || "Line chart" }, host);
    root.style.minWidth = "560px";
    const all = spec.series.flatMap((s) => s.points);
    const xmax = spec.xmax != null ? spec.xmax : Math.max(...all.map((p) => p[0]), 1);
    const xmin = spec.xmin != null ? spec.xmin : 0;
    const ymin = spec.ymin != null ? spec.ymin : 0;
    const ymax = spec.ymax != null ? spec.ymax : Math.max(...all.map((p) => p[1]), 1);
    const X = (v) => L + ((v - xmin) / Math.max(xmax - xmin, 1e-6)) * (W - L - R);
    const Y = (v) => H - BOT - ((v - ymin) / Math.max(ymax - ymin, 1e-6)) * (H - TOP - BOT);
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const v = ymin + ((ymax - ymin) * i) / ticks;
      svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: C.ruleSoft }, root);
      T(root, L - 8, Y(v) + 4, spec.percent ? pct(v) : v.toFixed(1), { "text-anchor": "end", "font-size": 11, fill: C.muted });
    }
    const xt = 5;
    for (let i = 0; i <= xt; i++) {
      const v = xmin + ((xmax - xmin) * i) / xt;
      T(root, X(v), H - BOT + 18, Math.round(v), { "text-anchor": "middle", "font-size": 11, fill: C.muted });
    }
    T(root, (L + W - R) / 2, H - 6, spec.xlabel || "iteration", { "text-anchor": "middle", "font-size": 11, fill: C.soft });
    T(root, 12, TOP + 2, spec.ylabel || "", { "font-size": 11, fill: C.soft });
    const labels = [];
    spec.series.forEach((s, k) => {
      const col = s.color || SERIES[k % 4];
      const d = s.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
      svg("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-dasharray": s.dash != null ? s.dash : DASH[k % 4], "stroke-linejoin": "round", "stroke-linecap": "round" }, root);
      if (s.points.length) {
        const p = s.points[s.points.length - 1];
        svg("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: col, stroke: C.paper, "stroke-width": 2 }, root);
        labels.push({ y: Y(p[1]), name: s.name, val: spec.percent ? pct(p[1]) : p[1].toFixed(2) });
      }
    });
    labels.sort((a, b) => a.y - b.y);
    labels.forEach((l, i) => { if (i && l.y < labels[i - 1].y + 15) l.y = labels[i - 1].y + 15; });
    labels.forEach((l) => T(root, W - R + 10, l.y + 4, `${l.name} ${l.val}`, { "font-size": 11, fill: C.ink }));
    const cross = svg("line", { y1: TOP, y2: H - BOT, stroke: C.muted, "stroke-width": 1, opacity: 0 }, root);
    const legend = document.createElement("div");
    legend.className = "c12-legend";
    legend.innerHTML = spec.series.map((s, k) => `<span><i style="background:${s.color || SERIES[k % 4]}"></i>${esc(s.name)}</span>`).join("");
    host.appendChild(legend);
    const tip = document.createElement("div");
    tip.className = "c12-tip";
    tip.setAttribute("aria-live", "polite");
    host.appendChild(tip);
    const hit = svg("rect", { x: L, y: TOP, width: W - L - R, height: H - TOP - BOT, fill: "transparent" }, root);
    function move(e) {
      const pt = root.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const q = pt.matrixTransform(root.getScreenCTM().inverse());
      const xv = xmin + ((q.x - L) / (W - L - R)) * (xmax - xmin);
      cross.setAttribute("x1", q.x); cross.setAttribute("x2", q.x); cross.setAttribute("opacity", 0.6);
      const rows = spec.series.map((s) => {
        if (!s.points.length) return "";
        let best = s.points[0];
        s.points.forEach((p) => { if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p; });
        return `${esc(s.name)}: <b>${spec.percent ? pct(best[1]) : best[1].toFixed(2)}</b>`;
      });
      tip.innerHTML = `${spec.xlabel || "iteration"} ${Math.round(xv)} &middot; ` + rows.join(" &middot; ");
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); tip.innerHTML = spec.tip || "Move across the chart to read values."; });
    tip.innerHTML = spec.tip || "Move across the chart to read values.";
    return root;
  }

  /* ---------- Figure 1: the environment loop ---------- */
  (function envFigure() {
    const fig = document.getElementById("fig-env");
    if (!fig) return;
    const demo = D.demo;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 450;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "One episode of the filesystem environment" }, stage);
    root.style.minWidth = "640px";
    const paths = Object.keys(demo.files);
    T(root, 20, 26, "state: the filesystem", { "font-size": 12, fill: C.muted });
    const cards = {};
    paths.forEach((p, i) => {
      const x = 20 + (i % 2) * 215, y = 40 + Math.floor(i / 2) * 134;
      const g = svg("g", null, root);
      const box = B(g, x, y, 200, 122);
      T(g, x + 12, y + 22, p, { "font-size": 13.5, "font-weight": 600 });
      const lines = Object.keys(demo.files[p]).map((k, j) => {
        const bg = svg("rect", { x: x + 8, y: y + 33 + j * 27, width: 184, height: 24, rx: 2, fill: C.orange, "fill-opacity": 0 }, g);
        const t = T(g, x + 14, y + 50 + j * 27, `${k} = ${demo.files[p][k]}`, { "font-size": 13, fill: C.soft });
        return { k, bg, t };
      });
      cards[p] = { box, lines };
    });
    T(root, 470, 26, "task", { "font-size": 12, fill: C.muted });
    B(root, 470, 34, 470, 44, { fill: "#fff" });
    T(root, 484, 62, demo.task, { "font-size": 14.5, "font-weight": 600 });
    T(root, 470, 104, "history: action -> tool output", { "font-size": 12, fill: C.muted });
    const rows = demo.steps.map((s, i) => {
      const g = svg("g", { opacity: 0 }, root);
      const y = 112 + i * 56;
      B(g, 470, y, 470, 48, { fill: "#fff" });
      T(g, 482, y + 20, `a${i}  ${s.action}`, { "font-size": 13.5, fill: C.blue, "font-weight": 600 });
      T(g, 482, y + 39, `o${i}  ${s.observation}`, { "font-size": 12.5, fill: C.soft });
      return g;
    });
    const rg = svg("g", { opacity: 0 }, root);
    B(rg, 470, 340, 470, 64, { fill: "#fff", stroke: C.green, "stroke-width": 1.5 });
    T(rg, 484, 366, "verifier: final files == target files", { "font-size": 13.5 });
    T(rg, 484, 390, "reward R = 1, given once, on the last step", { "font-size": 13.5, fill: C.green, "font-weight": 600 });
    const tg = svg("g", { opacity: 0 }, root);
    T(tg, 20, 436, "tau = (s0, a0, r0, s1, a1, r1, ..., sT)  every field is recorded", { "font-size": 13.5, fill: C.ink });
    const s1 = demo.steps;
    const srcPath = s1[0].observation.split(" ")[1];
    const editParts = s1[2].action.split(" ");
    const frames = [
      { note: "<code>reset()</code> copies the files and returns the task as text. The agent sees only this sentence. It does not know which file holds <code>" + esc(demo.source) + "</code>." },
      { note: "<code>" + esc(s1[0].action) + "</code> asks where the key lives. The tool answers with a path, which must become an argument of the next call." },
      { note: "<code>" + esc(s1[1].action) + "</code> prints the file. The value of the source key is the input to the next call. This is what makes level 3 a dependent-tools task." },
      { note: "<code>" + esc(s1[2].action) + "</code> changes the state. Only <code>edit</code> can change the files. The reward has not moved: it is still zero." },
      { note: "<code>submit</code> ends the episode. The environment compares the files with the target. Any wrong edit, anywhere, gives zero." },
      { note: "We keep every action, every observation and the reward. Later sections compute log-probabilities and advantages over exactly this record." },
    ];
    stepper(fig, {
      frames, tween: 800, hold: 2400,
      render(i, t, from) {
        const vis = (k, f) => (k <= f ? 1 : 0);
        rows.forEach((g, k) => g.setAttribute("opacity", lerp(vis(k + 1, from), vis(k + 1, i), t)));
        rg.setAttribute("opacity", lerp(from >= 4 ? 1 : 0, i >= 4 ? 1 : 0, t));
        tg.setAttribute("opacity", lerp(from >= 5 ? 1 : 0, i >= 5 ? 1 : 0, t));
        const show = t < 0.5 ? from : i;
        paths.forEach((p) => {
          const c = cards[p];
          c.box.setAttribute("stroke", (show === 1 && p === srcPath) || (show === 2 && p === srcPath) || (show >= 3 && p === editParts[1]) ? C.orange : C.rule);
          c.box.setAttribute("stroke-width", (show >= 1 && (p === srcPath || (show >= 3 && p === editParts[1]))) ? 2 : 1);
          c.lines.forEach((l) => {
            const isSrc = p === srcPath && l.k === demo.source && show === 2;
            const isEdit = p === editParts[1] && l.k === editParts[2] && show >= 3;
            l.bg.setAttribute("fill-opacity", isSrc || isEdit ? 0.18 : 0);
            l.t.textContent = isEdit ? `${l.k} = ${editParts[3]}` : `${l.k} = ${demo.files[p][l.k]}`;
            l.t.setAttribute("fill", isSrc || isEdit ? C.ink : C.soft);
          });
        });
      },
    });
  })();

  /* ---------- Figure 2: one episode as one token sequence ---------- */
  (function tapeFigure() {
    const fig = document.getElementById("fig-tape");
    if (!fig) return;
    const tape = D.tape;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 330;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "Trajectory token tape" }, stage);
    root.style.minWidth = "640px";
    const w = svg("g", null, root);
    const CW = 66, X0 = 110, n = tape.tokens.length;
    const rowY = { tok: 40, id: 88, mask: 130, lp: 172, adv: 214, pg: 256 };
    const rowName = { tok: "token", id: "id", mask: "action_mask", lp: "log pi", adv: "A (adv)", pg: "A x log pi" };
    const rowsG = {};
    Object.keys(rowY).forEach((r) => {
      rowsG[r] = svg("g", { opacity: 0 }, w);
      T(rowsG[r], X0 - 12, rowY[r] + 24, rowName[r], { "text-anchor": "end", "font-size": 11.5, fill: C.muted });
    });
    const A = tape.adv;
    tape.tokens.forEach((tok, i) => {
      const x = X0 + i * CW, on = tape.mask[i];
      B(rowsG.tok, x, rowY.tok, CW - 4, 38, { fill: on ? "rgba(11,94,168,0.14)" : C.hi, stroke: on ? C.blue : C.rule });
      T(rowsG.tok, x + (CW - 4) / 2, rowY.tok + 24, tok, { "text-anchor": "middle", "font-size": tok.length > 8 ? 9.5 : 11.5 });
      T(rowsG.id, x + (CW - 4) / 2, rowY.id + 22, tape.ids[i], { "text-anchor": "middle", "font-size": 11.5, fill: C.soft });
      B(rowsG.mask, x, rowY.mask, CW - 4, 30, { fill: on ? C.blue : C.hi, stroke: on ? C.blue : C.ruleSoft });
      T(rowsG.mask, x + (CW - 4) / 2, rowY.mask + 20, on, { "text-anchor": "middle", "font-size": 11.5, fill: on ? "#fff" : C.muted });
      if (on) {
        T(rowsG.lp, x + (CW - 4) / 2, rowY.lp + 22, tape.logp[i].toFixed(2), { "text-anchor": "middle", "font-size": 11.5, fill: C.ink });
        B(rowsG.adv, x, rowY.adv, CW - 4, 30, { fill: "rgba(214,111,0,0.18)", stroke: C.orange });
        T(rowsG.adv, x + (CW - 4) / 2, rowY.adv + 20, "+" + A.toFixed(2), { "text-anchor": "middle", "font-size": 11.5 });
        T(rowsG.pg, x + (CW - 4) / 2, rowY.pg + 22, (A * tape.logp[i]).toFixed(2), { "text-anchor": "middle", "font-size": 11.5, fill: C.soft });
      }
    });
    const first = tape.states[0];
    const arrow = svg("g", { opacity: 0 }, w);
    const ax = X0 + first * CW + (CW - 4) / 2, bx = ax + CW;
    svg("path", { d: `M${ax} ${rowY.tok - 2} C ${ax} ${rowY.tok - 26}, ${bx} ${rowY.tok - 26}, ${bx} ${rowY.tok - 2}`, fill: "none", stroke: C.orange, "stroke-width": 2 }, arrow);
    T(arrow, ax + CW / 2, rowY.tok - 26, "logits at <act> score the next token", { "text-anchor": "middle", "font-size": 11.5, fill: C.ink });
    const sum = svg("g", { opacity: 0 }, root);
    B(sum, 300, 288, 620, 32, { fill: "#fff", stroke: C.orange });
    const total = tape.mask.reduce((a, m, i) => a + (m ? A * tape.logp[i] : 0), 0) / tape.mask.reduce((a, m) => a + m, 0);
    T(sum, 312, 309, `surrogate = mean over ${tape.mask.reduce((a, m) => a + m, 0)} action tokens = ${total.toFixed(3)}`, { "font-size": 13 });
    const cams = [
      { x: 480, w: 960 }, { x: 900, w: 1700 }, { x: 900, w: 1700 }, { x: X0 + first * CW + 60, w: 760 },
      { x: 900, w: 1700 }, { x: 900, w: 1700 }, { x: 900, w: 1700 },
    ];
    const shown = [["tok", "id"], ["tok", "id"], ["tok", "id", "mask"], ["tok", "id", "mask"], ["tok", "id", "mask", "lp"], ["tok", "id", "mask", "lp", "adv"], ["tok", "id", "mask", "lp", "adv", "pg"]];
    const frames = [
      { note: "The prompt: <code>&lt;bos&gt;</code>, the task, then <code>&lt;act&gt;</code>, which means your turn. The policy reads this and starts generating." },
      { note: "The whole episode is one sequence (B = 1, T = " + n + "). Blue tokens were sampled by the policy. The rest were written by the environment." },
      { note: "<code>action_mask</code> is 1 on sampled tokens. Same idea as the assistant-only loss mask in C6: the model is never scored on what the environment wrote." },
      { note: "The logits at position t predict the token at t + 1. So one forward pass over the sequence gives every log-probability we need. No replay of the rollout is required." },
      { note: "<code>token_logprobs</code> gathers log &pi; of each taken token. These numbers come from the cloned policy on this demonstration, on the Apple M5 CPU." },
      { note: "Trajectory-level credit: this episode's advantage A = +" + A.toFixed(2) + " lands on every action token. It is an example value from a group with one winner of eight." },
      { note: "The loss weights each log-probability by A and averages over action tokens. Environment tokens contribute nothing." },
    ];
    stepper(fig, {
      frames, tween: 1000, hold: 2300,
      render(i, t, from) {
        const a = cams[from], b = cams[i];
        const vw = lerp(a.w, b.w, t), vx = lerp(a.x, b.x, t);
        const s = VW / vw;
        w.setAttribute("transform", `translate(${VW / 2 - vx * s} ${(VH - 288 * s) / 2 - 6}) scale(${s})`);
        Object.keys(rowsG).forEach((r) => {
          const pa = shown[from].includes(r) ? 1 : 0, pb = shown[i].includes(r) ? 1 : 0;
          rowsG[r].setAttribute("opacity", lerp(pa, pb, t));
        });
        arrow.setAttribute("opacity", lerp(from === 3 ? 1 : 0, i === 3 ? 1 : 0, t));
        sum.setAttribute("opacity", lerp(from === 6 ? 1 : 0, i === 6 ? 1 : 0, t));
      },
    });
  })();

  /* ---------- Figure 3: distribution shift ---------- */
  (function shiftFigure() {
    const fig = document.getElementById("fig-shift");
    if (!fig) return;
    const rows = D.shift.rows;
    if (!rows || !rows.length) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 400;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "Distribution shift: states and success" }, stage);
    root.style.minWidth = "640px";
    function node(p, x, y, text, col) {
      const g = svg("g", { opacity: 0 }, p);
      B(g, x, y, 128, 36, { fill: col === C.red ? "rgba(184,67,47,0.12)" : "rgba(11,94,168,0.12)", stroke: col });
      T(g, x + 64, y + 23, text, { "text-anchor": "middle", "font-size": 11.5 });
      return g;
    }
    T(root, 16, 26, "abstract states (action type : outcome type)", { "font-size": 12, fill: C.muted });
    const exp = [["start", 16, 56], ["search:found", 170, 56], ["read:ok", 324, 56], ["edit:ok", 16, 136], ["submit", 170, 136]];
    const expNodes = exp.map(([s, x, y]) => node(root, x, y, s, C.blue));
    const expLabel = svg("g", { opacity: 0 }, root);
    T(expLabel, 16, 200, "every demonstration walks this path", { "font-size": 12.5, fill: C.blue });
    const off = [["read:error", 16, 240], ["search:found", 170, 240], ["read:ok", 324, 240], ["edit:ok", 16, 320], ["submit", 170, 320]];
    const offNodes = off.map(([s, x, y], k) => node(root, x, y, s, k === 0 ? C.orange : C.red));
    const offLabel = svg("g", { opacity: 0 }, root);
    T(offLabel, 16, 230, "after one harmless mistake, nothing below is in the demos", { "font-size": 12.5, fill: C.red });
    // bars
    const BX = 520, BW = 400, BY = 340, BH = 280;
    const gb = svg("g", { opacity: 0 }, root);
    T(gb, BX, 26, "success on unseen tasks, greedy, Apple M5 CPU", { "font-size": 12, fill: C.muted });
    for (let i = 0; i <= 4; i++) {
      const y = BY - (BH * i) / 4;
      svg("line", { x1: BX + 30, x2: BX + BW, y1: y, y2: y, stroke: C.ruleSoft }, gb);
      T(gb, BX + 24, y + 4, 25 * i + "%", { "text-anchor": "end", "font-size": 10.5, fill: C.muted });
    }
    const bars = rows.map((r, k) => {
      const x = BX + 50 + k * 90;
      const b = svg("rect", { x, y: BY, width: 56, height: 0, rx: 3, fill: C.blue }, gb);
      const v = T(gb, x + 28, BY - 6, "", { "text-anchor": "middle", "font-size": 12.5, "font-weight": 600 });
      T(gb, x + 28, BY + 18, k + (k === 1 ? " mistake" : " mistakes"), { "text-anchor": "middle", "font-size": 10.5, fill: C.soft });
      const e = svg("line", { x1: x - 6, x2: x + 62, y1: BY, y2: BY, stroke: C.orange, "stroke-width": 2, opacity: 0 }, gb);
      return { b, v, e, r };
    });
    T(gb, BX + 30, BY + 40, "orange line: the scripted expert from the same perturbed starts", { "font-size": 11, fill: C.soft });
    const frames = [
      { note: "A demonstration is one path through state space. The expert never errs, so its path is narrow." },
      { note: "All of the policy's training data lives on this path. It has perfect loss on these states and no information about any other." },
      { note: "We force one small mistake: a read of a file that does not exist. The task is still fully solvable. The expert would carry on." },
      { note: "From here, the correct next action is the same as before. But the context is not. Every state that follows contains tokens the policy has never seen at those positions." },
      { note: "Measured: with no mistakes the cloned policy succeeds " + pct(rows[0].success) + " of the time." },
      { note: "One forced mistake: " + pct(rows[1].success) + ". The expert from the same starts: " + pct(rows[1].expert) + "." },
      { note: "Two mistakes: " + pct(rows[2].success) + ". Three: " + pct(rows[3].success) + ". The mistakes are harmless. The drop is entirely the policy's." },
    ];
    stepper(fig, {
      frames, tween: 900, hold: 2500,
      render(i, t, from) {
        const reveal = (cond) => lerp(cond(from) ? 1 : 0, cond(i) ? 1 : 0, t);
        expNodes.forEach((n) => n.setAttribute("opacity", reveal((f) => f >= 0)));
        expLabel.setAttribute("opacity", reveal((f) => f >= 1));
        offNodes[0].setAttribute("opacity", reveal((f) => f >= 2));
        offNodes.slice(1).forEach((n) => n.setAttribute("opacity", reveal((f) => f >= 3)));
        offLabel.setAttribute("opacity", reveal((f) => f >= 3));
        gb.setAttribute("opacity", reveal((f) => f >= 4));
        bars.forEach((b, k) => {
          const at = (f) => (f >= 4 + Math.min(k, 2) ? b.r.success : 0);
          const v = lerp(at(from), at(i), t), h = v * BH;
          b.b.setAttribute("y", BY - h);
          b.b.setAttribute("height", h);
          b.v.setAttribute("y", BY - h - 6);
          b.v.textContent = at(i) || t < 1 && at(from) ? pct(v) : "";
          const e = at(i) !== 0 || b.r.success === 0 && i >= 4 + Math.min(k, 2);
          b.e.setAttribute("opacity", e ? 1 : 0);
          b.e.setAttribute("y1", BY - b.r.expert * BH);
          b.e.setAttribute("y2", BY - b.r.expert * BH);
        });
      },
    });
  })();

  /* ---------- Figure 4: group-relative advantage explorer ---------- */
  (function groupFigure() {
    const fig = document.getElementById("fig-group");
    if (!fig) return;
    const gIn = fig.querySelector("#grp-g");
    const scale = fig.querySelector("#grp-scale");
    const cells = fig.querySelector("#grp-cells");
    const bars = fig.querySelector("#grp-bars");
    const note = fig.querySelector(".ct-figure__note");
    let rewards = [1, 0, 0, 0, 0, 0, 0, 0];
    function resize() {
      const g = +gIn.value;
      fig.querySelector("#grp-g-out").textContent = g;
      while (rewards.length < g) rewards.push(0);
      rewards.length = g;
    }
    function draw() {
      resize();
      const g = rewards.length;
      const mean = rewards.reduce((a, b) => a + b, 0) / g;
      const std = Math.sqrt(rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(g - 1, 1));
      const adv = rewards.map((r) => (scale.checked ? (r - mean) / (std + 1e-4) : r - mean));
      cells.innerHTML = rewards.map((r, i) => `<button type="button" class="c12-cell${r === 1 ? " is-win" : r > 0 ? " is-mid" : ""}" data-i="${i}" aria-label="Trajectory ${i + 1} reward ${r}. Click to change.">&tau;<sub>${i + 1}</sub><b>${r}</b></button>`).join("");
      const m = Math.max(1, ...adv.map(Math.abs));
      bars.innerHTML = adv.map((a, i) => {
        const h = (Math.abs(a) / m) * 62;
        const zero = Math.abs(a) < 1e-9;
        return `<div class="c12-advbar" title="A${i + 1} = ${a.toFixed(3)}"><span class="c12-advbar__up">${a > 0 ? `<i style="height:${h}px"></i>` : ""}</span><span class="c12-advbar__down">${a < 0 ? `<i style="height:${h}px"></i>` : ""}</span><em>${zero ? "0" : (a > 0 ? "+" : "") + a.toFixed(2)}</em></div>`;
      }).join("");
      const set = (id, v) => { fig.querySelector(id).textContent = v; };
      set("#grp-mean", mean.toFixed(3));
      set("#grp-std", std.toFixed(3));
      const pos = adv.filter((a) => a > 0), maxA = Math.max(...adv);
      set("#grp-max", maxA.toFixed(3));
      const flat = Math.max(...rewards) === Math.min(...rewards);
      set("#grp-signal", flat ? "none" : "active");
      if (flat) note.innerHTML = "All rewards are equal, so every advantage is exactly 0. This group contributes no policy gradient. The rollouts were paid for and teach nothing. Count these groups in your logs.";
      else if (pos.length === 1 && scale.checked) note.innerHTML = "One winner of " + g + " gets A = " + maxA.toFixed(2) + ". A rare success gets a large push. Every loser is pushed down a little. The advantages sum to zero.";
      else if (!scale.checked) note.innerHTML = "Mean baseline only (the Dr. GRPO variant). A rare win now gets at most 1 minus the mean. Nothing is amplified, so low-variance groups no longer shout.";
      else note.innerHTML = pos.length + " of " + g + " beat the group mean. The more common success is, the smaller the push it gets.";
    }
    cells.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      const i = +b.dataset.i;
      rewards[i] = rewards[i] === 0 ? 1 : rewards[i] === 1 ? 0.5 : 0;
      draw();
    });
    fig.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const g = +gIn.value, p = b.dataset.preset;
      rewards = Array.from({ length: g }, (_, i) => p === "one" ? (i === 0 ? 1 : 0) : p === "half" ? (i % 2 === 0 ? 1 : 0) : p === "win" ? 1 : p === "almost" ? (i === 0 ? 0 : 1) : 0);
      draw();
    }));
    gIn.addEventListener("input", draw);
    scale.addEventListener("input", draw);
    draw();
  })();

  /* ---------- Figure 5: trajectory replay with per-step advantages ---------- */
  (function replayFigure() {
    const fig = document.getElementById("fig-replay");
    if (!fig || !D.replay || !D.replay.length) return;
    const groupSel = fig.querySelector("#rp-group");
    const memberBox = fig.querySelector("#rp-members");
    const list = fig.querySelector("#rp-steps");
    const stepIn = fig.querySelector("#rp-step");
    const note = fig.querySelector(".ct-figure__note");
    let g = 0, m = 0, credit = "traj";
    D.replay.forEach((x, i) => {
      const o = document.createElement("option");
      o.value = i; o.textContent = `Level ${x.level}, ${x.wins}/${x.members.length} wins: ${x.task}`;
      groupSel.appendChild(o);
    });
    function pickDefault() {
      const mem = D.replay[g].members;
      m = Math.max(0, mem.findIndex((x) => x.success));
    }
    function draw() {
      const G = D.replay[g];
      const M = G.members[m];
      memberBox.innerHTML = G.members.map((x, i) => `<button type="button" aria-pressed="${i === m}" data-m="${i}" class="c12-chip ${x.success ? "is-win" : "is-lose"}">&tau;<sub>${i + 1}</sub> ${x.success ? "pass" : "fail"}</button>`).join("");
      stepIn.max = M.steps.length - 1;
      if (+stepIn.value > M.steps.length - 1) stepIn.value = M.steps.length - 1;
      const cur = +stepIn.value;
      fig.querySelector("#rp-step-out").textContent = `${cur + 1} / ${M.steps.length}`;
      const vals = M.steps.map((s) => (credit === "traj" ? M.traj_adv : credit === "gae" ? s.gae : s.td));
      const mx = Math.max(0.3, ...vals.map(Math.abs));
      list.innerHTML = M.steps.map((s, i) => {
        const v = vals[i], w = (Math.abs(v) / mx) * 50;
        return `<div class="c12-step${i === cur ? " is-current" : ""}${i > cur ? " is-future" : ""}">
          <div class="c12-step__t"><code>${esc(s.action || "(empty)")}</code><span class="${s.error ? "is-err" : ""}">${esc(s.obs)}</span></div>
          <div class="c12-step__v">V ${s.value.toFixed(2)}</div>
          <div class="c12-step__bar"><i style="${v >= 0 ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:${v >= 0 ? C.blue : C.orange}"></i><em>${(v >= 0 ? "+" : "") + v.toFixed(2)}</em></div>
        </div>`;
      }).join("");
      const s = M.steps[cur];
      note.innerHTML = credit === "traj"
        ? `Trajectory level: every action of this ${M.success ? "winning" : "losing"} episode gets A = ${M.traj_adv.toFixed(2)}, including <code>${esc(s.action || "(empty)")}</code>. The update cannot tell a useful step from a wasted one.`
        : credit === "gae"
          ? `Action level, GAE with &lambda; = 0.95: this step gets ${s.gae.toFixed(2)}. V(s<sub>t</sub>) is the critic's estimate of the final reward before the action.`
          : `One-step TD error (&lambda; = 0): V(s<sub>t+1</sub>) &minus; V(s<sub>t</sub>) = ${s.td.toFixed(2)}. It says whether this one action raised or lowered the critic's estimate.`;
    }
    groupSel.addEventListener("input", () => { g = +groupSel.value; pickDefault(); stepIn.value = 0; draw(); });
    memberBox.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { m = +b.dataset.m; stepIn.value = 0; draw(); } });
    fig.querySelectorAll("[data-credit]").forEach((b) => b.addEventListener("click", () => {
      credit = b.dataset.credit;
      fig.querySelectorAll("[data-credit]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw();
    }));
    stepIn.addEventListener("input", draw);
    pickDefault();
    draw();
  })();

  /* ---------- Figure 6: learning-curve explorer ---------- */
  (function curveFigure() {
    const fig = document.getElementById("fig-curves");
    if (!fig) return;
    const host = fig.querySelector("#cv-chart");
    const note = fig.querySelector(".ct-figure__note");
    const sel = fig.querySelector("#cv-exp");
    const suiteSel = fig.querySelector("#cv-suite");
    const evalPts = (run, suite) => {
      const pts = [[0, run.start[suite]]];
      run.history.forEach((h) => { if (h.eval && h.eval[suite] != null) pts.push([h.iter, h.eval[suite]]); });
      return pts;
    };
    const EXP = D.curves;
    if (!Object.keys(EXP).length) return;
    Object.keys(EXP).forEach((k) => { const o = document.createElement("option"); o.value = k; o.textContent = EXP[k].title; sel.appendChild(o); });
    function suites(exp) { return Object.keys(exp.runs[0].start); }
    function fill() {
      const exp = EXP[sel.value];
      const prev = suiteSel.value;
      suiteSel.innerHTML = suites(exp).map((s) => `<option value="${s}">${s}</option>`).join("");
      if (suites(exp).includes(prev)) suiteSel.value = prev;
      else if (exp.suite) suiteSel.value = exp.suite;
    }
    function draw() {
      const exp = EXP[sel.value];
      const suite = suiteSel.value;
      const series = exp.runs.map((r) => ({ name: r.name, points: evalPts(r, suite) }));
      lineChart(host, { series, percent: true, ymin: 0, ymax: 1, ylabel: `greedy success on the ${suite} suite`, xlabel: exp.xlabel || "iteration", label: exp.title });
      note.innerHTML = exp.note;
    }
    sel.addEventListener("input", () => { fill(); draw(); });
    suiteSel.addEventListener("input", draw);
    sel.value = Object.keys(EXP)[0];
    fill();
    draw();
  })();

  /* ---------- Figure 7: the reward hack, two small multiples ---------- */
  (function hackFigure() {
    const fig = document.getElementById("fig-hack");
    if (!fig) return;
    const hack = D.hack;
    if (!hack) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 430;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "Reward hack: shaped return against true success" }, stage);
    root.style.minWidth = "640px";
    function panel(x, title, ymax, percent) {
      const P = { x, y: 44, w: 420, h: 220, ymax, percent };
      T(root, x, 26, title, { "font-size": 13, "font-weight": 600 });
      for (let i = 0; i <= 4; i++) {
        const v = (ymax * i) / 4, yy = P.y + P.h - (P.h * i) / 4;
        svg("line", { x1: x + 34, x2: x + P.w, y1: yy, y2: yy, stroke: C.ruleSoft }, root);
        T(root, x + 28, yy + 4, percent ? pct(v) : v.toFixed(1), { "text-anchor": "end", "font-size": 10.5, fill: C.muted });
      }
      T(root, x + 34 + (P.w - 34) / 2, P.y + P.h + 20, "iteration (shaped reward on)", { "text-anchor": "middle", "font-size": 10.5, fill: C.muted });
      P.path = svg("path", { fill: "none", stroke: C.orange, "stroke-width": 2, "stroke-linejoin": "round" }, root);
      P.dot = svg("circle", { r: 4.5, fill: C.orange, stroke: C.paper, "stroke-width": 2 }, root);
      P.val = T(root, x, 0, "", { "font-size": 12.5, "font-weight": 600 });
      return P;
    }
    const a = panel(20, "What the hacked reward reports: shaped return", Math.ceil(Math.max(...hack.ret)), false);
    const b = panel(510, "What we wanted: true task success", 1, true);
    const xs = hack.iters, N = xs.length, xmax = xs[N - 1];
    function setPath(P, ys, upto) {
      const X = (v) => P.x + 34 + (v / xmax) * (P.w - 34);
      const Y = (v) => P.y + P.h - (v / P.ymax) * P.h;
      const n = Math.max(1, Math.floor(upto));
      const pts = [];
      for (let i = 0; i < n; i++) pts.push([X(xs[i]), Y(ys[i])]);
      if (upto < N) {
        const f = upto - n, x0 = xs[n - 1], x1 = xs[n], y0 = ys[n - 1], y1 = ys[n];
        pts.push([X(lerp(x0, x1, f)), Y(lerp(y0, y1, f))]);
      }
      P.path.setAttribute("d", pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" "));
      const l = pts[pts.length - 1];
      P.dot.setAttribute("cx", l[0]); P.dot.setAttribute("cy", l[1]);
      const idx = Math.min(N - 1, Math.round(upto) - 1);
      P.val.setAttribute("x", Math.min(l[0] + 8, P.x + P.w - 40));
      P.val.setAttribute("y", Math.max(l[1] - 8, P.y + 8));
      P.val.textContent = P.percent ? pct(ys[idx]) : ys[idx].toFixed(2);
    }
    B(root, 20, 300, 920, 116, { fill: "#fff" });
    const tx = hack.traces.map((_) => svg("g", { opacity: 0 }, root));
    hack.traces.forEach((lines, k) => lines.slice(0, 5).forEach((l, j) => T(tx[k], 32, 322 + j * 20, l, { "font-size": 12, fill: j === 0 ? C.muted : C.ink })));
    const stops = [1, 0.25, 0.5, 0.75, 1].map((f, i) => (i === 0 ? 1 : Math.max(2, Math.round(f * N))));
    const frames = [
      { note: "The start: a policy cloned from demonstrations. Success is high. It calls tools only as needed, then submits." },
      { note: "The reward now pays +0.2 for every tool call that does not error, plus the outcome. It sounds reasonable. Sampled trajectories that keep working score higher in their group." },
      { note: "The group-relative advantage does its job. It pushes toward whatever scores higher. Return climbs." },
      { note: "True success falls while return rises. A dashboard that shows only return would call this run a success." },
      { note: "Final state. The policy stops submitting and spends its budget on calls that pay. This is the highest-scoring behaviour under the reward we wrote." },
    ];
    stepper(fig, {
      frames, tween: 1400, hold: 2600,
      render(i, t, from) {
        const u = lerp(stops[from], stops[i], t);
        setPath(a, hack.ret, u);
        setPath(b, hack.success, u);
        const late = (f) => (f >= 3 ? 1 : 0);
        tx[0].setAttribute("opacity", lerp(1 - late(from), 1 - late(i), t));
        tx[1].setAttribute("opacity", lerp(late(from), late(i), t));
      },
    });
  })();

  /* ---------- Figure 8: curriculum ladder ---------- */
  (function ladderFigure() {
    const fig = document.getElementById("fig-ladder");
    if (!fig) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 400;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "Curriculum ladder" }, stage);
    root.style.minWidth = "640px";
    const L = D.ladder;
    const rung = L.map((r, i) => {
      const y = VH - 74 - i * 68, x = 20 + i * 44;
      const g = svg("g", { opacity: 0.25 }, root);
      const box = B(g, x, y, 690 - i * 22, 58, { fill: "#fff" });
      T(g, x + 12, y + 23, `${i + 1}  ${r.name}`, { "font-size": 14, "font-weight": 600 });
      T(g, x + 12, y + 44, r.task, { "font-size": 12, fill: C.soft });
      T(g, x + 676 - i * 22, y + 23, r.tools, { "text-anchor": "end", "font-size": 12, fill: C.blue });
      return { g, box };
    });
    const frames = L.map((r) => ({ note: r.note }));
    stepper(fig, {
      frames, tween: 800, hold: 2600,
      render(i, t, from) {
        rung.forEach((r, k) => {
          const on = (f) => (k <= f ? 1 : 0.25);
          r.g.setAttribute("opacity", lerp(on(from), on(i), t));
          r.box.setAttribute("stroke", k === i ? C.orange : C.rule);
          r.box.setAttribute("stroke-width", k === i ? 2 : 1);
        });
      },
    });
    const side = svg("g", null, root);
    T(side, 740, 40, "BC success", { "font-size": 12, fill: C.muted });
    L.forEach((r, i) => {
      if (r.score == null) return;
      T(side, 740, 66 + i * 26, `${r.label}  ${pct(r.score)}`, { "font-size": 12.5 });
    });
  })();

  /* ---------- Figure 9: the agent-RL system ---------- */
  (function systemFigure() {
    const fig = document.getElementById("fig-system");
    if (!fig) return;
    const stage = fig.querySelector(".ct-figure__stage");
    const VW = 960, VH = 400;
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "Agent RL system pipeline" }, stage);
    root.style.minWidth = "640px";
    const S = D.system;
    const boxes = [
      ["task sampler", "sample", 20, 60], ["env workers", "rollout", 250, 60], ["agent rollouts", "rollout", 480, 60], ["trajectory store", "store", 710, 60],
      ["verifier", "verify", 710, 220], ["training batches", "batch", 480, 220], ["policy update", "update", 250, 220], ["new policy", null, 20, 220],
    ];
    const sub = [
      "levels 1-5, perturbed starts", "tools run, failures injected", "G per task, sampled ids kept", "append-only JSON lines",
      "replay actions, check files", "drop stale, pad, mask", "clipped surrogate on actions", "version + 1, back to rollouts",
    ];
    const bw = 210, bh = 96;
    const nodes = boxes.map((b, i) => {
      const g = svg("g", { opacity: 0.3 }, root);
      const r = B(g, b[2], b[3], bw, bh, { fill: "#fff" });
      T(g, b[2] + 12, b[3] + 26, b[0], { "font-size": 14, "font-weight": 600 });
      T(g, b[2] + 12, b[3] + 48, sub[i], { "font-size": 11, fill: C.soft });
      const pctText = T(g, b[2] + 12, b[3] + 76, "", { "font-size": 12, fill: C.orange, "font-weight": 600 });
      return { g, r, pctText, key: b[1] };
    });
    const links = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]];
    function link(a, b) {
      const A = boxes[a], B2 = boxes[b];
      if (A[3] === B2[3]) {
        if (A[2] < B2[2]) return [A[2] + bw, A[3] + bh / 2, B2[2], B2[3] + bh / 2];
        return [A[2], A[3] + bh / 2, B2[2] + bw, B2[3] + bh / 2];
      }
      return [A[2] + bw / 2, A[3] + bh, B2[2] + bw / 2, B2[3]];
    }
    links.forEach(([a, b]) => { const l = link(a, b); svg("line", { x1: l[0], y1: l[1], x2: l[2], y2: l[3], stroke: C.muted, "stroke-width": 1.5 }, root); });
    svg("path", { d: `M125 220 L125 156`, stroke: C.muted, "stroke-width": 1.5, "stroke-dasharray": "4 4", fill: "none" }, root);
    const packet = svg("circle", { r: 8, fill: C.orange, opacity: 0 }, root);
    const capt = T(root, 20, 360, "", { "font-size": 13, fill: C.soft });
    const run = S[0];
    const total = run.total;
    const share = (k) => (k && run.clock[k] != null ? (100 * run.clock[k]) / total : null);
    const frames = [
      { note: "The sampler picks tasks and start states. Exploration lives here as much as in the sampling temperature: diverse tasks, diverse starts." },
      { note: "Workers run the tools. Our toy tools take microseconds. Real tools take seconds, and that changes where the bottleneck is. Failures are injected here too." },
      { note: "The policy generates actions for every waiting episode in one batch. Each record stores the sampled token ids, the log-probabilities and the policy version." },
      { note: "Everything goes to disk first. An append-only store is what makes offline runs, audits and replay possible." },
      { note: "The verifier replays the recorded actions on a fresh copy of the task and decides the reward from the files. It never trusts the rollout's claim." },
      { note: "Batches drop stale versions and groups left too small. Padding and masks are built here." },
      { note: "The clipped surrogate runs over action tokens. In this run the update took " + (share("update") || 0).toFixed(0) + "% of the wall time and rollouts took " + (share("rollout") || 0).toFixed(0) + "%." },
      { note: "A new version goes back to the rollouts. Every trajectory records which version produced it, so staleness can be measured instead of guessed." },
    ];
    stepper(fig, {
      frames, tween: 900, hold: 2100,
      render(i, t, from) {
        nodes.forEach((n, k) => {
          const on = (f) => (k <= f ? 1 : 0.3);
          n.g.setAttribute("opacity", lerp(on(from), on(i), t));
          n.r.setAttribute("stroke", k === i ? C.orange : C.rule);
          n.r.setAttribute("stroke-width", k === i ? 2 : 1);
          const v = share(n.key);
          n.pctText.textContent = v != null && k <= i ? v.toFixed(1) + "% of wall time" : "";
        });
        let px, py;
        if (i === 0 || from > i) { px = boxes[i][2] + bw / 2; py = boxes[i][3] + bh / 2; }
        else {
          const a = boxes[from], b = boxes[i];
          px = lerp(a[2] + bw / 2, b[2] + bw / 2, t);
          py = lerp(a[3] + bh / 2, b[3] + bh / 2, t);
        }
        packet.setAttribute("cx", px); packet.setAttribute("cy", py);
        packet.setAttribute("opacity", 0.75);
        capt.textContent = `measured: ${S[0].stored} trajectories, ${S[0].checked} verified, ${S[0].mismatches} mismatches, ${total.toFixed(0)} s, Apple M5 CPU`;
      },
    });
  })();

  // Hand the figures to the preview exporter if it wants them.
  window.C12_READY = true;
})();
