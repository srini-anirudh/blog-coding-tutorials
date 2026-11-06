import sys, time, json
sys.path.insert(0, "<SCRATCH>")
from cdp import CDP
tag = sys.argv[1]; width = int(sys.argv[2]) if len(sys.argv) > 2 else 1440
figs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["fig-neg","fig-batch","fig-dpo","fig-blind","fig-iou","fig-grpo","fig-freeze","fig-hack"]
c = CDP(9409)
c.call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1, mobile=False)
c.call("Network.setCacheDisabled", cacheDisabled=True)
c.go("about:blank", 0.3)
c.go("http://localhost:8809/blogs/post-train-a-vlm/index.html?v=" + tag, 5)
print("logs", c.logs)
print("overflow", c.ev("[document.documentElement.scrollWidth, innerWidth]"))
print("h1", c.ev("document.querySelectorAll('h1').length"), "ld", c.ev("JSON.parse(document.querySelector('script[type=\"application/ld+json\"]').textContent).headline"))
for f in figs:
    ok = c.ev(f"!!document.getElementById('{f}')")
    if not ok: print("missing", f); continue
    r = c.ev(f"(()=>{{const e=document.getElementById('{f}'); e.scrollIntoView({{block:'start'}}); return 1}})()")
    c.sleep(0.6)
    # step through all frames via Next
    n = c.ev(f"(()=>{{const e=document.getElementById('{f}'); const st=e.querySelector('.ct-status'); return st?st.textContent:''}})()")
    total = int(n.split('/')[-1]) if n else 0
    c.ev(f"(()=>{{const b=document.getElementById('{f}').querySelector('[data-action=\"play-pause\"]'); if(b&&b.getAttribute('aria-pressed')==='true') b.click();}})()")
    for k in range(max(1, total)):
        rect = c.ev(f"(()=>{{const r=document.getElementById('{f}').getBoundingClientRect(); return [r.x+scrollX,r.y+scrollY,r.width,r.height]}})()")
        c.sleep(1.5)
        c.shot(f"shots/{tag}-{f}-{k}.png", {"x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3]})
        if total: c.ev(f"document.getElementById('{f}').querySelector('[data-action=\"next\"]').click()")
    print(f, "frames", total, "over", c.ev(f"(()=>{{const s=document.getElementById('{f}').querySelector('.ct-figure__stage'); return [s.scrollWidth,s.clientWidth]}})()"))
print("logs", c.logs)
