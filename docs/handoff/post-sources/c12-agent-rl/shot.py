import sys, time, json
sys.path.insert(0, "<SCRATCH>")
from cdp import CDP
tag = sys.argv[1]; width = int(sys.argv[2]) if len(sys.argv) > 2 else 1440
c = CDP(9412)
c.call("Network.enable"); c.call("Network.setCacheDisabled", cacheDisabled=True)
c.call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1, mobile=False)
c.go("about:blank", 0.3); c.go("http://localhost:8812/blogs/train-an-agent-with-rl/", 3)
print("logs", c.logs)
print("overflow", c.ev("document.documentElement.scrollWidth + ' / ' + innerWidth"))
print(c.ev("[...document.querySelectorAll('figure.ct-figure')].map(f=>f.id+':'+Math.round(f.getBoundingClientRect().width)+'x'+Math.round(f.getBoundingClientRect().height)+' sw'+f.querySelector('.ct-figure__stage').scrollWidth).join('\\n')"))
out = "<SCRATCH>/code-series/c12-agent-rl/"
ids = sys.argv[3].split(",") if len(sys.argv) > 3 else ["fig-env","fig-tape","fig-shift","fig-group"]
for fid in ids:
    frames = int(c.ev(f"(document.querySelector('#{fid} .ct-status')||{{textContent:'/ 1'}}).textContent.split('/')[1]"))
    n = c.ev(f"(()=>{{const f=document.getElementById('{fid}');f.scrollIntoView({{block:'center'}});return 1}})()")
    c.sleep(0.4)
    c.ev(f"(()=>{{const b=document.querySelector('#{fid} [data-action=play-pause]');if(b&&b.getAttribute('aria-pressed')==='true')b.click();}})()")
    last = frames if len(sys.argv) > 4 else 1
    for fr in range(max(last,1)):
        if fr: c.ev(f"document.querySelector('#{fid} [data-action=next]').click()"); c.sleep(1.3)
        else: c.sleep(0.5)
        r = c.ev(f"(()=>{{const r=document.getElementById('{fid}').getBoundingClientRect();return [r.x,r.y,r.width,r.height]}})()")
        c.shot(f"{out}{tag}_{fid}_{fr}.png", clip={"x":max(r[0],0),"y":max(r[1],0),"width":min(r[2],width),"height":min(r[3],1000)})
print("logs", c.logs)
