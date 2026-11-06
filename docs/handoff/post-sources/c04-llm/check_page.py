import sys, json, time
sys.path.insert(0, "<SCRATCH>")
from cdp import CDP
URL = "http://localhost:8804/blogs/build-a-modern-llm-from-scratch/index.html"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/x"
c = CDP(9404)
def set_w(w, h=1000):
    c.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
for w in (1440, 390):
    set_w(w)
    c.go("about:blank", 0.3)
    c.go(URL, 3)
    r = c.ev("""(() => { const de = document.documentElement;
       const figs = [...document.querySelectorAll('.ct-figure')].map(f => ({id:f.id, ctl:!!f.querySelector('.ct-controls'), status:(f.querySelector('.ct-status')||{}).textContent}));
       const bad=[...document.querySelectorAll('main *')].filter(e=>{const r=e.getBoundingClientRect();return r.right>innerWidth+1 && !e.closest('.ct-figure__stage,.code-listing,.terminal,.code-output,table,.comparison-table-wrap,pre,.shape-table-wrap,svg')}).slice(0,6).map(e=>e.tagName+'.'+e.className);
       return {w:innerWidth, sw:de.scrollWidth, figs, bad, title:document.title, h1:document.querySelectorAll('h1').length, listings:document.querySelectorAll('.code-listing').length,
         ld:(()=>{try{JSON.parse(document.querySelector('script[type="application/ld+json"]').textContent);return 'ok'}catch(e){return String(e)}})(), pending:(document.body.innerText.match(/\\[pending/g)||[]).length}; })()""")
    print(json.dumps(r)[:1400])
    print("logs", c.logs[:8]); c.logs.clear()
    for f in r["figs"]:
        if f["ctl"]:
            n = int(f["status"].split("/")[1])
            for i in range(n):
                c.ev(f"document.querySelector('#{f['id']} [data-action=next]').click()")
                c.sleep(0.05)
            c.ev(f"document.querySelector('#{f['id']}').scrollIntoView()")
    c.sleep(1.2)
    print("after stepping logs", c.logs[:8]); c.logs.clear()
    if w == 1440:
        for f in r["figs"]:
            box = c.ev(f"(()=>{{const b=document.getElementById('{f['id']}').getBoundingClientRect();return {{x:b.x+scrollX,y:b.y+scrollY,width:b.width,height:b.height}}}})()")
            c.ev(f"document.getElementById('{f['id']}').scrollIntoView()"); c.sleep(0.25)
            c.shot(f"{OUT}_{f['id']}.png", box)
