import json,time,base64,urllib.request,websocket
class CDP:
    def __init__(s,port=9333):
        tabs=json.load(urllib.request.urlopen(f'http://localhost:{port}/json'))
        t=[x for x in tabs if x['type']=='page'][0]
        s.ws=websocket.create_connection(t['webSocketDebuggerUrl'],timeout=60,suppress_origin=True); s.i=0; s.logs=[]
        s.call('Runtime.enable'); s.call('Page.enable'); s.call('Log.enable'); s.call('Network.enable'); s.call('Network.setCacheDisabled',cacheDisabled=True)
    def _note(s,r):
        m=r.get('method')
        if m=='Runtime.exceptionThrown':
            d=r['params']['exceptionDetails']; s.logs.append((d.get('exception',{}) or {}).get('description',d.get('text')))
        elif m=='Runtime.consoleAPICalled' and r['params']['type'] in('error','warning'):
            s.logs.append('console:'+' '.join(str(a.get('value',a.get('description',''))) for a in r['params']['args']))
        elif m=='Log.entryAdded' and r['params']['entry']['level']=='error':
            s.logs.append('log:'+r['params']['entry']['text'][:200])
    def call(s,m,**p):
        s.i+=1; s.ws.send(json.dumps({'id':s.i,'method':m,'params':p}))
        while True:
            r=json.loads(s.ws.recv())
            if r.get('id')==s.i: return r.get('result',r)
            s._note(r)
    def ev(s,expr):
        r=s.call('Runtime.evaluate',expression=expr,returnByValue=True,awaitPromise=True)
        return r.get('result',{}).get('value',r)
    def go(s,url,wait=4):
        s.call('Page.navigate',url=url); s.sleep(wait)
    def sleep(s,t):
        end=time.time()+t
        while time.time()<end:
            s.ws.settimeout(max(0.05,end-time.time()))
            try: s._note(json.loads(s.ws.recv()))
            except Exception: pass
        s.ws.settimeout(60)
    def shot(s,path,clip=None):
        p={'format':'png'}
        if clip: p['clip']=dict(clip,scale=1)
        r=s.call('Page.captureScreenshot',**p); open(path,'wb').write(base64.b64decode(r['data']))
