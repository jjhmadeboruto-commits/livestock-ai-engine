import urllib.request
import urllib.error

url = 'https://livestock-ai-engine.onrender.com/api/estimate-weight?animal_type=dairy_cow'
headers = {'Origin': 'https://livestock-ai-frontend.vercel.app', 'Content-Type': 'application/octet-stream'}

def print_resp(r):
    try:
        print(r.status)
        for k, v in r.getheaders():
            print(f"{k}: {v}")
        body = r.read(5000)
        print('BODY:', body[:500])
    except Exception as e:
        print('PRINT ERROR', e)

print('START POST CHECK')
try:
    req = urllib.request.Request(url, data=b'', headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        print('GOT RESPONSE')
        print_resp(r)
except urllib.error.HTTPError as he:
    print('HTTPError', he.code)
    try:
        for k, v in he.headers.items():
            print(f"{k}: {v}")
        try:
            body = he.read(5000)
            print('BODY:', body[:500])
        except Exception:
            pass
    except Exception as ex:
        print('HEADER PRINT ERROR', ex)
except Exception as e:
    print('ERROR', repr(e))
