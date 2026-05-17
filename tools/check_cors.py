import urllib.request
import urllib.error


def print_headers(resp):
    print(resp.status)
    for k, v in resp.getheaders():
        print(f"{k}: {v}")


if __name__ == '__main__':
    urls = [
        ('ROOT', 'https://livestock-ai-engine.onrender.com/'),
        ('OPTIONS', 'https://livestock-ai-engine.onrender.com/api/estimate-weight?animal_type=dairy_cow')
    ]

    # GET root
    try:
        req = urllib.request.Request(urls[0][1], method='GET')
        with urllib.request.urlopen(req, timeout=15) as r:
            print('==', urls[0][0], '==')
            print_headers(r)
    except Exception as e:
        print('ROOT ERROR', repr(e))

    # OPTIONS preflight
    try:
        req2 = urllib.request.Request(urls[1][1], method='OPTIONS', headers={
            'Origin': 'https://livestock-ai-frontend.vercel.app',
            'Access-Control-Request-Method': 'POST'
        })
        with urllib.request.urlopen(req2, timeout=15) as r2:
            print('==', urls[1][0], '==')
            print_headers(r2)
    except urllib.error.HTTPError as he:
        print('OPTIONS HTTPError', he.code)
        try:
            for k, v in he.headers.items():
                print(f"{k}: {v}")
        except Exception:
            pass
    except Exception as e:
        print('OPTIONS ERROR', repr(e))

    # POST test (empty body) to observe error responses and headers
    try:
        print('== POST TEST ==')
        req3 = urllib.request.Request(urls[1][1], method='POST', headers={
            'Origin': 'https://livestock-ai-frontend.vercel.app',
            'Content-Type': 'application/octet-stream'
        }, data=b'')
        with urllib.request.urlopen(req3, timeout=20) as r3:
            print('POST OK')
            print_headers(r3)
    except urllib.error.HTTPError as he:
        print('POST HTTPError', he.code)
        try:
            for k, v in he.headers.items():
                print(f"{k}: {v}")
        except Exception:
            pass
    except Exception as e:
        print('POST ERROR', repr(e))
