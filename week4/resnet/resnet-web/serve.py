"""웹 실습 파일과 공유 데이터를 제공하는 로컬 HTTP 서버."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
STUDENT = ROOT / 'resnet-classroom'


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        requested = unquote(urlsplit(self.path).path)
        resolved = Path(self.translate_path(requested)).resolve()
        allowed = any(resolved == folder or folder in resolved.parents for folder in [WEB, STUDENT / 'data'])
        if not allowed or any(part.startswith('.') or part == 'node_modules' for part in resolved.relative_to(ROOT).parts):
            self.send_error(404)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not (STUDENT / 'data/base/manifest.csv').is_file():
        parser.error('Place resnet-classroom next to resnet-web before starting.')
    if not (WEB / 'catalog.json').is_file() or not (WEB / 'assets/tf.min.js').is_file():
        parser.error('Missing web files: catalog.json or assets/tf.min.js.')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT)))
    print(f'Local: http://localhost:{args.port}/resnet-web/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
