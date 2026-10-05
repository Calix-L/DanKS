"""Run the public game service locally: python run.py."""
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')
    return args


def main():
    args = parse_args()
    import uvicorn
    print(f'Open http://{args.host}:{args.port}/solo — Ctrl+C to stop', flush=True)
    uvicorn.run('backend.solo_app:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
