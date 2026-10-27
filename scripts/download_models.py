"""Download one or both models without installing a runtime or starting a server."""
import argparse
import sys


def setup_models(model):
    if model in ('apdrawinggan', 'all'):
        from setup_apdrawinggan import setup
        setup()
    if model in ('cartoon', 'all'):
        from setup_cartoon import setup
        setup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('all', 'apdrawinggan', 'cartoon'), default='all')
    args = parser.parse_args()
    try:
        setup_models(args.model)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Download failed: {exc}', file=sys.stderr)
        sys.exit(1)
