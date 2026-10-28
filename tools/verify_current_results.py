"""Compatibility entry point for current QACLF evidence verification."""
import argparse
import json

from verify_qaclf_results import verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-c-index', action='store_true')
    parser.add_argument('--recompute-probability', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.recompute_c_index, args.recompute_probability), indent=2))
