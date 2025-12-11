"""This script is used for extracting specific combinations from a cartesian product."""

import argparse
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", nargs="+", type=str, action="append")
    parser.add_argument("--id", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    lists = args.l
    idx = args.id
    cart_prod = list(itertools.product(*lists))
    print(" ".join(cart_prod[idx]))


if __name__ == "__main__":
    main()
