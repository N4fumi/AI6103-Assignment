import _bootstrap
import argparse
from src.visualize import generate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot saved trajectories; mathematical Beta/cosine plots do not require training.')
    parser.add_argument('--theory-only', action='store_true')
    generate(theory_only=parser.parse_args().theory_only)
