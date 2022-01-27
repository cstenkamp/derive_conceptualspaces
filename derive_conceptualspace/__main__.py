import sys
from os.path import dirname
sys.path.append(dirname(__file__))

from .cli.run_pipeline import cli

if __name__ == "__main__":
    cli(obj={})