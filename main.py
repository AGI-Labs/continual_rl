import sys
from utils.argparse_manager import ArgparseManager


if __name__ == "__main__":
    experiment, policy = ArgparseManager.parse(sys.argv[1:])
