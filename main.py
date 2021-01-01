import sys
from torch import multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.utils.argparse_manager import ArgparseManager


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    experiment, policy = ArgparseManager.parse(sys.argv[1:])
    summary_writer = SummaryWriter(log_dir=experiment.output_dir)
    experiment.try_run(policy, summary_writer=summary_writer)
