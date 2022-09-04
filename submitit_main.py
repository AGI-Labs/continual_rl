import submitit
import sys
from torch import multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.utils.argparse_manager import ArgparseManager


def start(args):
    # Pytorch multiprocessing requires either forkserver or spawn.
    try:
        multiprocessing.set_start_method("spawn")
    except ValueError as e:
        # Windows doesn't support forking, so fall back to spawn instead
        assert "cannot find context" in str(e)
        multiprocessing.set_start_method("spawn")

    experiment, policy = ArgparseManager.parse(args)

    if experiment is None:
        raise RuntimeError("No experiment started. Most likely there is no new run to start.")

    summary_writer = SummaryWriter(log_dir=experiment.output_dir)
    experiment.try_run(policy, summary_writer=summary_writer)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="tmp/results")

    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=24*60*3, slurm_partition="dev")

    job = executor.submit(start, sys.argv[1:])
    print(job.job_id)  # ID of your job

    output = job.result()  # waits for completion and returns output
