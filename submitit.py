import submitit
import sys


if __name__ == "__main__":
    function = submitit.helpers.CommandFunction(["python", *sys.argv[1:]])
    executor = submitit.AutoExecutor(folder="tmp/log_test")
    executor.update_parameters(timeout_min=1440, slurm_partition="learnlab,learnfair", slurm_gpus_per_task=1, slurm_cpus_per_task=80, slurm_mem_gb=80)
    job = executor.submit(function)

    # The returned python path is the one used in slurm.
    # It should be the same as when running out of slurm!
    # This means that everything that is installed in your
    # conda environment should work just as well in the cluster
    print(job.result())
