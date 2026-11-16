#!/bin/bash
#SBATCH --job-name=Panel_Alp_1_Cellulate_0_0_Simulation_Stage_L_1_Act_0
#SBATCH --partition=general1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4400
#SBATCH --extra-node-info=2:20:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL

#SBATCH --array=0-4

#SBATCH --output=%x_%A_%a.out

#SBATCH --begin=now+0hour

# Reference listing; use the portable CPU workflow in docs/workflow.md.
printf "%s\n" "Reference template: use python -m cavity_sbi --help." >&2
exit 2

task_index_mini=0
task_index_maxi=$((SLURM_NTASKS_PER_NODE*SLURM_NNODES))

task_index_start=$((SLURM_ARRAY_TASK_ID*task_index_maxi+task_index_mini))
task_index_final=$(((SLURM_ARRAY_TASK_ID+1)*task_index_maxi+task_index_mini-1))

for task_index in $(seq $task_index_start $task_index_final); do
    task_name="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${task_index}.out"
    task="./mars_projects/panel/${SLURM_JOB_NAME}.py"
    python $task $task_index >& $task_name &
done

wait
