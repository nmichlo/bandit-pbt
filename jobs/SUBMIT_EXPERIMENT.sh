#!/bin/bash

# UCB: STEPPED INSTEAD OF EXPLOITED
sbatch --array=1-10 -n1 -p ha -J '15_ucbS' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest_ucb_incr_mode="stepped"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbSO' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbSO"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest_ucb_incr_mode="stepped" --pbt-disable-eager-step'

# STEPPED INSTEAD OF EXPLOITED and/or ONLY STEP EVERY EPOCH and/or READY AFTER 1 NOT 2
sbatch --array=1-10 -n1 -p ha -J '15_ucbSR' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest_ucb_incr_mode="stepped" --pbt-members-ready-after="1"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbSE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest_ucb_incr_mode="stepped" --cnn-steps-per-epoch="1" --pbt-target-steps="7"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbSRE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest_ucb_incr_mode="stepped" --cnn-steps-per-epoch="1" --pbt-target-steps="7" --pbt-members-ready-after="1"'

# DISABLED EAGER STEP
sbatch --array=1-10 -n1 -p ha -J '15_ucbO' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbO"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --pbt-disable-eager-step'
sbatch --array=1-10 -n1 -p ha -J '15_ranO' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranO"  --pbt-members="15"  --pbt-exploit-suggest="ran" --pbt-disable-eager-step'

# READY AFTER 1 NOT 2
sbatch --array=1-10 -n1 -p ha -J '15_ranR' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranR"  --pbt-members="15"  --pbt-exploit-suggest="ran" --pbt-members-ready-after="1"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbR' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbR"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --pbt-members-ready-after="1"'

# ONLY STEP EVERY EPOCH + READY AFTER 1 NOT 2
sbatch --array=1-10 -n1 -p ha -J '15_ranRE' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranER"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbRE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbER"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1"'

 ONLY STEP EVERY EPOCH + READY AFTER 1 NOT 2 + REDUCED MAX LEARNING RATE 0.1->0.01
sbatch --array=1-10 -n1 -p ha -J '15ucbREA' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbER"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1" --cnn_lr_max="0.01"'
sbatch --array=1-10 -n1 -p ha -J '15ucbREA' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbER"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1" --cnn_lr_max="0.01"'

# ONLY STEP EVERY EPOCH
sbatch --array=1-10 -n1 -p ha -J '15_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranEL"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbEL"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'

# ONLY STEP EVERY EPOCH
sbatch --array=1-10 -n1 -p ha -J '15_ranE' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranE"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="7"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbE"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"'

# ONLY STEP EVERY EPOCH - BUT MORE STEPS
sbatch --array=1-10 -n1 -p ha -J '15_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranEL"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
sbatch --array=1-10 -n1 -p ha -J '15_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbEL"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
sbatch --array=1-10 -n1 -p ha -J '25_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ranEL"  --pbt-members="25"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
sbatch --array=1-10 -n1 -p ha -J '25_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ucbEL"  --pbt-members="25"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
sbatch --array=1-10 -n1 -p ha -J '40_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ranEL"  --pbt-members="40"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
sbatch --array=1-10 -n1 -p ha -J '40_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ucbEL"  --pbt-members="40"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'

# ORIG
sbatch --array=1-10 -n1 -p ha -J '15_ran' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ran"  --pbt-members="15"  --pbt-exploit-suggest="ran"'
sbatch --array=1-10 -n1 -p ha -J '15_ucb' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucb"  --pbt-members="15"  --pbt-exploit-suggest="ucb"'
sbatch --array=1-10 -n1 -p ha -J '25_ran' -o 'logs/25_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ran"  --pbt-members="25"  --pbt-exploit-suggest="ran"'
sbatch --array=1-10 -n1 -p ha -J '25_ucb' -o 'logs/25_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ucb"  --pbt-members="25"  --pbt-exploit-suggest="ucb"'
sbatch --array=1-10 -n1 -p ha -J '40_ran' -o 'logs/40_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ran"  --pbt-members="40"  --pbt-exploit-suggest="ran"'
sbatch --array=1-10 -n1 -p ha -J '40_ucb' -o 'logs/40_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ucb"  --pbt-members="40"  --pbt-exploit-suggest="ucb"'

# ORIG EXTRA
sbatch --array=1-10 -n1 -p batch -J '15_gr' -o 'logs/15_gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_gr"  --pbt-members="15"  --pbt-exploit-suggest="gr"'
sbatch --array=1-10 -n1 -p batch -J '15_sm' -o 'logs/15_sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_sm"  --pbt-members="15"  --pbt-exploit-suggest="sm"'
sbatch --array=1-10 -n1 -p batch -J '15_e-gr' -o 'logs/15_e-gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_e-gr"  --pbt-members="15"  --pbt-exploit-suggest="e-gr"'
sbatch --array=1-10 -n1 -p batch -J '15_e-sm' -o 'logs/15_e-sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_e-sm"  --pbt-members="15"  --pbt-exploit-suggest="e-sm"'
sbatch --array=1-10 -n1 -p batch -J '15_e-ucb' -o 'logs/15_e-ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_e-ucb"  --pbt-members="15"  --pbt-exploit-suggest="e-ucb"'
sbatch --array=1-10 -n1 -p batch -J '25_gr' -o 'logs/25_gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_gr"  --pbt-members="25"  --pbt-exploit-suggest="gr"'
sbatch --array=1-10 -n1 -p batch -J '25_sm' -o 'logs/25_sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_sm"  --pbt-members="25"  --pbt-exploit-suggest="sm"'
sbatch --array=1-10 -n1 -p batch -J '25_e-gr' -o 'logs/25_e-gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-gr"  --pbt-members="25"  --pbt-exploit-suggest="e-gr"'
sbatch --array=1-10 -n1 -p batch -J '25_e-sm' -o 'logs/25_e-sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-sm"  --pbt-members="25"  --pbt-exploit-suggest="e-sm"'
sbatch --array=1-10 -n1 -p batch -J '25_e-ucb' -o 'logs/25_e-ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-ucb"  --pbt-members="25"  --pbt-exploit-suggest="e-ucb"'
sbatch --array=1-10 -n1 -p batch -J '40_gr' -o 'logs/40_gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_gr"  --pbt-members="40"  --pbt-exploit-suggest="gr"'
sbatch --array=1-10 -n1 -p batch -J '40_sm' -o 'logs/40_sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_sm"  --pbt-members="40"  --pbt-exploit-suggest="sm"'
sbatch --array=1-10 -n1 -p batch -J '40_e-gr' -o 'logs/40_e-gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_e-gr"  --pbt-members="40"  --pbt-exploit-suggest="e-gr"'
sbatch --array=1-10 -n1 -p batch -J '40_e-sm' -o 'logs/40_e-sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_e-sm"  --pbt-members="40"  --pbt-exploit-suggest="e-sm"'
sbatch --array=1-10 -n1 -p batch -J '40_e-ucb' -o 'logs/40_e-ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_e-ucb"  --pbt-members="40"  --pbt-exploit-suggest="e-ucb"'
