#!/bin/bash

# 8 CHARS MAX - <size><suggest-key><steps><code>

# SIZES:
  # 15  (15 * 0.2 = 3)
  # 25  (25 * 0.2 = 5)
  # 40  (40 * 0.2 = 8)

# SUGGEST KEYS:
  # ra = random
  # so = softmax
  # es = e-softmax
  # gr = greedy
  # eg = e-greedy
  # eu = e-ucb
  # uc = ucb

# CODES:
  # S - Step based suggestion, not exploit based
  # O - Ordered, not randomized
  # E - 1 pbt step per 1 epoch, not 5 pbt steps per 1 epoch
  # R - Members ready after 1 step, not 2
  # L - Lower max learning rate to 0.01, not 0.1


# READY, LEARN
sbatch --array=1-25 -n1 -p ha -J '15ra20rl_' -o 'logs/15ra20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra20rl_" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15uc20rl_' -o 'logs/15uc20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20rl_" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15uc20rls' -o 'logs/15uc20rls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20rls" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p ha -J '15gr20rl_' -o 'logs/15gr20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr20rl_" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15so20rl_' -o 'logs/15so20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so20rl_" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eg20rl_' -o 'logs/15eg20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg20rl_" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15es20rl_' -o 'logs/15es20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es20rl_" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eu20rl_' -o 'logs/15eu20rl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20rl_" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eu20rls' -o 'logs/15eu20rls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20rls" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'

# EPOCH, READY, LEARN
sbatch --array=1-25 -n1 -p ha -J '15ra7erl_' -o 'logs/15ra7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra7erl_" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15uc7erl_' -o 'logs/15uc7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7erl_" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15uc7erls' -o 'logs/15uc7erls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7erls" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p ha -J '15gr7erl_' -o 'logs/15gr7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr7erl_" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15so7erl_' -o 'logs/15so7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so7erl_" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eg7erl_' -o 'logs/15eg7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg7erl_" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15es7erl_' -o 'logs/15es7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es7erl_" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eu7erl_' -o 'logs/15eu7erl__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7erl_" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eu7erls' -o 'logs/15eu7erls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7erls" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'

# LEARN
sbatch --array=1-25 -n1 -p ha -J '15ra20_l_' -o 'logs/15ra20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra20_l_" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15uc20_l_' -o 'logs/15uc20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20_l_" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15uc20_ls' -o 'logs/15uc20_ls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20_ls" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p ha -J '15gr20_l_' -o 'logs/15gr20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr20_l_" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p ha -J '15so20_l_' -o 'logs/15so20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so20_l_" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eg20_l_' -o 'logs/15eg20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg20_l_" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15es20_l_' -o 'logs/15es20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es20_l_" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eu20_l_' -o 'logs/15eu20_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20_l_" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '15eu20_ls' -o 'logs/15eu20_ls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20_ls" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'

# EPOCH, LEARN
sbatch --array=1-25 -n1 -p batch -J '15ra7e_l_' -o 'logs/15ra7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra7e_l_" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p batch -J '15uc7e_l_' -o 'logs/15uc7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7e_l_" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p batch -J '15uc7e_ls' -o 'logs/15uc7e_ls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7e_ls" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p batch -J '15gr7e_l_' -o 'logs/15gr7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr7e_l_" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
sbatch --array=1-25 -n1 -p batch -J '15so7e_l_' -o 'logs/15so7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so7e_l_" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p batch -J '15eg7e_l_' -o 'logs/15eg7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg7e_l_" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p batch -J '15es7e_l_' -o 'logs/15es7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es7e_l_" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p batch -J '15eu7e_l_' -o 'logs/15eu7e_l__%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7e_l_" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p batch -J '15eu7e_ls' -o 'logs/15eu7e_ls_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7e_ls" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'

# READY
sbatch --array=1-25 -n1 -p batch -J '15ra20r__' -o 'logs/15ra20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra20r__" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
sbatch --array=1-25 -n1 -p batch -J '15uc20r__' -o 'logs/15uc20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20r__" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
sbatch --array=1-25 -n1 -p batch -J '15uc20r_s' -o 'logs/15uc20r_s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20r_s" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p batch -J '15gr20r__' -o 'logs/15gr20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr20r__" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
sbatch --array=1-25 -n1 -p batch -J '15so20r__' -o 'logs/15so20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so20r__" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15eg20r__' -o 'logs/15eg20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg20r__" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15es20r__' -o 'logs/15es20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es20r__" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15eu20r__' -o 'logs/15eu20r___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20r__" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15eu20r_s' -o 'logs/15eu20r_s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20r_s" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --pbt-members-ready-after="1" --suggest-ucb-incr-mode="stepped"'

# EPOCH, READY
sbatch --array=1-25 -n1 -p batch -J '15ra7er__' -o 'logs/15ra7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra7er__" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
sbatch --array=1-25 -n1 -p batch -J '15uc7er__' -o 'logs/15uc7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7er__" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
sbatch --array=1-25 -n1 -p batch -J '15uc7er_s' -o 'logs/15uc7er_s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7er_s" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p batch -J '15gr7er__' -o 'logs/15gr7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr7er__" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
sbatch --array=1-25 -n1 -p batch -J '15so7er__' -o 'logs/15so7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so7er__" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15eg7er__' -o 'logs/15eg7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg7er__" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15es7er__' -o 'logs/15es7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es7er__" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15eu7er__' -o 'logs/15eu7er___%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7er__" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1"'
#sbatch --array=1-25 -n1 -p batch -J '15eu7er_s' -o 'logs/15eu7er_s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7er_s" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --pbt-members-ready-after="1" --suggest-ucb-incr-mode="stepped"'

#
sbatch --array=1-25 -n1 -p batch -J '15ra20___' -o 'logs/15ra20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra20___" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="20" --comet-enable --pbt-print'
sbatch --array=1-25 -n1 -p batch -J '15uc20___' -o 'logs/15uc20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20___" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print'
sbatch --array=1-25 -n1 -p batch -J '15uc20__s' -o 'logs/15uc20__s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc20__s" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="20" --comet-enable --pbt-print --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p batch -J '15gr20___' -o 'logs/15gr20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr20___" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="20" --comet-enable --pbt-print'
sbatch --array=1-25 -n1 -p batch -J '15so20___' -o 'logs/15so20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so20___" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="20" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15eg20___' -o 'logs/15eg20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg20___" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="20" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15es20___' -o 'logs/15es20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es20___" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="20" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15eu20___' -o 'logs/15eu20____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20___" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15eu20__s' -o 'logs/15eu20__s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu20__s" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="20" --comet-enable --pbt-print --suggest-ucb-incr-mode="stepped"'

# EPOCH
sbatch --array=1-25 -n1 -p batch -J '15ra7e___' -o 'logs/15ra7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15ra7e___" --pbt-members="15" --pbt-exploit-suggest="ran"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
sbatch --array=1-25 -n1 -p batch -J '15uc7e___' -o 'logs/15uc7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7e___" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
sbatch --array=1-25 -n1 -p batch -J '15uc7e__s' -o 'logs/15uc7e__s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15uc7e__s" --pbt-members="15" --pbt-exploit-suggest="ucb"   --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p batch -J '15gr7e___' -o 'logs/15gr7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15gr7e___" --pbt-members="15" --pbt-exploit-suggest="gr"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
sbatch --array=1-25 -n1 -p batch -J '15so7e___' -o 'logs/15so7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15so7e___" --pbt-members="15" --pbt-exploit-suggest="sm"    --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15eg7e___' -o 'logs/15eg7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eg7e___" --pbt-members="15" --pbt-exploit-suggest="e-gr"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15es7e___' -o 'logs/15es7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15es7e___" --pbt-members="15" --pbt-exploit-suggest="e-sm"  --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15eu7e___' -o 'logs/15eu7e____%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7e___" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print'
#sbatch --array=1-25 -n1 -p batch -J '15eu7e__s' -o 'logs/15eu7e__s_%A_%a.log' --wrap 'python -m tsucb.experiment --experiment-name="15eu7e__s" --pbt-members="15" --pbt-exploit-suggest="e-ucb" --pbt-target-steps="7" --cnn-steps-per-epoch="1" --comet-enable --pbt-print --suggest-ucb-incr-mode="stepped"'






## UCB: STEPPED INSTEAD OF EXPLOITED
#sbatch --array=1-10 -n1 -p ha -J '15_ucbS' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbSO' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbSO"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest-ucb-incr-mode="stepped" --pbt-disable-eager-step'

## STEPPED INSTEAD OF EXPLOITED and/or ONLY STEP EVERY EPOCH and/or READY AFTER 1 NOT 2
#sbatch --array=1-10 -n1 -p ha -J '15_ucbSR' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest-ucb-incr-mode="stepped" --pbt-members-ready-after="1"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbSE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest-ucb-incr-mode="stepped" --cnn-steps-per-epoch="1" --pbt-target-steps="7"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbSRE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbS"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --suggest-ucb-incr-mode="stepped" --cnn-steps-per-epoch="1" --pbt-target-steps="7" --pbt-members-ready-after="1"'

## DISABLED EAGER STEP
#sbatch --array=1-10 -n1 -p ha -J '15_ucbO' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbO"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --pbt-disable-eager-step'
#sbatch --array=1-10 -n1 -p ha -J '15_ranO' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranO"  --pbt-members="15"  --pbt-exploit-suggest="ran" --pbt-disable-eager-step'

## READY AFTER 1 NOT 2
#sbatch --array=1-10 -n1 -p ha -J '15_ranR' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranR"  --pbt-members="15"  --pbt-exploit-suggest="ran" --pbt-members-ready-after="1"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbR' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbR"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --pbt-members-ready-after="1"'

## ONLY STEP EVERY EPOCH + READY AFTER 1 NOT 2
#sbatch --array=1-10 -n1 -p ha -J '15_ranER' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranER"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbER' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbER"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1"'

## ONLY STEP EVERY EPOCH + READY AFTER 1 NOT 2 + REDUCED MAX LEARNING RATE 0.1->0.01
#sbatch --array=1-10 -n1 -p ha -J '15_ucb_ERA' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbERA"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucb_ERA' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbERA"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'

#sbatch --array=1-25 -n1 -p ha -J '25_ran_ERA' -o 'logs/25_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ran_ERA"  --pbt-members="25"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_ucb_ERA' -o 'logs/25_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ucb_ERA"  --pbt-members="25"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_ucb_ERAS' -o 'logs/25_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ucb_ERAS"  --pbt-members="25"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'
#sbatch --array=1-25 -n1 -p ha -J '25_gr_ERA' -o 'logs/25_gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_gr_ERA"  --pbt-members="25"  --pbt-exploit-suggest="gr" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_sm_ERA' -o 'logs/25_sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_sm_ERA"  --pbt-members="25"  --pbt-exploit-suggest="sm" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_e-gr_ERA' -o 'logs/25_e-gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-gr_ERA"  --pbt-members="25"  --pbt-exploit-suggest="e-gr" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_e-sm_ERA' -o 'logs/25_e-sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-sm_ERA"  --pbt-members="25"  --pbt-exploit-suggest="e-sm" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_e-ucb_ERA' -o 'logs/25_e-ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-ucb_ERA"  --pbt-members="25"  --pbt-exploit-suggest="e-ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01"'
#sbatch --array=1-25 -n1 -p ha -J '25_e-ucb_ERAS' -o 'logs/25_e-ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_e-ucb_ERAS"  --pbt-members="25"  --pbt-exploit-suggest="e-ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="10"  --pbt-members-ready-after="1" --cnn-lr-max="0.01" --suggest-ucb-incr-mode="stepped"'

## ONLY STEP EVERY EPOCH
#sbatch --array=1-10 -n1 -p ha -J '15_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranEL"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbEL"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'

## ONLY STEP EVERY EPOCH
#sbatch --array=1-10 -n1 -p ha -J '15_ranE' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranE"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="7"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbE' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbE"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="7"'

## ONLY STEP EVERY EPOCH - BUT MORE STEPS
#sbatch --array=1-10 -n1 -p ha -J '15_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ranEL"  --pbt-members="15"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucbEL"  --pbt-members="15"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
#sbatch --array=1-10 -n1 -p ha -J '25_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ranEL"  --pbt-members="25"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
#sbatch --array=1-10 -n1 -p ha -J '25_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ucbEL"  --pbt-members="25"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
#sbatch --array=1-10 -n1 -p ha -J '40_ranEL' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ranEL"  --pbt-members="40"  --pbt-exploit-suggest="ran" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'
#sbatch --array=1-10 -n1 -p ha -J '40_ucbEL' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ucbEL"  --pbt-members="40"  --pbt-exploit-suggest="ucb" --cnn-steps-per-epoch="1" --pbt-target-steps="15"'

## ORIG
#sbatch --array=1-10 -n1 -p ha -J '15_ran' -o 'logs/15_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ran"  --pbt-members="15"  --pbt-exploit-suggest="ran"'
#sbatch --array=1-10 -n1 -p ha -J '15_ucb' -o 'logs/15_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_ucb"  --pbt-members="15"  --pbt-exploit-suggest="ucb"'
#sbatch --array=1-10 -n1 -p ha -J '25_ran' -o 'logs/25_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ran"  --pbt-members="25"  --pbt-exploit-suggest="ran"'
#sbatch --array=1-10 -n1 -p ha -J '25_ucb' -o 'logs/25_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="25_ucb"  --pbt-members="25"  --pbt-exploit-suggest="ucb"'
#sbatch --array=1-10 -n1 -p ha -J '40_ran' -o 'logs/40_ran_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ran"  --pbt-members="40"  --pbt-exploit-suggest="ran"'
#sbatch --array=1-10 -n1 -p ha -J '40_ucb' -o 'logs/40_ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="40_ucb"  --pbt-members="40"  --pbt-exploit-suggest="ucb"'

## ORIG EXTRA
#sbatch --array=1-10 -n1 -p batch -J '15_gr' -o 'logs/15_gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_gr"  --pbt-members="15"  --pbt-exploit-suggest="gr"'
#sbatch --array=1-10 -n1 -p batch -J '15_sm' -o 'logs/15_sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_sm"  --pbt-members="15"  --pbt-exploit-suggest="sm"'
#sbatch --array=1-10 -n1 -p batch -J '15_e-gr' -o 'logs/15_e-gr_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_e-gr"  --pbt-members="15"  --pbt-exploit-suggest="e-gr"'
#sbatch --array=1-10 -n1 -p batch -J '15_e-sm' -o 'logs/15_e-sm_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_e-sm"  --pbt-members="15"  --pbt-exploit-suggest="e-sm"'
#sbatch --array=1-10 -n1 -p batch -J '15_e-ucb' -o 'logs/15_e-ucb_%A_%a.log' --wrap 'python -m tsucb.experiment  --comet-enable  --pbt-print  --experiment-name="15_e-ucb"  --pbt-members="15"  --pbt-exploit-suggest="e-ucb"'