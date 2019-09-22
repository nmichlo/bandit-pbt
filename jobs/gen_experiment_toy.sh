
# UTIL
SCRIPTS_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)")/scripts"

if [ -n "$1" ]; then
    FILE="$1"
else
    FILE="SUBMIT_EXPERIMENT_TOY.sh"
fi

# REMOVE EXISTING
rm -r "$FILE" || true

# ============================================================= #

# SLURM
RUN_SCRIPT="sbatch -n1 -p ha -J {pbt-exploiter}-{i} -o logs/{pbt-exploiter}-{i}-%j.log --wrap"


# ============================================================= #

REPEATS="10000"
STEPS="30"
MEMBERS="10 20 40"
READY_AFTER="2 5"
UCB_C="0.025 0.05 0.1 0.2 0.5 1.0 2.0"

# ============================================================= #

# GENERATE - TS
EXPERIMENT_SCRIPT=\
"python -m tsucb.experiments.experiment_toy"\
" --enable-comet"\
" --experiment-name={pbt-exploiter}_{i}"\
" --experiment-repeats={experiment-repeats}"\
" --pbt-steps={pbt-steps}"\
" --pbt-members={pbt-members}"\
" --pbt-members-ready-after={pbt-members-ready-after}"\
" --pbt-exploiter={pbt-exploiter}"\
" --ucb-c={ucb-c}"

python "$SCRIPTS_DIR/grid.py" -v -o "$FILE" "$RUN_SCRIPT '$EXPERIMENT_SCRIPT'" \
      -c experiment-repeats:      $REPEATS     \
      -c pbt-steps:               $STEPS       \
      -c pbt-members:             $MEMBERS     \
      -c pbt-members-ready-after: $READY_AFTER \
      -c pbt-exploiter:           'ts'         \
      -c ucb-c:                   $UCB_C       \

# ============================================================= #

# GENERATE - UCB
EXPERIMENT_SCRIPT=\
"python -m tsucb.experiments.experiment_toy"\
" --enable-comet"\
" --experiment-name={pbt-exploiter}_{i}"\
" --experiment-repeats={experiment-repeats}"\
" --pbt-steps={pbt-steps}"\
" --pbt-members={pbt-members}"\
" --pbt-members-ready-after={pbt-members-ready-after}"\
" --pbt-exploiter={pbt-exploiter}"\
" --ucb-incr-mode={ucb-incr-mode}"\
" --ucb-reset-mode={ucb-reset-mode}"\
" --ucb-subset-mode={ucb-subset-mode}"\
" --ucb-normalise-mode={ucb-normalise-mode}"\
" --ucb-c={ucb-c}"

python "$SCRIPTS_DIR/grid.py" -v -o "$FILE" "$RUN_SCRIPT '$EXPERIMENT_SCRIPT'" \
      -c experiment-repeats:      $REPEATS     \
      -c pbt-steps:               $STEPS       \
      -c pbt-members:             $MEMBERS     \
      -c pbt-members-ready-after: $READY_AFTER \
      -c pbt-exploiter:           'ucb'        \
      -c ucb-c:                   $UCB_C       \
      -c ucb-incr-mode:           'exploited'  'stepped'                                   \
      -c ucb-reset-mode:          'exploited'  'explored'       'explored_or_exploited'    \
      -c ucb-subset-mode:         'all'        'exclude_bottom' 'top'                      \
      -c ucb-normalise-mode:      'population' 'subset'                                    \

# ============================================================= #

FILE_BEGINING="#!/bin/bash$(printf "\n")"
echo "$FILE_BEGINING" | cat - "$FILE" > temp && mv temp "$FILE"
chmod +x "$FILE"