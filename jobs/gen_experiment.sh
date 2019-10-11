
# ============================================================= #
# UTIL                                                          #
# ============================================================= #

SCRIPTS_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)")/scripts"

# ============================================================= #
# OUTPUT FILE                                                   #
# ============================================================= #

if [ -n "$1" ]; then
    FILE="$1"
else
    FILE="SUBMIT_EXPERIMENT.sh"
fi

# REMOVE EXISTING
rm -r "$FILE" || true

# CREATE FILE FOR APPENDING:
printf "#!/bin/bash\n\n" > "$FILE"
chmod +x "$FILE"

# ============================================================= #
# SLURM COMMAND                                                 #
# ============================================================= #

# %j is the job id
# %A is the array job id
# %a is the array task id

RUN_SCRIPT="sbatch --array=1-{slurm-array-size} -n1 -p ha -J '{experiment-name}' -o 'logs/{experiment-name}_%A_%a.log' --wrap"

# ============================================================= #
# EXPERIMENT COMMAND                                            #
# ============================================================= #

#EXPERIMENT_SCRIPT=\
#'python -m tsucb.experiment'\
#'  --experiment-repeats="{experiment-repeats}"'\
#'  --experiment-name="{experiment-name}"'\
#'  --experiment-id="{experiment-id}"'\
#'  --experiment-type="{experiment-type}"'\
#'  --experiment-seed="{experiment-seed}"'\
#'  --cnn-dataset="{cnn-dataset}"'\
#'  --cnn-batch-size="{cnn-batch-size}"'\
#'  --cnn-steps-per-epoch="{cnn-steps-per-epoch}"'\
#'  --pbt-print'\
#'  --pbt-target-steps="{pbt-target-steps}"'\
#'  --pbt-members="{pbt-members}"'\
#'  --pbt-members-ready-after="{pbt-members-ready-after}"'\
#'  --pbt-exploit-strategy="{pbt-exploit-strategy}"'\
#'  --pbt-exploit-suggest="{pbt-exploit-suggest}"'\
#'  --suggest-ucb-incr-mode="{suggest-ucb-incr-mode}"'\
#'  --suggest-ucb-c="{suggest-ucb-c}"'\
#'  --suggest-softmax-temp="{suggest-softmax-temp}"'\
#'  --suggest-eps="{suggest-eps}"'\
#'  --strategy-ts-ratio-top="{strategy-ts-ratio-top}"'\
#'  --strategy-ts-ratio-bottom="{strategy-ts-ratio-bottom}"'\
#'  --strategy-tt-confidence="{strategy-tt-confidence}"'\
#'  --comet-enable'\
#'  --comet-project-name="{comet-project-name}"'\

EXPERIMENT_SCRIPT=\
'python -m tsucb.experiment'\
'  --comet-enable'\
'  --pbt-print'\
'  --experiment-name="{experiment-name}"'\
'  --pbt-members="{pbt-members}"'\
'  --pbt-exploit-suggest="{pbt-exploit-suggest}"'\

# ============================================================= #
# GENERATE                                                      #
# ============================================================= #

SLURM_REPEATS=10
declare -a POPULATION_SIZES=("15" "25" "40")
declare -a SUGGESTORS=("ran" "ucb")

for size in "${POPULATION_SIZES[@]}"; do
    for sugg in "${SUGGESTORS[@]}"; do
        python "$SCRIPTS_DIR/grid.py" -v -o "$FILE" "$RUN_SCRIPT '$EXPERIMENT_SCRIPT'" \
            -c experiment-name:     "${size}_$sugg" \
            -c pbt-members:         "$size" \
            -c pbt-exploit-suggest: "$sugg" \
            -c slurm-array-size:    "$SLURM_REPEATS"
    done
done

echo >> "$FILE"
declare -a SUGGESTORS=("gr" "sm" "e-gr" "e-sm" "e-ucb")

for size in "${POPULATION_SIZES[@]}"; do
    for sugg in "${SUGGESTORS[@]}"; do
        python "$SCRIPTS_DIR/grid.py" -v -o "$FILE" "$RUN_SCRIPT '$EXPERIMENT_SCRIPT'" \
            -c experiment-name:     "${size}_$sugg" \
            -c pbt-members:         "$size" \
            -c pbt-exploit-suggest: "$sugg" \
            -c slurm-array-size:    "$SLURM_REPEATS"
    done
done
