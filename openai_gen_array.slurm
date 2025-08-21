#!/bin/bash
#SBATCH --job-name=openai_gen
#SBATCH --output=logs/openai_gen_%A_%a.out
#SBATCH --error=logs/openai_gen_%A_%a.err
#SBATCH --array=0-17
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=l40s          # or a100, etc.
#SBATCH --gres=gpu:1              # request 1 GPU

# make sure logs dir exists even if Slurm creates files elsewhere first
mkdir -p logs

# If your script reads OPENAI_API_KEY from env, export it here or set it before sbatch
# export OPENAI_API_KEY="sk-..."

TOTAL_VERBS=177
CHUNK_SIZE=10

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

for FILE_KEY in agent_location agent_instrument agent_patient all_roles; do
    echo "[INFO] Processing $FILE_KEY for verbs $START:$END (chunk $SLURM_ARRAY_TASK_ID)"
    python /ix1/xli/dgt12/openAI_generator_batch.py \
        --file "$FILE_KEY" \
        --start "$START" \
        --end "$END" \
        --chunk-id "$SLURM_ARRAY_TASK_ID"
done
