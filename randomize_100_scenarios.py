import random

# Config
INPUT_FILE = "extracted_sentences_only.txt"
OUTPUT_FILE = "sampled_scenarios_for_annotation.txt"
SAMPLE_SIZE = 100
SEED = 42  # for reproducibility

# Load sentences
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Safety check
if SAMPLE_SIZE > len(sentences):
    raise ValueError(f"Requested sample size ({SAMPLE_SIZE}) exceeds available sentences ({len(sentences)}).")

# Sample randomly
random.seed(SEED)
sampled = random.sample(sentences, SAMPLE_SIZE)

# Write to plain text file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sentence in sampled:
        f.write(sentence + "\n")

print(f"[INFO] Wrote {SAMPLE_SIZE} sampled sentences to '{OUTPUT_FILE}'")
