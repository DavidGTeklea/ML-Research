# LLM-Research
This repo organizes the files that I run on Pitt CRC's cluster and is related to the work that I do for Dr. Lorraine Xi Liang at the University of Pittsburgh

# Verb Scenario Generation Pipeline with LLMs and AMR Parsing

This project explores how Large Language Models (LLMs) generate plausible event-based scenarios for a curated set of verbs, with a focus on both scalability and semantic consistency. We integrate OpenAI and Llama-2 models with AMR parsing to assess the quality of generated outputs against human annotators.

## Features

- **End-to-End Pipeline**: Automates the process of:
  - Loading verbs from structured spreadsheets
  - Generating scenarios with GPT-4 / GPT-3.5 / GPT-4o or LLaMA-2
  - Writing model outputs to text
  - Postprocessing and plausibility scoring
  - Parsing outputs using IBM’s Transition AMR Parser

- **Cluster-Scale LLM Inference**:
  - Utilizes Pitt CRC’s Slurm batch system for distributed, parallel scenario generation
  - Efficient 13B Llama-2 execution via 8-bit quantization and `device_map=auto`
  - Reduced total runtime from **15 minutes → under 5 minutes per verb set**

- **Quantitative Evaluation**:
  - Numerical plausibility scores (1–10 scale)
  - Verb-wise plausibility statistics for model benchmarking

- **Semantic Validation via AMR**:
  - Parses LLM outputs to assess presence of agent, patient, location, and instrument
  - Compares parser coverage to expected semantic roles per verb

## Models Used

| Model       | Role                                      |
|-------------|-------------------------------------------|
| GPT-4 / GPT-4o / GPT-3.5 | Scenario generation & comparison |
| LLaMA-2 (13B, quantized) | Open-source scenario generation  |
| IBM Transition AMR Parser | Semantic role extraction        |

## Setup

- Python 3.11+
- Dependencies:
  - `openai`, `transformers`, `torch`, `pandas`, `slurm`, `fitz`, `nltk`, `amrlib` (IBM parser)
- GPU Runtime: Pitt CRC interactive or batch jobs (`sbatch`)
- HuggingFace access token required for LLaMA-2

  This project runs across two environments:

### LLM Scenario Generation (e.g. OpenAI or LLaMA-2)

For Pitt CRC users running on a Jupyter OnDemand or interactive session:

```bash
# Load Python 3.11 with pip enabled
module load python/ondemand-jupyter-python3.11

# Install dependencies (into ~/.local/)
python -m pip install --user \
  transformers accelerate sentencepiece pandas openpyxl huggingface_hub openai==0.28 accelerate

# Set your OpenAI API Key (replace with your own)
export OPENAI_API_KEY="sk-WHATEVER-YOUR-KEY-IS
```

### AMR Parser Installation

Create a clean conda environment (Python 3.8) as recommended by the [Transition AMR Parser GitHub](https://github.com/sinantie/transition-amr-parser):

```bash
conda create -y -p ./amr_env python=3.8
conda activate ./amr_env
```

```bash
pip install transition-neural-parser

# Required for CUDA acceleration
pip install --no-index torch-scatter \
   -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
```

## Usage

1. Generate scenarios:
    ```bash
    python generate_scenarios.py --verbs verb_list.csv (optional argument)
    ```

2. Run AMR parsing:
    ```bash
    python parse_amr_sentences.py generated_sentences.txt 
    ```
## Example Output

- Scenario (for verb "whisper"):
- Parsed Roles:
- Agent: nurse
- Patient: patient
- Location: hospital room
- Instrument: soft tone
- Human Plausibility: 9.3/10
- AMR Role Coverage: 4/4

## Results

- GPT-4o achieves the highest average plausibility (9.1) across verbs.
- Average runtime per verb reduced by over 66% with optimized parallelization.

## Future Work

- Expand verb set to include less plausible scenarios
- Formalize annotation schema and increase annotator pool
- Develop semantic fidelity metric beyond role count

Credits

Research by **David Teklea**  
Advised by Dr. Lorraine Li  
University of Pittsburgh, Summer 2025 to present Day  
Project support via Pitt CRC & OpenAI API

---


