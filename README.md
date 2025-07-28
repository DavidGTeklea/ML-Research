# LLM-Research
This repo organizes the files that I run on Pitt CRC's cluster and is related to the work that I do for Dr. Lorraine Xi Liang at the University of Pittsburgh

# Verb Scenario Generation Pipeline with LLMs and AMR Parsing

This project explores how Large Language Models (LLMs) generate plausible event-based scenarios for a curated set of verbs, with a focus on both scalability and semantic consistency. We integrate OpenAI and Llama-2 models with AMR parsing to assess the quality of generated outputs against human annotators.

## 🚀 Features

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
  - Human annotation agreement rate of **~92%**
  - Verb-wise plausibility statistics for model benchmarking

- **Semantic Validation via AMR**:
  - Parses LLM outputs to assess presence of agent, patient, location, and instrument
  - Compares parser coverage to expected semantic roles per verb

## 🧠 Models Used

| Model       | Role                                      |
|-------------|-------------------------------------------|
| GPT-4 / GPT-4o / GPT-3.5 | Scenario generation & comparison |
| LLaMA-2 (13B, quantized) | Open-source scenario generation  |
| IBM Transition AMR Parser | Semantic role extraction        |

## 🛠 Setup

- Python 3.11+
- Dependencies:
  - `openai`, `transformers`, `torch`, `pandas`, `slurm`, `fitz`, `nltk`, `amrlib` (IBM parser)
- GPU Runtime: Pitt CRC interactive or batch jobs (`sbatch`)
- HuggingFace access token required for LLaMA-2

## 🖥 Usage

1. Generate scenarios:
    ```bash
    python generate_scenarios.py --verbs verb_list.csv --model gpt-4o
    ```

2. Run AMR parsing:
    ```bash
    python run_amr_parser.py --input model_outputs.txt --output amr_graphs.txt
    ```

3. Run plausibility scoring:
    ```bash
    python score_scenarios.py --input model_outputs.txt
    ```

4. Launch on Pitt CRC:
    ```bash
    sbatch batch_jobs/generate_verbs.slurm
    ```

## 📈 Example Output

- Scenario (for verb "whisper"):
