# -*- coding: utf-8 -*-
import os
import sys
import argparse
import openai
import json

# ==== Configuration ====
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("Error: OPENAI_API_KEY not set in environment.")

# ==== Prompt Template (Prompt 2.0 Style) ====
def build_prompt(scenario):
    return f"""Assume you are a linguistics and ML expert. There are four different roles that may be used in a scenario:

- agent (who/what performs the action; must be specific and unique across examples),
- patient (who/what is the recipient of the action; must differ in each case),
- instrument (the means of performing the action),
- location (where/direction of the action).

Given the following sentence, identify the roles that are present. If a role is not present, return "None" for that role.

Sentence: "{scenario}"

Respond with this format exactly:
Agent: ...
Patient: ...
Instrument: ...
Location: ...
"""

# ==== Argument Parser ====
parser = argparse.ArgumentParser(description="Identify semantic roles in a list of natural language scenarios using OpenAI API.")
parser.add_argument(
    "--input",
    type=str,
    default="extracted_sentences_only.txt",
    help="Path to input text file containing one scenario per line."
)
parser.add_argument(
    "--output",
    type=str,
    default="extracted_scenarios_with_roles_evaluator.txt",
    help="Path to output file where role-annotated results will be written."
)
args = parser.parse_args()

# ==== Read Input Scenarios ====
with open(args.input, "r", encoding="utf-8") as f:
    scenarios = [line.strip() for line in f if line.strip()]

# ==== Process and Generate Role Annotations ====
with open(args.output, "w", encoding="utf-8") as fout:
    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] Processing: {scenario}")
        prompt = build_prompt(scenario)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )
            output_text = response.choices[0].message.content.strip()

            # Write output to file
            fout.write(f"Scenario {i}: {scenario}\n")
            fout.write(output_text + "\n\n")
        except Exception as e:
            print(f"Error on scenario {i}: {e}")
            fout.write(f"Scenario {i}: {scenario}\nError: {e}\n\n")

print(f"\nDone. Output written to {args.output}")
