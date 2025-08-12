import re
from difflib import SequenceMatcher

def clean_key(key):
    # Removes leading digits and dots, e.g., "3. agent"-->"agent"
    return re.sub(r"^\d+\.\s*", "", key.strip().lower())

def parse_blocks(lines):
    blocks = []
    block = {}
    for line in lines:
        line = line.strip()
        if not line:
            if block:
                blocks.append(block)
                block = {}
            continue
        if line.lower().startswith("scenario"):
            block = {"sentence": line.split(":", 1)[1].strip()}
        elif ":" in line:
            key, val = line.split(":", 1)
            cleaned_key = clean_key(key)
            block[cleaned_key] = val.strip()
    if block:
        blocks.append(block)
    return blocks


def compare_strs(a, b, threshold=0.85):
    a = a.lower().strip()
    b = b.lower().strip()
    return SequenceMatcher(None, a, b).ratio() >= threshold

# === Load files ===
with open("extracted_scenarios_with_roles_evaluator.txt", "r", encoding="utf-8") as f:
    gold_lines = f.read().splitlines()

with open("extracted_scenarios_with_roles_constructor.txt", "r", encoding="utf-8") as f:
    pred_lines = f.read().splitlines()

gold_blocks = parse_blocks(gold_lines)
pred_blocks = parse_blocks(pred_lines)

roles = ["agent", "patient", "instrument", "location"]
role_correct = {r: 0 for r in roles}
role_total = {r: 0 for r in roles}

print("=== DEBUG: Checking per-scenario matches (70% threshold) ===")
for idx, (gold, pred) in enumerate(zip(gold_blocks, pred_blocks)):
    print(f"\n[Scenario {idx+1}]")
    for role in roles:
        gval = gold.get(role)
        pval = pred.get(role)
        if gval:
            role_total[role] += 1
            if pval:
                print(f"[THRESHOLD DEBUG] Agent: gold='{gval}' vs pred='{pval}' → comparing? {'yes' if pval else 'no'}")

                match = compare_strs(gval, pval, threshold=0.7)
                if match:
                    role_correct[role] += 1
                print(f"  {role.title()}:")
                print(f"    Gold: {gval}")
                print(f"    Pred: {pval}")
                print(f"    Match: {match}")
            else:
                print(f"  [WARN] Missing {role} in predicted block.")
        else:
            print(f"  [WARN] Missing {role} in gold block.")

# === Summary ===
print("\n=== Role Matching Evaluation ===")
for role in roles:
    correct = role_correct[role]
    total = role_total[role]
    acc = correct / total if total else 0.0
    print(f"{role.title():<10}: {correct}/{total} correct ({acc:.2%})")
