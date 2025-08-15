# evaluate_role_detectors.py  (normalizes GPT spans -> booleans)
import re
import argparse
import pandas as pd
from collections import defaultdict, Counter

ROLE_COLS = ["Agent", "Patient", "Instrument", "Location"]
HEADER_RE = re.compile(r"^==== Verb:\s*(.+?)\s*====\s*$")
LINE_KV_RE = re.compile(r"^(Agent|Patient|Instrument|Location):\s*(.*)$")

def _truthy_from_gpt_value(v: str) -> bool:
    """
    Normalize GPT value text -> boolean presence.
    Accept anything non-empty and not an explicit 'none/false/0' as True.
    """
    if v is None:
        return False
    s = str(v).strip().strip('"').strip("'").lower()
    return s not in ("", "none", "null", "n/a", "na", "false", "0")

def _truthy_from_amr_value(v: str) -> bool:
    return str(v).strip().lower() == "true"

def load_gold(excel_path, sheet_name=0, prefer_block="MWD"):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    # Case A: already flat (Verb + role columns)
    if "Verb" in df.columns and all(r in df.columns for r in ROLE_COLS):
        gold = {}
        for _, row in df.iterrows():
            verb = str(row["Verb"]).strip().lower()
            if not verb:
                continue
            truth = {r: bool(int(row.get(r, 0))) for r in ROLE_COLS}
            gold[verb] = truth
        return gold

    # Case B: MWD/UW layout w/ a label row at index 0
    lemma_col = None
    for cand in ["verbs", "Verb", "VERB"]:
        if cand in df.columns:
            lemma_col = cand
            break
    if lemma_col is None:
        raise ValueError("Could not find a lemma column ('verbs'/'Verb').")

    header_row_idx = 0
    label_map = {
        "agent-compatible": "Agent",
        "patient-compatible": "Patient",
        "instrument-compatible": "Instrument",
        "location-compatible": "Location",
    }
    cols = list(df.columns)

    def labels_after(anchor_col):
        out = {}
        if anchor_col not in df.columns:
            return out
        start_idx = cols.index(anchor_col)
        for c in cols[start_idx:]:
            lbl = str(df.loc[header_row_idx, c]).strip().lower()
            if lbl in label_map and label_map[lbl] not in out:
                out[label_map[lbl]] = c
            if len(out) == 4:
                break
        return out

    role_to_col = {}
    if prefer_block.upper() == "MWD" and "MWD" in df.columns:
        role_to_col = labels_after("MWD")
    elif prefer_block.upper() == "UW" and "UW" in df.columns:
        role_to_col = labels_after("UW")

    if len(role_to_col) < 4:
        # fallback: first 4 labeled columns anywhere
        for c in cols:
            lbl = str(df.loc[header_row_idx, c]).strip().lower()
            if lbl in label_map and label_map[lbl] not in role_to_col:
                role_to_col[label_map[lbl]] = c
            if len(role_to_col) == 4:
                break

    if len(role_to_col) < 4:
        raise ValueError("Could not infer the 4 role columns; found {}".format(role_to_col))

    data = df.iloc[header_row_idx + 1:].copy()

    gold = {}
    for _, row in data.iterrows():
        verb = str(row[lemma_col]).strip().lower()
        if not verb or verb == "target_name":
            continue
        truth = {}
        for r in ROLE_COLS:
            val = row.get(role_to_col[r], 0)
            try:
                truth[r] = bool(int(val))
            except Exception:
                s = str(val).strip().lower()
                truth[r] = s not in ("", "0", "nan", "none", "false")
        gold[verb] = truth
    return gold

def parse_gpt_blocks(path):
    """
    Returns list[(verb, {role: bool})]
    GPT values like 'doctor', '"a brush"', 'None', '' are normalized -> booleans.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        cur_verb, roles = None, {}
        for raw in f:
            line = raw.strip()
            m = HEADER_RE.match(line)
            if m:
                if cur_verb and roles:
                    out.append((cur_verb, {r: _truthy_from_gpt_value(v) for r, v in roles.items()}))
                cur_verb, roles = m.group(1).strip().lower(), {}
                continue
            m2 = LINE_KV_RE.match(line)
            if m2:
                roles[m2.group(1)] = m2.group(2)
        if cur_verb and roles:
            out.append((cur_verb, {r: _truthy_from_gpt_value(v) for r, v in roles.items()}))
    return out

def parse_amr_blocks(path):
    """
    Returns list[(verb, {role: bool})] from AMR booleans.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        cur_verb, roles = None, {}
        for raw in f:
            line = raw.strip()
            m = HEADER_RE.match(line)
            if m:
                if cur_verb and roles:
                    out.append((cur_verb, {r: _truthy_from_amr_value(v) for r, v in roles.items()}))
                cur_verb, roles = m.group(1).strip().lower(), {}
                continue
            m2 = LINE_KV_RE.match(line)
            if m2:
                roles[m2.group(1)] = m2.group(2)
        if cur_verb and roles:
            out.append((cur_verb, {r: _truthy_from_amr_value(v) for r, v in roles.items()}))
    return out

def aggregate_verb_level(preds):
    """
    Collapse sentence-level -> verb-level by OR.
    preds: list[(verb, {role: bool})]
    returns dict verb -> {role: bool}
    """
    agg = defaultdict(lambda: {r: False for r in ROLE_COLS})
    for verb, rolemap in preds:
        for r in ROLE_COLS:
            agg[verb][r] = agg[verb][r] or rolemap.get(r, False)
    return agg

def compute_metrics(preds, gold):
    agg = aggregate_verb_level(preds)
    counts = {r: Counter() for r in ROLE_COLS}
    for verb, truth in gold.items():
        pred = agg.get(verb, {r: False for r in ROLE_COLS})
        for r in ROLE_COLS:
            t, p = truth[r], pred[r]
            if t and p: counts[r]["TP"] += 1
            elif not t and p: counts[r]["FP"] += 1
            elif t and not p: counts[r]["FN"] += 1

    def prf(c):
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        return prec, rec, f1

    rows = []
    for r in ROLE_COLS:
        p, r_, f = prf(counts[r])
        rows.append((r, counts[r]["TP"], counts[r]["FP"], counts[r]["FN"], round(p,3), round(r_,3), round(f,3)))
    macro_f1 = sum(x[6] for x in rows) / len(rows) if rows else 0.0

    # Additional quick stats: how many predicted positives (verb-level)
    agg = aggregate_verb_level(preds)
    pred_pos = {r: sum(1 for v in agg.values() if v[r]) for r in ROLE_COLS}
    gold_pos = {r: sum(1 for v in gold.values() if v[r]) for r in ROLE_COLS}

    return rows, macro_f1, pred_pos, gold_pos

def pretty_print(tag, rows, macro, pred_pos, gold_pos):
    print("\n[{}] Role\tTP\tFP\tFN\tPrec\tRec\tF1\t(PredPos/GoldPos)".format(tag))
    for row in rows:
        r, tp, fp, fn, pr, rc, f1 = row
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}/{}".format(r, tp, fp, fn, pr, rc, f1, pred_pos[r], gold_pos[r]))
    print("[{}] Macro-F1: {:.3f}".format(tag, macro))

def union_preds(p1, p2):
    vset = set([v for v,_ in p1]) | set([v for v,_ in p2])
    d1 = {v:m for v,m in p1}; d2 = {v:m for v,m in p2}
    out = []
    for v in vset:
        m = {r: (d1.get(v,{}).get(r,False) or d2.get(v,{}).get(r,False)) for r in ROLE_COLS}
        out.append((v,m))
    return out

def intersect_preds(p1, p2):
    vset = set([v for v,_ in p1]) | set([v for v,_ in p2])
    d1 = {v:m for v,m in p1}; d2 = {v:m for v,m in p2}
    out = []
    for v in vset:
        m = {r: (d1.get(v,{}).get(r,False) and d2.get(v,{}).get(r,False)) for r in ROLE_COLS}
        out.append((v,m))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="NLP_project_verb_list_MWD.xlsx")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--block", default="MWD", choices=["MWD","UW"])
    ap.add_argument("--gpt", default="gpt_role_detector_output.txt")
    ap.add_argument("--amr", default="extracted_sentences_with_verbs_amr_detector_output.txt")
    args = ap.parse_args()

    print("[INFO] Loading gold from {} (sheet={}, block={})".format(args.excel, args.sheet, args.block))
    gold = load_gold(args.excel, args.sheet, args.block)
    print("[INFO] Gold verbs: {}".format(len(gold)))

    print("[INFO] Parsing GPT output: {}".format(args.gpt))
    gpt_preds = parse_gpt_blocks(args.gpt)
    rows, macro, pred_pos, gold_pos = compute_metrics(gpt_preds, gold)
    pretty_print("GPT", rows, macro, pred_pos, gold_pos)

    print("[INFO] Parsing AMR output: {}".format(args.amr))
    amr_preds = parse_amr_blocks(args.amr)
    rows, macro, pred_pos, gold_pos = compute_metrics(amr_preds, gold)
    pretty_print("AMR", rows, macro, pred_pos, gold_pos)

    # Ensembles
    u = union_preds(gpt_preds, amr_preds)
    rows, macro, pred_pos, gold_pos = compute_metrics(u, gold)
    pretty_print("UNION (GPT ∪ AMR)", rows, macro, pred_pos, gold_pos)

    i = intersect_preds(gpt_preds, amr_preds)
    rows, macro, pred_pos, gold_pos = compute_metrics(i, gold)
    pretty_print("INTERSECTION (GPT ∩ AMR)", rows, macro, pred_pos, gold_pos)

if __name__ == "__main__":
    main()
