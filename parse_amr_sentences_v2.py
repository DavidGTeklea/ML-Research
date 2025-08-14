# parse_amr_sentences_v2_debug.py  (no f-strings)
# RUn this by saying python -u parse_amr_sentences_v2.py extracted_sentences_with_verbs.txt 2>&1 | tee amr_run.log to also get error logs
import sys, os, traceback
import penman
from transition_amr_parser.parse import AMRParser

def parse_line(line):
    if "|" not in line:
        return None, None
    verb, sent = line.split("|", 1)
    return verb.strip(), sent.strip()

def is_animate_heuristic(word):
    return bool(word) and word.lower() in set([
        "man","woman","child","soldier","person","teacher","doctor","boy","girl",
        "agent","nurse","student","officer","mother","father","adult","human"
    ])

def roles_from_amr(amr_graph_str):
    roles = {"Agent": False, "Patient": False, "Instrument": False, "Location": False}
    g = penman.decode(amr_graph_str)

    var2concept, edges = {}, {}
    for s, r, t in g.triples:
        if r == ":instance":
            var2concept[s] = t
        else:
            edges.setdefault(s, []).append((r, t))

    for _, r, _ in g.triples:
        if r == ":ARG0": roles["Agent"] = True
        elif r == ":ARG1": roles["Patient"] = True
        elif r == ":instrument": roles["Instrument"] = True
        elif r == ":location": roles["Location"] = True

    def dfs_ARG1_chain(var, depth, seen):
        if var in seen or depth > 4:
            return False
        seen.add(var)
        concept = var2concept.get(var, "")
        if is_animate_heuristic(concept):
            return True
        for rel, tgt in edges.get(var, []):
            if rel == ":ARG1" and tgt in var2concept:
                if dfs_ARG1_chain(tgt, depth + 1, seen):
                    return True
        return False

    if not roles["Agent"] and g.triples:
        root_var = g.triples[0][0]
        if dfs_ARG1_chain(root_var, 0, set()):
            roles["Agent"] = True
    return roles

def main():
    try:
        print("[BOOT] __name__ = {}".format(__name__)); sys.stdout.flush()
        print("[BOOT] argv = {}".format(sys.argv)); sys.stdout.flush()
        print("[BOOT] CWD  = {}".format(os.getcwd())); sys.stdout.flush()

        if len(sys.argv) < 2:
            print("Usage: python parse_amr_sentences_v2_debug.py <verb_sentence_file>"); sys.stdout.flush()
            sys.exit(1)

        in_path = sys.argv[1]
        print("[INFO] Input (given): {}".format(in_path)); sys.stdout.flush()
        in_abs = os.path.abspath(in_path)
        print("[INFO] Input (abs)  : {}".format(in_abs)); sys.stdout.flush()
        if not os.path.exists(in_path):
            print("[ERROR] Input file not found."); sys.stdout.flush()
            sys.exit(2)

        out_path = in_path.rsplit(".", 1)[0] + "_amr_detector_output.txt"
        out_abs = os.path.abspath(out_path)
        print("[INFO] Output (abs) : {}".format(out_abs)); sys.stdout.flush()

        # Create/clear output immediately so you can `ls` it
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("[RUN-START]\n")
        print("[INFO] Output file created/cleared."); sys.stdout.flush()

        print("[INFO] Loading AMR model AMR3-structbart-L ..."); sys.stdout.flush()
        parser = AMRParser.from_pretrained("AMR3-structbart-L")
        print("[INFO] Model loaded."); sys.stdout.flush()

        total = 0; written = 0; skipped = 0

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "a", encoding="utf-8") as fout:

            for raw in fin:
                raw = raw.strip()
                if not raw:
                    continue
                total += 1
                verb, sentence = parse_line(raw)
                if not verb or not sentence:
                    skipped += 1
                    fout.write("\n[SKIP] Malformed line: {}\n".format(raw))
                    continue

                if total % 25 == 0:
                    print("[INFO] processed {} lines...".format(total)); sys.stdout.flush()

                try:
                    tokens, _ = parser.tokenize(sentence)
                    annots, machines = parser.parse_sentence(tokens)
                    amr_penman = machines.get_amr().to_penman(jamr=False, isi=True)
                    roles = roles_from_amr(amr_penman)

                    fout.write("==== Verb: {} ====\n".format(verb))
                    fout.write("Sentence: {}\n".format(sentence))
                    fout.write("Detector: AMR\n")
                    fout.write("Agent: {}\n".format(roles["Agent"]))
                    fout.write("Patient: {}\n".format(roles["Patient"]))
                    fout.write("Instrument: {}\n".format(roles["Instrument"]))
                    fout.write("Location: {}\n".format(roles["Location"]))
                    fout.write("AMR:\n{}\n\n".format(amr_penman))
                    written += 1

                except Exception as e:
                    fout.write("==== Verb: {} ====\n".format(verb))
                    fout.write("Sentence: {}\n".format(sentence))
                    fout.write("Detector: AMR\n")
                    fout.write("Agent: False\nPatient: False\nInstrument: False\nLocation: False\n")
                    fout.write("[ERROR] {}\n\n".format(e))

        print("[DONE] Lines read: {}, written: {}, skipped: {}.".format(total, written, skipped)); sys.stdout.flush()
        print("[DONE] Output saved to: {}".format(out_abs)); sys.stdout.flush()

    except Exception:
        tb = traceback.format_exc()
        # make sure you see the failure even if stdout is buffered by the environment
        try:
            sys.stderr.write(tb + "\n"); sys.stderr.flush()
        except Exception:
            pass
        # also dump it into the output file if we already made it
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n[CRASH]\n{}\n".format(tb))
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
