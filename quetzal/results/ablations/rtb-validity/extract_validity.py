import sys, os, glob, csv, statistics
sys.path.insert(0, "figures")
from make_fig18_training_curves import _run_name, _read_history, series, WANDB_ROOT

names_wanted = set(l.strip() for l in open('/tmp/rtb_names.txt') if l.strip())

# Build name -> list of (run_dir, n_config_lines) once, single pass over config.yaml files
candidates = {}
for config_path in glob.glob(os.path.join(WANDB_ROOT, "run-*", "files", "config.yaml")):
    nm = _run_name(config_path)
    if nm not in names_wanted:
        continue
    run_dir = os.path.dirname(os.path.dirname(config_path))
    candidates.setdefault(nm, []).append(run_dir)

print(f"[index] {len(candidates)} / {len(names_wanted)} names have at least one wandb run dir", file=sys.stderr)

out = []
missing = []
for n in sorted(names_wanted):
    dirs = candidates.get(n, [])
    if not dirs:
        missing.append(n)
        continue
    best_rows, best_dir = None, None
    for run_dir in dirs:
        wandb_files = glob.glob(os.path.join(run_dir, "run-*.wandb"))
        if not wandb_files:
            continue
        rows = _read_history(wandb_files[0])
        if best_rows is None or len(rows) > len(best_rows):
            best_rows, best_dir = rows, run_dir
    if not best_rows:
        missing.append(n)
        continue
    xs, ys = series(best_rows, "train/valid_frac")
    if not ys:
        missing.append(n)
        continue
    tail_n = max(3, int(len(ys) * 0.2))
    tail_mean = statistics.mean(ys[-tail_n:])
    full_mean = statistics.mean(ys)
    out.append((n, len(ys), full_mean, tail_mean))

print(f"found {len(out)} / {len(names_wanted)}; missing {len(missing)}", file=sys.stderr)
for m in missing:
    print("MISSING:", m, file=sys.stderr)

with open('/tmp/rtb_validity.csv','w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['name','n_steps','full_mean_valid_frac','tail_mean_valid_frac'])
    w.writerows(out)
print("wrote", len(out), "rows", file=sys.stderr)
