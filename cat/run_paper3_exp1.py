"""
run_paper3_exp1.py — Paper 3, Experiment 1.
===========================================
ATTACK-INCREMENTAL continual learning + PGD-TRADES adversarial training,
run INDEPENDENTLY on each of the three datasets (CICIoT, CICIoMT, CICIoV).

For each (backbone, seed, dataset):
  1. Load the dataset in multi-class mode and split attacks into a
     class-incremental sequence by attack FAMILY (task_splitter).
  2. Train the backbone task-by-task with ContinualAdversarialTrainer
     (CL terms: EWC + KD + replay; AT term: PGD-TRADES).
  3. After each task t, evaluate clean accuracy on every task seen so far
     -> build the T x T task-accuracy matrix.
  4. Compute continual-learning metrics: Average Accuracy (AA),
     Backward Transfer (BWT, forgetting), and per-task robustness.
  5. Evaluate robust accuracy under the four-attack suite on the full
     validation set.

The headline comparison is AAM-TRANS vs vanilla Transformer: the vanilla
backbone (no EWC/KD memory) is expected to suffer catastrophic forgetting
(BWT << 0), whereas AAM-TRANS retains earlier-attack knowledge.

Usage:
  python run_paper3_exp1.py --quick                     # smoke test
  python run_paper3_exp1.py                              # full run
  python run_paper3_exp1.py --run-dir runs/exp1/<TS>    # resume
"""
from __future__ import annotations

import argparse, csv, logging, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATASET_PATHS, DATASET_ORDER, LABEL_COLUMN, BENIGN_LABELS,
                    DATASET_SAMPLE_FRACTIONS, TEST_SIZE, SEQ_LEN, SEQ_NOISE_STD)
from data_loader import load_dataset
from models import (AAMTransIDS, StandardTransformerIDS, LSTMModel, DNNModel,
                    RobustDenoiseTransformerIDS)
from attacks import AdversarialAttacker, FeatureConstraintClassifier
from continual_adversarial import (ContinualAdversarialTrainer,
                                    BiasCorrection, BiCWrapped,
                                    fit_bias_correction)
from task_splitter import split_into_attack_tasks
from evaluation import evaluate_robustness
from incremental_metrics import (compute_average_accuracy,
                                  compute_backward_transfer,
                                  compute_remembering_rate,
                                  compute_catastrophic_forgetting_index)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("p3exp1")


# ──────────────────────────────────────────────────────────────────────────
def build_model(backbone, n_features, n_classes, seq_len):
    aam = dict(input_dim=n_features, d_model=96, n_heads=4, n_layers=3,
               d_ff=384, n_classes=n_classes, max_seq_len=seq_len, dropout=0.1)
    if backbone == "aam_trans":          return AAMTransIDS(**aam)
    if backbone == "aam_trans_no_gates": return AAMTransIDS(use_adaptive_gates=False, **aam)
    if backbone == "aam_trans_no_anomaly": return AAMTransIDS(use_anomaly_head=False, **aam)
    if backbone in ("transformer", "transformer_cl"):
        # transformer_cl is the SAME architecture as transformer but receives
        # the continual-learning machinery (EWC/KD/replay) at training time,
        # to isolate the architecture contribution from the CL contribution.
        return StandardTransformerIDS(input_dim=n_features, d_model=96,
                                      n_heads=4, n_layers=3, d_ff=384,
                                      n_classes=n_classes, max_seq_len=seq_len,
                                      dropout=0.1)
    if backbone == "rdt":
        # Robust Denoising Transformer: feature-denoising + spectral norm
        return RobustDenoiseTransformerIDS(
            input_dim=n_features, d_model=96, n_heads=4, n_layers=3,
            d_ff=384, n_classes=n_classes, max_seq_len=seq_len, dropout=0.1)
    if backbone == "lstm":
        return LSTMModel(input_dim=n_features, n_classes=n_classes,
                         hidden_dim=128, n_layers=2, dropout=0.1)
    if backbone == "dnn":
        return DNNModel(input_dim=n_features, n_classes=n_classes,
                        hidden_dim=256, dropout=0.1)
    raise ValueError(f"Unknown backbone: {backbone}")


def _train_surrogate(X_seq, y, n_classes, n_features, device, seed,
                     epochs=5, batch=512):
    """Train the surrogate the transfer attack crafts on.

    Stands in for an attacker who trains their own model on public data:
    a different architecture from any target backbone, its own seed, and
    standard training with no adversarial component. It never sees the
    target's weights or gradients, which is what keeps the transfer
    attack a zero-query black-box attack.
    """
    from models.baselines import DNNModel
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    sur = DNNModel(input_dim=n_features, n_classes=n_classes,
                   hidden_dim=256, dropout=0.1).to(device)
    opt = torch.optim.Adam(sur.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(torch.as_tensor(X_seq).float(),
                                        torch.as_tensor(y).long())
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     generator=g)
    sur.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(sur(xb)["logits"], yb)
            opt.zero_grad(); loss.backward(); opt.step()
    sur.eval()
    logger.info(f"  surrogate trained ({epochs} epochs) for transfer attack")
    return sur


def _uses_cl(backbone: str) -> bool:
    """Which backbones receive the continual-learning machinery
    (EWC + KD + replay). The AAM-TRANS family always does; the
    transformer_cl isolation baseline does too (same vanilla
    architecture but WITH CL), so we can tell apart the architecture
    contribution from the CL-machinery contribution."""
    return (backbone.startswith("aam_trans")
            or backbone == "transformer_cl"
            or backbone == "rdt")


# Component combinations for the CL+AT ablation. Each maps to the five
# switches of ContinualAdversarialTrainer:
#   ewc / kd / replay (standard CL) and rrep / rfish (Paper-3 novelty:
#   robust replay + adversarial Fisher).
VARIANTS = {
    "noCL":      dict(ewc=False, kd=False, replay=False, rrep=False, rfish=False),
    "ewc":       dict(ewc=True,  kd=False, replay=False, rrep=False, rfish=False),
    "kd":        dict(ewc=False, kd=True,  replay=False, rrep=False, rfish=False),
    "replay":    dict(ewc=False, kd=False, replay=True,  rrep=False, rfish=False),
    # ewckd = EWC+KD only (no replay): matches the effective config of the
    # existing 10K transformer results, for a fair non-transformer comparison.
    "ewckd":     dict(ewc=True,  kd=True,  replay=False, rrep=False, rfish=False),
    "cat":       dict(ewc=True,  kd=True,  replay=True,  rrep=False, rfish=False),
    "robustcat": dict(ewc=True,  kd=True,  replay=True,  rrep=True,  rfish=True),
    # BiC (Wu et al. 2019): rehearsal + distillation + a post-task linear
    # bias-correction stage (handled separately via fit_bias_correction).
    "bic":       dict(ewc=False, kd=True,  replay=True,  rrep=False, rfish=False),
}


def resolve_variant(name: str, backbone: str) -> dict:
    """Return the component switches for a variant. ``auto`` derives them
    from the backbone (legacy behaviour): CL backbones get standard CAT,
    everything else gets the no-CL sequential-AT baseline."""
    if name == "auto":
        return dict(VARIANTS["cat"]) if _uses_cl(backbone) \
            else dict(VARIANTS["noCL"])
    return dict(VARIANTS[name])


def _loader(X, y, batch=512, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def clean_accuracy(model, X, y, device, batch=1024):
    """Top-1 accuracy of the current model on (X, y)."""
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
        yb = torch.tensor(y[i:i + batch], dtype=torch.long, device=device)
        logits = model(xb)["logits"]
        correct += (logits.argmax(1) == yb).sum().item()
        total   += len(yb)
    return correct / max(total, 1)


@torch.no_grad()
def clean_acc_mcc(model, X, y, device, batch=1024):
    """Accuracy and MCC on (X, y). Used for the selection split, where a
    hyperparameter may legitimately be chosen without touching the
    held-out partition that results are reported on."""
    from sklearn.metrics import matthews_corrcoef
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
        preds.append(model(xb)["logits"].argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return float((p == y).mean()), float(matthews_corrcoef(y, p))


MASTER_HEADER = ["dataset", "seed", "backbone", "variant", "n_tasks",
                 # continual-learning metrics
                 "AA", "BWT", "remembering_rate", "CFI",
                 # robust classification (benign-vs-attack family space)
                 "clean_acc", "clean_mcc", "clean_f1", "clean_bal_acc",
                 # selection split (empty unless --sel-size is given)
                 "sel_acc", "sel_mcc",
                 "fgsm_acc", "fgsm_mcc", "fgsm_asr",
                 "pgd_acc", "pgd_mcc", "pgd_asr",
                 "square_acc", "square_mcc",
                 "transfer_acc", "transfer_mcc",
                 "epochs", "epsilon", "beta", "ewc_lambda", "status"]


def _load_completed(master_path):
    """Configs already finished successfully, so a resumed run skips them.

    Rows whose status is not ``ok`` are deliberately NOT counted: a
    config that crashed should be retried, not silently kept as a hole.
    """
    if not os.path.exists(master_path):
        return set()
    done = set()
    with open(master_path, "r", newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames and rd.fieldnames != MASTER_HEADER:
            missing = set(MASTER_HEADER) - set(rd.fieldnames or [])
            raise SystemExit(
                f"\nCannot resume into {master_path}: it was written with a "
                f"different set of columns.\nMissing here: "
                f"{sorted(missing) or 'none'}\n"
                f"Appending would corrupt the file. Use a fresh --run-dir "
                f"for runs on the current schema.\n")
        for r in rd:
            if r.get("status") == "ok" and (r.get("AA") or "").strip():
                var = r.get("variant", "auto") or "auto"
                done.add((r["backbone"], var, int(r["seed"]), r["dataset"]))
    return done


# ──────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    if args.run_dir:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) \
                  else os.path.join(here, args.run_dir)
        resume = True
    else:
        run_dir = os.path.join(here, "runs", "exp1",
                               time.strftime("%Y%m%d_%H%M%S"))
        resume = False
    os.makedirs(run_dir, exist_ok=True)
    master_path = os.path.join(run_dir, "paper3_exp1_master.csv")
    completed = _load_completed(master_path) if resume else set()

    logger.info(f"Paper3 EXP1 (attack-incremental CL+AT) | device={device} | "
                f"run_dir={run_dir} | resume={resume} | done={len(completed)}")

    write_header = not (resume and os.path.exists(master_path))
    fh = open(master_path, "a" if resume else "w", newline="", encoding="utf-8")
    w  = csv.writer(fh)
    if write_header:
        w.writerow(MASTER_HEADER)
        fh.flush()

    datasets  = args.datasets or DATASET_ORDER
    seeds     = args.seeds
    backbones = args.backbones
    epochs    = args.epochs if args.epochs else (3 if args.quick else 20)

    # Optional balanced-subset data source (class-balanced subsets)
    variant = args.variant
    if args.balanced:
        bdir = (args.balanced_dir if os.path.isabs(args.balanced_dir)
                else os.path.join(here, args.balanced_dir))
        dataset_paths = {k: os.path.join(bdir, f"balanced_{k}.csv")
                         for k in DATASET_PATHS}
        frac_fn = lambda ds: 1.0
        logger.info(f"Using BALANCED subsets from {bdir} (sample_frac=1.0)")
    else:
        dataset_paths = DATASET_PATHS
        frac_fn = lambda ds: DATASET_SAMPLE_FRACTIONS.get(ds, 0.10)
    logger.info(f"Variant = {variant}")

    for backbone in backbones:
        for seed in seeds:
            for ds_name in datasets:
                if (backbone, variant, int(seed), ds_name) in completed:
                    logger.info(f"SKIP {backbone}/{variant} s{seed} "
                                f"{ds_name} (done)")
                    continue
                t0 = time.time()
                tag = f"{backbone}_{variant}_s{seed}_{ds_name}"
                logger.info(f"\n{'='*60}\n{tag}\n{'='*60}")
                model = None
                try:
                    torch.manual_seed(seed); np.random.seed(seed)
                    frac = frac_fn(ds_name)
                    data = load_dataset(
                        csv_path=dataset_paths[ds_name], dataset_name=ds_name,
                        label_column=LABEL_COLUMN, mode="multiclass",
                        sample_frac=frac, test_size=TEST_SIZE,
                        seq_len=SEQ_LEN, seq_noise=SEQ_NOISE_STD,
                        random_seed=seed,
                        benign_label=BENIGN_LABELS.get(ds_name),
                        # balanced subsets are pre-sized; use the whole file
                        # and skip the per-dataset config sub-sampling.
                        force_frac=args.balanced,
                        balance_train_only=args.split_safe,
                        sel_size=args.sel_size)

                    tasks, family_names = split_into_attack_tasks(data, ds_name)
                    T = len(tasks)
                    if T < 2:
                        logger.warning(f"  {ds_name}: only {T} task(s) — "
                                       f"skipping (need >=2 for CL).")
                        continue
                    n_classes = len(family_names)
                    logger.info(f"  Families ({n_classes}): {family_names}")
                    logger.info(f"  Tasks (T={T}): "
                                f"{[t['family'] for t in tasks]}")

                    # cuDNN's fused RNN/LSTM kernels cannot run backward in
                    # eval() mode, which adversarial training (TRADES inner
                    # loop, EWC Fisher, FGSM eval) requires. Disable cuDNN for
                    # recurrent backbones; Transformer/DNN keep it on.
                    torch.backends.cudnn.enabled = backbone not in ("lstm", "rnn")

                    model = build_model(backbone, data["n_features"],
                                        n_classes, SEQ_LEN).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                                 weight_decay=1e-5)
                    sw = resolve_variant(variant, backbone)
                    cat = ContinualAdversarialTrainer(
                        epsilon=args.epsilon, alpha=args.alpha,
                        pgd_steps=args.pgd_steps, beta_trades=args.beta,
                        ewc_lambda=args.ewc_lambda, kd_temp=args.kd_temp,
                        use_ewc=sw["ewc"], use_kd=sw["kd"],
                        use_replay=sw["replay"],
                        robust_replay=sw["rrep"], robust_fisher=sw["rfish"],
                        random_seed=seed)

                    # T x T clean-accuracy task matrix
                    task_mat = np.zeros((T, T), dtype=float)

                    # BiC: post-task bias-correction stack + class bookkeeping.
                    # Enabled by the dedicated 'bic' variant OR by --bias-correct
                    # on top of any variant (e.g. cat -> CAT+bias-correction).
                    use_bic   = (variant == "bic") or args.bias_correct
                    bias_corr = BiasCorrection() if use_bic else None
                    seen_cls  = set()

                    for t_id, task in enumerate(tasks):
                        logger.info(f"  -- Task {t_id+1}/{T}: "
                                    f"family={task['family']} "
                                    f"(train n={len(task['y_train'])})")

                        Xtr, ytr = task["X_train"], task["y_train"]
                        Xbic, ybic = None, None
                        # BiC needs a balanced validation split to fit the
                        # bias-correction stage. It must be DISJOINT from the
                        # evaluation set (task["X_val"]), so we hold out 10% of
                        # this task's TRAINING data (Wu et al.'s protocol: the
                        # bias stage is fit on reserved samples, not on test).
                        if use_bic and t_id > 0:
                            rng = np.random.default_rng(1000 + seed * 10 + t_id)
                            n = len(ytr)
                            perm = rng.permutation(n)
                            k = max(1, int(round(0.10 * n)))
                            hold, keep = perm[:k], perm[k:]
                            Xbic = np.asarray(Xtr)[hold]; ybic = np.asarray(ytr)[hold]
                            Xtr = np.asarray(Xtr)[keep]; ytr = np.asarray(ytr)[keep]

                        loader = _loader(Xtr, ytr)
                        # Class-balanced focal loss for THIS task's label mix
                        cat.set_class_loss(ytr, n_classes, device)
                        cat.train_task(model, loader, optimizer, device,
                                       epochs=epochs)

                        # BiC stage: fit (alpha,beta) for the classes this task
                        # introduced, on a balanced old(replay)/new(holdout)
                        # split with the backbone frozen. Old samples come from
                        # the replay buffer (tasks 0..t-1); new samples from the
                        # held-out training slice carved above.
                        if use_bic:
                            new_cls = sorted(set(np.unique(task["y_train"]).tolist())
                                             - seen_cls)
                            # The base task (t_id==0) gets no correction stage,
                            # as in Wu et al.; correction starts once there are
                            # old classes to balance against.
                            if t_id > 0 and cat.cl.memory.size > 0:
                                n_old = min(cat.cl.memory.size, 2000)
                                x_old, y_old = cat.cl.memory.sample(n_old)
                                a_bic, b_bic = fit_bias_correction(
                                    model, bias_corr, new_cls,
                                    x_old, y_old, Xbic, ybic, device,
                                    robust=args.robust_bc,
                                    adv_eps=args.epsilon, adv_alpha=args.alpha,
                                    adv_steps=args.pgd_steps,
                                    adv_w=args.bc_adv_w)
                                logger.info(f"     BiC stage: new_cls={new_cls} "
                                            f"alpha={a_bic:.3f} beta={b_bic:+.3f}")
                            seen_cls.update(np.unique(task["y_train"]).tolist())

                        eval_model = (BiCWrapped(model, bias_corr)
                                      if use_bic else model)

                        # eval clean acc on all tasks seen so far
                        for j in range(t_id + 1):
                            acc = clean_accuracy(eval_model, tasks[j]["X_val"],
                                                 tasks[j]["y_val"], device)
                            task_mat[j, t_id] = acc
                        # consolidate CL state for next task: fill replay
                        # buffer, freeze KD teacher, estimate EWC Fisher
                        # (adversarial when robust_fisher is on).
                        cat.consolidate_task(model, task["X_train"],
                                             task["y_train"], loader, device)

                    # The deployed BiC classifier is the corrected one; use it
                    # for the robustness evaluation and checkpoint below.
                    eval_model = (BiCWrapped(model, bias_corr)
                                  if use_bic else model)

                    AA  = compute_average_accuracy(task_mat)
                    BWT = compute_backward_transfer(task_mat)
                    REM = compute_remembering_rate(task_mat)
                    CFI = compute_catastrophic_forgetting_index(task_mat)
                    logger.info(f"  CL metrics: AA={AA:.4f}  BWT={BWT:+.4f}  "
                                f"REM={REM:.4f}  CFI={CFI:.4f}")

                    # Persist the T x T task-accuracy matrix for figures/audit
                    tm_path = os.path.join(run_dir, f"taskmat_{tag}.csv")
                    np.savetxt(tm_path, task_mat, delimiter=",", fmt="%.6f",
                               header=",".join(t["family"] for t in tasks),
                               comments="")

                    # ── Robust evaluation on the full validation set ───────
                    # The model has a MULTICLASS head in family-label space,
                    # so we evaluate the attack suite in that same space
                    # (mapping the fine y_val labels to family indices).
                    from task_splitter import build_family_mapping
                    c2f, _ = build_family_mapping(data["class_names"], ds_name)
                    yval_fam = c2f[data["y_val"]].astype(np.int64)

                    # Selection split: reported separately so a
                    # hyperparameter can be chosen on it without the
                    # held-out numbers ever informing the choice.
                    sel_acc = sel_mcc = ""
                    if data.get("X_sel_seq") is not None:
                        ysel_fam = c2f[data["y_sel"]].astype(np.int64)
                        sel_acc, sel_mcc = clean_acc_mcc(
                            eval_model, data["X_sel_seq"], ysel_fam, device)
                        logger.info(f"  selection split: acc={sel_acc:.4f} "
                                    f"mcc={sel_mcc:.4f}")
                    fcc = FeatureConstraintClassifier()
                    # Un-normalised features: the projection and the
                    # validity check both work in raw units, so the bounds
                    # must be learned there too.
                    fcc.fit(data["X_train_unscaled"], data["feature_names"])
                    full_val = _loader(data["X_val_seq"], yval_fam,
                                       shuffle=False)
                    attacker = AdversarialAttacker(model=eval_model, device=device,
                                                   fcc=fcc, scaler=data["scaler"])
                    # The transfer attack needs a surrogate the attacker
                    # trained themselves. Without one the attacker falls back
                    # to an UNTRAINED network, whose gradients are noise, so
                    # the "transfer" column would just re-measure the Gaussian
                    # control. Train an independent one: different
                    # architecture, different seed, no adversarial training.
                    try:
                        ytr_fam = c2f[data["y_train"]].astype(np.int64)
                        surrogate = _train_surrogate(
                            data["X_train_seq"], ytr_fam, n_classes,
                            data["n_features"], device, seed + 1000)
                        attacker.set_surrogate(surrogate)
                    except Exception as se:
                        logger.warning(f"  surrogate training failed: {se}")
                    from config import AttackConfig
                    try:
                        rob = evaluate_robustness(eval_model, full_val, attacker,
                                                  device, AttackConfig(),
                                                  fcc=fcc, scaler=data["scaler"],
                                                  class_names=family_names)
                    except Exception as ee:
                        logger.warning(f"  robust eval failed: {ee}")
                        rob = {}

                    def g(atk, m):
                        return rob.get(atk, {}).get(m, "") if isinstance(rob, dict) else ""

                    # Persist the FULL robust metric table (all 17 metrics x
                    # all attacks) for this config, mirroring Paper 2.
                    if isinstance(rob, dict) and rob:
                        rb_path = os.path.join(run_dir, f"robust_{tag}.csv")
                        metric_keys = ["accuracy", "precision", "recall", "f1",
                                       "f1_macro", "mcc", "kappa",
                                       "balanced_accuracy", "gmean",
                                       "roc_auc", "pr_auc", "fpr", "far",
                                       "fnr", "dr", "asr", "validity_rate"]
                        with open(rb_path, "w", newline="",
                                  encoding="utf-8") as rfh:
                            rw = csv.writer(rfh)
                            rw.writerow(["attack"] + metric_keys)
                            for atk, md in rob.items():
                                if isinstance(md, dict):
                                    rw.writerow([atk] +
                                                [md.get(k, "") for k in metric_keys])

                    # Save the final trained model so this config can be
                    # re-evaluated later WITHOUT retraining.
                    try:
                        torch.save(model.state_dict(),
                                   os.path.join(run_dir, f"model_{tag}.pth"))
                    except Exception as se:
                        logger.warning(f"  checkpoint save failed: {se}")

                    w.writerow([
                        ds_name, seed, backbone, variant, T,
                        f"{AA:.4f}", f"{BWT:.4f}", f"{REM:.4f}", f"{CFI:.4f}",
                        g("Clean", "accuracy"), g("Clean", "mcc"),
                        g("Clean", "f1"), g("Clean", "balanced_accuracy"),
                        sel_acc, sel_mcc,
                        g("FGSM_eps0.1", "accuracy"), g("FGSM_eps0.1", "mcc"),
                        g("FGSM_eps0.1", "asr"),
                        g("PGD_eps0.1", "accuracy"), g("PGD_eps0.1", "mcc"),
                        g("PGD_eps0.1", "asr"),
                        g("Square_eps0.1", "accuracy"), g("Square_eps0.1", "mcc"),
                        g("Transfer_eps0.1", "accuracy"),
                        g("Transfer_eps0.1", "mcc"),
                        epochs, args.epsilon, args.beta, args.ewc_lambda,
                        "ok",
                    ])
                    fh.flush()
                    logger.info(f"  {tag} done in {(time.time()-t0)/60:.1f} min")

                except Exception as e:
                    logger.exception(f"FAILED {tag}: {e}")
                    # 30-column row to match the header (blanks for metrics)
                    w.writerow([ds_name, seed, backbone, variant, ""] +
                               [""] * 20 +
                               [epochs, args.epsilon, args.beta,
                                args.ewc_lambda,
                                f"error: {type(e).__name__}"])
                    fh.flush()
                finally:
                    if model is not None:
                        del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache(); torch.cuda.synchronize()

    fh.close()
    logger.info(f"\nEXP1 complete. Master: {master_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="Resume into existing dir.")
    ap.add_argument("--balanced", action="store_true",
                    help="Use class-balanced subsets from --balanced-dir "
                         "(build with make_balanced_subset.py first).")
    ap.add_argument("--sel-size", type=float, default=0.0,
                    help="Fraction of TRAIN carved off as a selection "
                         "split (e.g. 0.15). Reported as sel_acc/sel_mcc, "
                         "so hyperparameters can be chosen without "
                         "consulting the held-out partition.")
    ap.add_argument("--split-safe", action="store_true",
                    help="Balance the train split only, after the train/val "
                         "split. Use with a --balanced-dir built by "
                         "make_balanced_subset.py --no-oversample.")
    ap.add_argument("--balanced-dir", default="balanced_data",
                    help="Folder with balanced_<DS>.csv (e.g. "
                         "balanced_data_30k). Used only with --balanced.")
    ap.add_argument("--variant", default="auto", choices=list(VARIANTS) + ["auto"],
                    help="CL+AT component combo: noCL | ewc | kd | replay | "
                         "cat | robustcat. 'auto' = legacy per-backbone.")
    ap.add_argument("--quick", action="store_true", help="3 epochs smoke test.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epoch count (e.g. 10 for a fast <1h pass). "
                         "If unset: 3 with --quick, else 20.")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASET_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 13, 21, 42])
    ap.add_argument("--backbones", nargs="+",
                    default=["aam_trans", "aam_trans_no_gates", "transformer"],
                    choices=["aam_trans", "aam_trans_no_gates",
                             "aam_trans_no_anomaly", "transformer",
                             "transformer_cl", "rdt", "lstm", "dnn"])
    ap.add_argument("--epsilon",   type=float, default=0.10)
    ap.add_argument("--alpha",     type=float, default=0.02)
    ap.add_argument("--pgd-steps", type=int,   default=7)
    ap.add_argument("--beta",      type=float, default=6.0)
    ap.add_argument("--ewc-lambda", type=float, default=1000.0,
                    help="EWC penalty strength (sensitivity sweep, pt 10).")
    ap.add_argument("--kd-temp",    type=float, default=2.0,
                    help="KD distillation temperature (sensitivity sweep).")
    ap.add_argument("--bias-correct", action="store_true",
                    help="Add the BiC bias-correction stage on top of the "
                         "chosen variant (e.g. --variant cat --bias-correct "
                         "= CAT+BC). Requires replay (needs old exemplars).")
    ap.add_argument("--robust-bc", action="store_true",
                    help="Fit the bias-correction stage on clean + PGD-"
                         "adversarial calibration samples (Robust Bias "
                         "Correction). Only effective with --bias-correct.")
    ap.add_argument("--bc-adv-w", type=float, default=1.0,
                    help="Weight on the adversarial term in the robust bias-"
                         "correction fit (>1 emphasises worst-case logits).")
    main(ap.parse_args())
