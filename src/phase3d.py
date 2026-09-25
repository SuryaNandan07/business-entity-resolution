"""Focused Phase 3D experiments using only Phase 3A train/validation pairs.

The frozen holdout is deliberately not read by this module. Derived features
are generic interactions of existing name/address evidence, so no new country
rules or external data are introduced.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from src.modeling import load_pairs, tune_threshold, threshold_metrics, training_columns
from src.evaluate import macro_f05

FOLDER = Path('output/pair_features')
OUT = Path('output/phase3d')
MODEL_DIR = Path('output/baseline_models')


def derive(x, names):
    pos = {n: i for i, n in enumerate(names)}
    def col(n): return x[:, pos[n]]
    # Small, generic, evidence-backed interactions.
    extra = np.column_stack([
        col('address_rare_token_dice') * (col('numeric_token_jaccard') > 0),
        col('address_token_jaccard') * col('address_length_ratio'),
        col('name_rare_token_dice') * col('name_trigram_dice'),
        ((col('shared_numeric_token_count') == 0) & (col('numeric_token_jaccard') == 0) & (col('address_token_jaccard') > .5)).astype('float32'),
        ((col('address_rare_token_dice') > .5) & (col('name_trigram_dice') < .2)).astype('float32'),
    ]).astype('float32')
    return np.column_stack([x, extra]).astype('float32'), [
        'address_rare_numeric_interaction', 'address_containment_length_interaction',
        'name_rare_trigram_interaction', 'numeric_conflict', 'strong_address_weak_name']


def main():
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads((FOLDER / 'schema.json').read_text())
    prior = json.loads((FOLDER / 'report.json').read_text())
    names = schema['feature_columns']
    split = pd.read_csv(FOLDER / 's1_split.csv')
    train_ids = set(split.loc[split.split == 'train', 'source1_entity_id'])
    val_ids = set(split.loc[split.split == 'validation', 'source1_entity_id'])
    x, y, _ = load_pairs(FOLDER / 'train_features.csv.gz', names, prior['balances']['train']['candidate_pairs'], train_ids)
    v, vy, metadata = load_pairs(FOLDER / 'validation_features.csv.gz', names, prior['balances']['validation']['candidate_pairs'], val_ids, True)
    selected = training_columns(x, names)
    base_names = [names[i] for i in selected]
    x, v = x[:, selected], v[:, selected]
    ordered = sorted(val_ids); lookup = {s: i for i, s in enumerate(ordered)}
    entity_indices = np.array([lookup[s] for s, _ in metadata])
    # Full truth counts are recorded in Phase 3A; use the saved labels plus the
    # known validation total to preserve the honest missing-link denominator.
    truth_counts = np.full(len(ordered), 0, dtype=np.int32)
    # Read the compact ground truth directly; this is training data, not holdout.
    from src.candidates import load_truth
    truth = load_truth(Path('dataset/train/train_ground_truth.tsv'), val_ids)
    truth_counts = np.array([len(truth[s]) for s in ordered])
    experiments = {'old_baseline': (x, v, base_names, False)}
    dx, extra_names = derive(x, base_names); dv, _ = derive(v, base_names)
    experiments['interaction_features'] = (dx, dv, base_names + extra_names, True)
    report = {'validation_s1': len(val_ids), 'train_pairs': len(y), 'validation_pairs': len(vy), 'experiments': {},
              'features_added': extra_names, 'selection_rule': 'highest validation macro F0.5; threshold tuned only on validation'}
    best = None
    for exp, (tx, vx, exp_names, derived) in experiments.items():
        start = time.perf_counter()
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=50,
            learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=42)
        model.fit(tx, y); probabilities = model.predict_proba(vx)[:, 1]
        optimum, curve = tune_threshold(probabilities, vy, entity_indices, truth_counts)
        result = {'features': exp_names, 'derived': derived, 'best': optimum,
                  'roc_auc': float(roc_auc_score(vy, probabilities)),
                  'average_precision': float(average_precision_score(vy, probabilities)),
                  'runtime_seconds': time.perf_counter() - start}
        report['experiments'][exp] = result
        pd.DataFrame(curve).to_csv(OUT / f'{exp}_thresholds.csv', index=False)
        if best is None or optimum['macro_f05'] > best[1]: best = (exp, optimum['macro_f05'], model, exp_names)
    # Failure counts on the fixed validation candidate set at the selected threshold.
    selected_exp, _, selected_model, selected_names = best
    selected_x = experiments[selected_exp][1]
    probs = selected_model.predict_proba(selected_x)[:, 1]
    threshold = report['experiments'][selected_exp]['best']['threshold']
    report['failure_counts'] = {
        'classifier_false_negatives': int(((vy == 1) & (probs < threshold)).sum()),
        'classifier_false_positives': int(((vy == 0) & (probs >= threshold)).sum()),
        'candidate_generation_misses': int(truth_counts.sum() - vy.sum()),
        'script_related_candidates': 'Not identifiable from Phase 3A numeric-only feature files; requires raw paired text inspection.',
        'address_related_false_negatives': int(((vy == 1) & (probs < threshold) & (v[:, names.index('address_rare_token_dice')] < .2)).sum()),
    }
    # Save the selected validation-only model for explicit review; no holdout is
    # consumed here and this artifact is not the Phase 3B frozen model.
    import joblib
    joblib.dump({'model': selected_model, 'feature_names': selected_names, 'threshold': threshold}, OUT / 'selected_validation_model.joblib', compress=3)
    report['selected_experiment'] = selected_exp
    report['runtime_seconds'] = time.perf_counter() - started
    from src.ranking import peak_memory_mib
    report['peak_memory_mib'] = peak_memory_mib()
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT / 'summary.md').write_text('# Phase 3D validation-only experiment\n\n' + json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
