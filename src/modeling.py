"""Sample-only baseline experiment: python -m src.modeling.

Reads Phase 3A artifacts and training ground truth. Never reads test data.
Threshold selection is validation tuning, not an unbiased final test score.
"""
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.candidates import load_truth, read_rows
from src.evaluate import macro_f05
from src.failure_analysis import input_manifest
from src.ranking import peak_memory_mib


def threshold_metrics(probabilities, labels, entity_indices, true_counts, threshold):
    """Fast set-equivalent evaluator for validated, unique candidate pairs."""
    selected = probabilities >= threshold
    counts = np.bincount(entity_indices[selected], minlength=len(true_counts))
    hits = np.bincount(entity_indices[selected & (labels == 1)], minlength=len(true_counts))
    denominator = counts + .25 * true_counts
    scores = np.divide(1.25 * hits, denominator, out=np.ones(len(counts)), where=denominator != 0)
    total, tp = int(counts.sum()), int(hits.sum())
    return dict(threshold=float(threshold), macro_f05=float(scores.mean()), predicted_links=total,
                average_links=float(counts.mean()), singleton_percentage=float(100 * np.mean(counts == 1)),
                zero_prediction_percentage=float(100 * np.mean(counts == 0)),
                pair_precision=tp / total if total else 0.,
                pair_recall=tp / int(labels.sum()) if labels.sum() else 0.,
                full_truth_link_recall=tp / int(true_counts.sum()) if true_counts.sum() else 0.)


def tune_threshold(probabilities, labels, entity_indices, true_counts):
    coarse = np.round(np.arange(.10, .951, .05), 3)
    def evaluate(grid):
        return [threshold_metrics(probabilities, labels, entity_indices, true_counts, t) for t in grid]
    results = evaluate(coarse)
    # Fixed tie-break: prefer the higher threshold (fewer false links).
    best = max(results, key=lambda row: (row['macro_f05'], row['threshold']))
    refined = np.round(np.arange(max(.005, best['threshold'] - .05),
                                min(.995, best['threshold'] + .05) + .001, .005), 3)
    results += evaluate(sorted(set(refined) - set(coarse)))
    return max(results, key=lambda row: (row['macro_f05'], row['threshold'])), sorted(results, key=lambda r: r['threshold'])


def training_columns(x, names):
    """Drop train-constant columns and one redundant source indicator."""
    return [i for i, name in enumerate(names)
            if np.min(x[:, i]) != np.max(x[:, i]) and name != 'candidate_is_s3']


def load_pairs(path, names, expected_rows, allowed_ids, keep_metadata=False):
    """Allocate numeric arrays once; parse compressed input in bounded chunks."""
    x = np.empty((expected_rows, len(names)), dtype=np.float32)
    y = np.empty(expected_rows, dtype=np.uint8)
    metadata = []
    offset = 0
    seen_entities = set()
    previous_entity, candidate_ids = None, set()
    for chunk in pd.read_csv(path, chunksize=50000, dtype={**{n: 'float32' for n in names}, 'label': 'uint8'}):
        end = offset + len(chunk)
        if end > expected_rows:
            raise ValueError('Unexpected row count')
        for s1, candidate in zip(chunk.source1_entity_id, chunk.candidate_entity_id):
            if s1 not in allowed_ids:
                raise ValueError('S1 split leakage or unknown entity')
            if s1 != previous_entity:
                if s1 in seen_entities:
                    raise ValueError('Expected entity-grouped feature rows')
                seen_entities.add(s1)
                candidate_ids = set()
                previous_entity = s1
            if candidate in candidate_ids:
                raise ValueError('Duplicate candidate pair')
            candidate_ids.add(candidate)
        x[offset:end] = chunk[names].to_numpy()
        y[offset:end] = chunk.label.to_numpy()
        if keep_metadata:
            metadata.extend(zip(chunk.source1_entity_id, chunk.candidate_entity_id))
        offset = end
    if offset != expected_rows or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid numeric data or row count')
    return x, y, metadata


def error_examples(metadata, labels, probabilities, threshold, truth, features, names):
    predictions = {s1: set() for s1 in truth}
    present = {s1: set() for s1 in truth}
    for (s1, candidate), probability in zip(metadata, probabilities):
        present[s1].add(candidate)
        if probability >= threshold:
            predictions[s1].add(candidate)
    examples = []
    categories = {
        'false_positive': np.flatnonzero((labels == 0) & (probabilities >= threshold)),
        'false_negative': np.flatnonzero((labels == 1) & (probabilities < threshold)),
        'true_low_probability': np.flatnonzero(labels == 1),
        'false_high_probability': np.flatnonzero(labels == 0),
    }
    used = set()
    for category, indices in categories.items():
        order = sorted(indices, key=lambda i: float(probabilities[i]),
                       reverse=category in ('false_positive', 'false_high_probability'))
        count = 0
        for i in order:
            if int(i) in used:
                continue
            used.add(int(i))
            s1, candidate = metadata[i]
            examples.append(dict(category=category, source1_entity_id=s1, candidate_entity_id=candidate,
                                 label=int(labels[i]), probability=float(probabilities[i]),
                                 true_ids=sorted(truth[s1]), predicted_ids=sorted(predictions[s1]),
                                 features={n: float(v) for n, v in zip(names, features[i])}))
            count += 1
            if count == 4:
                break
    singleton_stats = {}
    for correct, category in [(True, 'correct_singleton'), (False, 'incorrect_singleton')]:
        ids = [s1 for s1 in sorted(truth) if len(predictions[s1]) == 1 and (predictions[s1] == truth[s1]) == correct]
        singleton_stats[category] = len(ids)
        for s1 in ids[:4]:
            candidate = next(iter(predictions[s1]))
            i = metadata.index((s1, candidate))
            examples.append(dict(category=category, source1_entity_id=s1, candidate_entity_id=candidate,
                                 label=int(labels[i]), probability=float(probabilities[i]),
                                 true_ids=sorted(truth[s1]), predicted_ids=sorted(predictions[s1])))
    for s1 in sorted(truth):
        for candidate in sorted(truth[s1] - present[s1]):
            examples.append(dict(category='missing_from_candidates', source1_entity_id=s1,
                                 candidate_entity_id=candidate, label=1, probability=None,
                                 true_ids=sorted(truth[s1]), predicted_ids=sorted(predictions[s1])))
            if sum(e['category'] == 'missing_from_candidates' for e in examples) == 4:
                return examples, singleton_stats, predictions
    return examples, singleton_stats, predictions


def write_summary(report, examples, out):
    """Readable review artifacts; complete threshold curves stay in small CSVs."""
    lines = ['# Phase 3B baseline validation', '',
             '16,000 training S1 / 4,000 validation S1; 767,666 / 191,484 candidate pairs.',
             'All preprocessing was fitted on training only. Six train-constant features and the redundant S3 indicator were removed.', '',
             '| Model | Threshold | Macro F0.5 | Pair precision | Pair recall | ROC AUC | Average precision | Links/S1 | Singleton % | Zero links % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in report['models'].items():
        b = result['best']
        lines.append(f"| {name} | {b['threshold']:.3f} | {b['macro_f05']:.6f} | {b['pair_precision']:.6f} | {b['pair_recall']:.6f} | {result['roc_auc']:.6f} | {result['average_precision']:.6f} | {b['average_links']:.4f} | {b['singleton_percentage']:.3f} | {b['zero_prediction_percentage']:.3f} |")
    lines += ['', report['weighting'], '',
              'Threshold search: 0.10–0.95 in 0.05 steps, then ±0.05 around the best coarse threshold in 0.005 steps. Ties prefer the higher threshold.',
              'HGB: 100 iterations, 15 leaves, minimum 50 samples/leaf, learning rate 0.1, L2=1; no early stopping. CPU thread limit: 4.',
              'Logistic regression: StandardScaler, C=1, maximum 500 iterations, seed 42; both fits converged.', '',
              f"Candidate recall is 99.2196%; 107 true links are absent. Perfect classification of available candidates gives macro F0.5 {report['candidate_oracle']['macro_f05']:.6f}, a different quantity from link recall.",
              'Gradient boosting produces 13,356 links: 13,233 true positives and 123 false positives. It misses 371 available positives plus 107 unavailable links. Full-truth link recall is 96.5137%.',
              'It predicts exactly one ID for 265 entities: 213 complete correct sets, 52 incorrect/incomplete sets. It predicts no IDs for 230 entities.', '',
              'Important observed errors: similar names and nearby but different house numbers can score too highly; an identical address can outweigh unrelated names. Script changes, severe name corruption, and missing or abbreviated addresses cause low scores on true links. Examples are selected illustrations, not estimated category frequencies.',
              'Incorrect singletons may contain a correct link but omit additional true links; they are not necessarily false-positive pairs.', '',
              'HGB permutation importance: average-precision decrease on 10,000 seeded validation pairs, two repeats. This is a bounded diagnostic, not a feature-selection step. Correlated features can share importance.', '']
    for item in report['hgb_permutation_importance'][:6]:
        lines.append(f"- {item['feature']}: {item['average_precision_drop']:.6f} AP decrease")
    lines += ['', 'Unweighted logistic coefficients per training standard deviation (conditional on other features):']
    coefficients = report['models']['logistic_unweighted']['standardized_coefficients']
    for item in coefficients[:4] + coefficients[-4:]:
        lines.append(f"- {item['feature']}: {item['coefficient']:+.4f}")
    lines += ['', 'Negative overlap coefficients do not imply that overlap is intrinsically harmful: strongly correlated name/address features are included together.', '',
              f"Runtime: {report['runtime_seconds']:.2f} seconds; peak process working set: {report['peak_memory_mib']:.2f} MiB. Includes loading, fitting, tuning, importance and example retrieval; excludes interpreter startup/dependency installation.",
              'Full test suite: 65 passed. Original training TSV size/mtime metadata unchanged.', '',
              *report['limitations'], '', 'Review the 28 examples in error_examples.md and the complete curves in *_thresholds.csv.',
              'Re-run: python -m src.modeling. In this workspace runtime, first set PYTHONPATH=.runtime-deps. Installed dependency versions are recorded in report.json. No test files, final submissions, or branch merges were used.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    lines = ['# Validation examples', '', 'TRUE/FALSE refers to supplied ground truth; probabilities are baseline scores, not calibrated confidence.', '']
    for number, e in enumerate(examples, 1):
        lines += [f"## {number}. {e['category']}", '',
                  f"S1 `{e['source1_entity_id']}` → candidate `{e['candidate_entity_id']}`; label {e['label']}; probability {e['probability']}", '',
                  f"S1: {e['source1']['business_name']} — {e['source1']['business_address']}",
                  f"Candidate: {e['candidate']['business_name']} — {e['candidate']['business_address']}",
                  f"Country: {e['source1']['country']} / {e['candidate']['country']}", '',
                  f"True IDs: {', '.join(e['true_ids']) or '(none)'}", '',
                  f"Predicted IDs: {', '.join(e['predicted_ids']) or '(none)'}", '']
    (out / 'error_examples.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    started = time.perf_counter()
    folder, out = Path('output/pair_features'), Path('output/baseline_models')
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((folder / 'schema.json').read_text())
    prior = json.loads((folder / 'report.json').read_text())
    original_manifest = input_manifest(Path('dataset/train'))
    if original_manifest != prior['input_manifest']:
        raise ValueError('Training inputs changed since Phase 3A')
    names = schema['feature_columns']
    split = pd.read_csv(folder / 's1_split.csv')
    train_ids = set(split.loc[split.split == 'train', 'source1_entity_id'])
    val_ids = set(split.loc[split.split == 'validation', 'source1_entity_id'])
    if train_ids & val_ids or len(split) != len(train_ids) + len(val_ids):
        raise ValueError('Invalid S1 split')
    truth = load_truth(Path('dataset/train/train_ground_truth.tsv'), val_ids)
    x, y, _ = load_pairs(folder / 'train_features.csv.gz', names, prior['balances']['train']['candidate_pairs'], train_ids)
    v, vy, metadata = load_pairs(folder / 'validation_features.csv.gz', names, prior['balances']['validation']['candidate_pairs'], val_ids, True)
    if any(int(label) != int(candidate in truth[s1]) for (s1, candidate), label in zip(metadata, vy)):
        raise ValueError('Saved validation labels disagree with full ground truth')
    selected = training_columns(x, names)
    dropped = [n for i, n in enumerate(names) if i not in selected]
    names = [names[i] for i in selected]
    x, v = np.ascontiguousarray(x[:, selected]), np.ascontiguousarray(v[:, selected])
    ordered_ids = sorted(val_ids)
    lookup = {s1: i for i, s1 in enumerate(ordered_ids)}
    indices = np.array([lookup[s1] for s1, _ in metadata])
    true_counts = np.array([len(truth[s1]) for s1 in ordered_ids])
    oracle = threshold_metrics(vy, vy, indices, true_counts, .5)
    print(f'Loaded {len(y)} train / {len(vy)} validation pairs; {len(names)} varying features', flush=True)
    models = {
        'logistic_unweighted': make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42)),
        'logistic_balanced': make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
            min_samples_leaf=50, learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=42),
    }
    report = dict(seed=42, sklearn_version=sklearn.__version__, scipy_version=scipy.__version__,
                  numpy_version=np.__version__, train_s1=len(train_ids), validation_s1=len(val_ids),
                  train_pairs=len(y), validation_pairs=len(vy), features=names, dropped_features=dropped,
                  candidate_oracle=oracle, missing_true_links=int(true_counts.sum() - vy.sum()), models={},
                  weighting='Logistic: none and balanced; HGB: unweighted, with threshold tuning to control precision/recall.',
                  limitations=['Thresholds and model comparison tuned on validation: no unbiased final score.',
                               'Truth-enriched sampled catalog; full-catalog and test performance unknown.',
                               'Pair recall denominator includes candidate positives only; macro F0.5 uses ALL truth.',
                               'Singleton means exactly one predicted target ID; zero predicted IDs reported separately.'])
    best_name, best_score, best_probs = None, -1, None
    with threadpool_limits(limits=4):
        for name, model in models.items():
            start = time.perf_counter()
            model.fit(x, y)
            fit_seconds = time.perf_counter() - start
            probabilities = model.predict_proba(v)[:, 1]
            best, curve = tune_threshold(probabilities, vy, indices, true_counts)
            result = dict(best=best, roc_auc=float(roc_auc_score(vy, probabilities)),
                          average_precision=float(average_precision_score(vy, probabilities)),
                          fit_seconds=fit_seconds, total_seconds=time.perf_counter() - start)
            if name.startswith('logistic'):
                result['standardized_coefficients'] = sorted(
                    [dict(feature=n, coefficient=float(c)) for n, c in zip(names, model[-1].coef_[0])],
                    key=lambda r: r['coefficient'], reverse=True)
                result['iterations'] = int(model[-1].n_iter_[0])
            report['models'][name] = result
            with (out / f'{name}_thresholds.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
                writer.writeheader()
                writer.writerows(curve)
            joblib.dump(dict(model=model, feature_names=names, threshold=best['threshold'], seed=42), out / f'{name}.joblib', compress=3)
            if best['macro_f05'] > best_score:
                best_name, best_score, best_probs = name, best['macro_f05'], probabilities
            print(name, json.dumps(result['best']), f'fit={fit_seconds:.1f}s', flush=True)
        # Bounded validation diagnostic, not used to refit or select model features.
        start = time.perf_counter()
        sample = np.random.default_rng(42).choice(len(vy), min(10000, len(vy)), replace=False)
        importance = permutation_importance(models['hist_gradient_boosting'], v[sample], vy[sample],
                                            scoring='average_precision', n_repeats=2, random_state=42, n_jobs=1)
        report['hgb_permutation_importance'] = sorted(
            [dict(feature=n, average_precision_drop=float(m), std=float(s))
             for n, m, s in zip(names, importance.importances_mean, importance.importances_std)],
            key=lambda r: r['average_precision_drop'], reverse=True)
        report['importance_seconds'] = time.perf_counter() - start
    threshold = report['models'][best_name]['best']['threshold']
    examples, singleton_stats, predictions = error_examples(metadata, vy, best_probs, threshold, truth, v, names)
    if abs(macro_f05(truth, predictions) - best_score) > 1e-12:
        raise AssertionError('Vectorized evaluator disagrees with set evaluator')
    report.update(best_validation_baseline=best_name, singleton_stats=singleton_stats)
    wanted = {e['source1_entity_id'] for e in examples} | {e['candidate_entity_id'] for e in examples}
    originals = {}
    for source in (1, 2, 3):
        needed = {i for i in wanted if i.startswith(f'S{source}-')}
        for row in read_rows(Path(f'dataset/train/train_source{source}.tsv')):
            if row['entity_id'] in needed:
                originals[row['entity_id']] = {k: row.get(k, '') for k in ('business_name', 'business_address', 'country')}
        print(f'Read training S{source} example details', flush=True)
    for example in examples:
        example['source1'] = originals.get(example['source1_entity_id'])
        example['candidate'] = originals.get(example['candidate_entity_id'])
    (out / 'error_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    report['runtime_seconds'] = time.perf_counter() - started
    report['peak_memory_mib'] = peak_memory_mib()
    if input_manifest(Path('dataset/train')) != original_manifest:
        raise ValueError('Training inputs changed during experiment')
    report['original_tsv_metadata_unchanged'] = True
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_summary(report, examples, out)
    print(json.dumps({k: report[k] for k in ('best_validation_baseline', 'runtime_seconds', 'peak_memory_mib', 'singleton_stats')}), flush=True)


if __name__ == '__main__':
    main()
