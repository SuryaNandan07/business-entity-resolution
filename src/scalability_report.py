"""Read-only checkpoint equivalence and report assembly for Phase 4A.1."""
import argparse
import json
from pathlib import Path
import sqlite3
import re
import pickle
from contextlib import closing
from datetime import datetime

OUT=Path('output/scalability')

def audit(reference,current):
    with closing(sqlite3.connect(f'file:{Path(reference).as_posix()}?mode=ro',uri=True)) as a, closing(sqlite3.connect(f'file:{Path(current).as_posix()}?mode=ro',uri=True)) as b:
        old={r[0] for r in a.execute('SELECT s1 FROM completed')}
        new={r[0] for r in b.execute('SELECT s1 FROM completed')}
        result=dict(reference_queries=len(old),compared_queries=len(old&new),candidate_id_changes=0,feature_value_changes=0)
        for s in sorted(old&new):
            left=dict(a.execute('SELECT candidate,features FROM pairs WHERE s1=?',(s,)))
            right=dict(b.execute('SELECT candidate,features FROM pairs WHERE s1=?',(s,)))
            result['candidate_id_changes']+=len(left.keys()^right.keys())
            result['feature_value_changes']+=sum(left[i]!=right[i] for i in left.keys()&right.keys())
    (OUT/'full_catalog_equivalence.json').write_text(json.dumps(result,indent=2))
    if result['candidate_id_changes'] or result['feature_value_changes']:raise ValueError(f'Full-catalog equivalence failed: {result}')
    print(json.dumps(result));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('reference',nargs='?');p.add_argument('current',nargs='?');a=p.parse_args()
    if a.reference and a.current:audit(a.reference,a.current)
    else:assemble()

def assemble():
    report_path=OUT/'holdout2_candidates.json'
    report=json.loads(report_path.read_text())
    report['s1_count']=report['metrics']['s1_count']
    builds=json.loads((OUT/'cache_build_observations.json').read_text())
    equivalence=json.loads((OUT/'equivalence.json').read_text())
    full_equivalence=json.loads((OUT/'full_catalog_equivalence.json').read_text())
    profile=json.loads((OUT/'reference_profile.json').read_text())
    reference=json.loads((OUT/'full_reference_ranking_profile.json').read_text())
    cache=Path('output/cache/train_catalog')
    from src.catalog_cache import signature
    sources=[Path(s['path']) for s in report['catalog']['signature']['sources']]
    assert json.loads(json.dumps(signature(sources)))==report['catalog']['signature'],'Target metadata changed'
    report['original_target_metadata_unchanged']=True
    with closing(sqlite3.connect(f'file:{(cache/"records.sqlite").as_posix()}?mode=ro',uri=True)) as db:
        s2,s3=db.execute("SELECT sum(entity_id LIKE 'S2-%'),sum(entity_id LIKE 'S3-%') FROM records").fetchone()
    assert (s2,s3)==(5034616,5285603),(s2,s3)
    report['verified_catalog_source_counts']={'S2':s2,'S3':s3}
    postings_files=list(cache.glob('postings-*.pickle'))
    if len(postings_files)==1:
        with postings_files[0].open('rb') as f:index=pickle.load(f)
        statistics=index.statistics()
        for kind,stats in statistics.items():
            nonempty=[len(v) for k,v in index.postings.items() if k[0]==kind and v is not None and len(v)]
            stats['nonempty_keys']=len(nonempty)
            stats['average_nonempty_postings']=sum(nonempty)/max(1,len(nonempty))
        (OUT/'full_index_statistics.json').write_text(json.dumps(statistics,indent=2))
        del index
    samples=[]
    for path in OUT.glob('*memory_samples.jsonl'):
        samples.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rank_build=report['ranking_index'].get('build_seconds',report['ranking_index'].get('original_build_seconds',0))
    retrieval_build=builds['normalized_build_seconds']+builds['postings_publication_wall_seconds']
    evaluation_wall=report['holdout_processing_wall_seconds']
    timing=dict(full_index_build_seconds=retrieval_build+rank_build,
        cache_load_seconds=report['cache_load_seconds'],candidate_generation_seconds=report['timings']['candidate_generation'],
        ranking_seconds=report['timings']['ranking'],feature_seconds=report['timings']['features'],
        inference_seconds=None,holdout_evaluation_wall_seconds=evaluation_wall,
        cold_pipeline_seconds=retrieval_build+rank_build+report['cache_load_seconds']+evaluation_wall,
        elapsed_including_interventions_seconds=report_path.stat().st_mtime-(datetime.fromisoformat(report['catalog']['created_utc']).timestamp()-report['catalog']['seconds']),
        peak_rss_bytes=max([report['peak_memory']['peak_rss']]+[s['peak_rss'] for s in samples]),
        peak_private_bytes=max(s['private'] for s in samples),cache_bytes=report['cache_bytes'],
        estimate_1_7m_evaluation_seconds=evaluation_wall/20000*1700000,
        estimate_1_7m_with_85_batch_indexes_seconds=builds['normalized_build_seconds']+85*(builds['postings_publication_wall_seconds']+rank_build+evaluation_wall))
    testlog=(OUT/'tests_current.log').read_text();match=re.search(r'Ran (\d+) tests',testlog)
    report['scalability_summary']=timing;report['equivalence']=equivalence;report['full_catalog_equivalence']=full_equivalence
    report['tests']={'runner_count':int(match.group(1)) if match else None,
        'tests_passed':len(re.findall(r'^test_.* \.\.\. ok$',testlog,re.M)),
        'suite_passed':bool(re.search(r'^OK\s*$',testlog,re.M)),
        'blocked_module':'test_modeling: Windows Application Control blocked sklearn.utils.arrayfuncs',
        'blocked_tests':5,'expected_test_count':79}
    report['files_created']=['SCALABILITY.md','src/audit_catalog_checkpoint.py',
        'src/catalog_cache.py','src/compact_ranking.py','src/disk_ranking.py',
        'src/profile_catalog.py','src/resource_guard.py','src/scalability_report.py',
        'src/scalable_holdout2.py','tests/test_catalog_cache.py',
        'tests/test_compact_ranking.py','tests/test_scalability_report.py',
        'tests/test_scalable_holdout2.py']
    report['files_modified']=['src/holdout2.py']
    (OUT/'final_report.json').write_text(json.dumps(report,indent=2))
    candidate_review(report,profile,reference)
    print(json.dumps(timing,indent=2))


def candidate_review(report,profile,reference):
    """Publish available measurements without importing or evaluating models."""
    t=report['scalability_summary'];e=report['equivalence'];f=report['full_catalog_equivalence']
    lines=['# Phase 4A.1 — candidate retrieval review','',
        'Full training-catalog indexing and Holdout 2 Top-50 retrieval are complete. HGB remains the production baseline. Frozen-model inference is deferred at the user’s request pending an approved Python environment.',
        f'Targets: 10,320,219 (S2: 5,034,616; S3: 5,285,603). S1 rows 60,001–80,000: {report["s1_count"]:,}; prior overlap: {report["overlap_count"]}; K={report["k"]}; maximum candidates/S1={report["maximum_candidates_per_s1"]}; zero-candidate S1={report["zero_candidate_s1_count"]}.','',
        '## Candidate recall','','| Group | S1 | True links | Recovered links | Recall | Candidate pairs |','|---|---:|---:|---:|---:|---:|']
    for group,m in [('All',report['metrics']),*report['by_source'].items(),*report['by_country'].items()]:
        recall='N/A' if m['candidate_recall'] is None else f'{m["candidate_recall"]:.6%}'
        lines.append(f'| {group or "(missing country)"} | {m["s1_count"]} | {m["true_links"]} | {m["candidate_true_links"]} | {recall} | {m["candidate_pairs"]} |')
    lines+=['','## Equivalence and engineering changes','',
        f'Development comparison: {e["queries"]} S1, {e["targets"]:,} targets; zero broad-candidate or ordered Top-50 changes, zero true links gained/lost, identical recall ({e["reference_recall"]:.6%}). Full-catalog reference checkpoint: {f["compared_queries"]} queries; candidate-ID changes={f["candidate_id_changes"]}; feature-byte changes={f["feature_value_changes"]}.',
        'Construction streams four required columns in 10,000-row chunks into SQLite. Integer postings, exact query-driven rescue keys, capped rescue postings using unchanged frequency limits, bounded disk-backed ranking segments, and committed checkpoints avoid full-catalog Python string/set structures. Cache validity includes source metadata, code/schema/configuration and frozen ranking inputs. No candidate rule, feature, K or threshold was changed.',
        'Recovery validated SQLite integrity, contiguous committed chunks, integer postings and sampled ranking totals. It resumed at record 3,810,000; completed chunks were not rebuilt.',
        f'The old scorer with scalar disk lookup took {reference["ranking_seconds"]:.3f}s for {reference["count"]} full-catalog queries ({reference["broad_candidates"]:,} broad pairs). Vector screening followed by unchanged exact rescoring addressed this measured ranking bottleneck.',
        'The legacy rescue build was measured on 100,000 targets, not all 10.3M: trigram postings were its largest measured rescue structure. Complete legacy full-catalog stage timings are unavailable. The earlier full rescue slowdown cannot be assigned an exact full-catalog runtime from this bounded profile.','',
        '| Reference stage (100k targets, 100 S1) | Seconds | RSS MiB | RSS delta MiB |','|---|---:|---:|---:|']
    for name,m in profile.items():
        if isinstance(m,dict) and 'rss_mib' in m:lines.append(f'| {name} | {m["seconds"]:.3f} | {m["rss_mib"]:.1f} | {m["rss_delta_mib"]:.1f} |')
    lines+=['','Isolated rescue stages replay the aggregate work; do not add their times to it. RSS deltas include allocator effects. Posting counts and average/max lengths are in reference_profile.json and full_index_statistics.json.','',
        '## Timing and memory','','| Measurement | Value |','|---|---:|']
    for name,value in t.items():lines.append(f'| {name} | {value} |')
    lines+=['',
        'Full index build is the sum of normalized storage, retrieval postings and disk-backed ranking construction. Cold-pipeline time adds measured cache load and Holdout processing wall time; it excludes development checks, rejected attempts and the user pause. Elapsed including interventions includes those delays. Inference time is unavailable, not zero.',
        'The 1.7M estimates use measured throughput and assume comparable query density and the same target catalog. Evaluation here means candidate generation, ranking and features only. The first estimate excludes new query-key indexes; the second includes 85 batches of 20,000 and their measured index construction. Neither includes model inference; neither is a guaranteed runtime. No competition files were processed.',
        'Peak RSS is the OS high-water mark across recorded attempts; private memory is sampled. The earlier in-memory ranking attempt stopped near 1.44 GiB resident when Windows available RAM fell below 2 GiB. Disk-backed construction replaced it. The Python 6 GB cap was not reached.',
        f'Complete-suite result: {report["tests"]}. See tests_current.log for per-test outcomes; historical passes are not represented as current passes.','',
        '## Remaining blocked work','',
        'Windows Application Control blocked scikit-learn’s native sklearn.utils.arrayfuncs module. At the user’s direction, neither frozen model was retried. An approved Python environment with compatible dependencies is required to load the existing model artifacts and evaluate the saved identical candidate-feature checkpoint.',
        'Still unavailable: HGB and LightGBM macro F0.5, pair precision/recall, predicted full-ground-truth recall, predicted links and average links/S1, singleton metrics, zero-prediction counts, source/country model breakdowns, inference timings, and the model adoption comparison. Candidate recall above is available independently. HGB remains selected; no conclusion about LightGBM superiority is supported.',
        f'Preserved candidate/feature checkpoint: `{report["checkpoint_path"]}`. All full-catalog caches and completed outputs remain in place and gitignored. No submission files, model training, threshold tuning, merge or adoption occurred. Stop for review.']
    lines+=['','## Files','','Created: '+', '.join(f'`{p}`' for p in report['files_created'])+'.',
        'Modified: '+', '.join(f'`{p}`' for p in report['files_modified'])+'.',
        'Generated artifacts: `output/cache/train_catalog/` and `output/scalability/` (gitignored). Existing `catboost_info/` was not modified by this work.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
