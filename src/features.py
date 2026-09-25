"""Reusable numeric pair features, entity-level splits, and separate labels.

Features never accept ground truth. Missing text provides zero similarity.
RapidFuzz features are included only when the optional package is installed.
"""

from functools import lru_cache
import math
import random

from src.candidate_improvements import char_ngrams, prepare_weighted
from src.ranking import jaccard

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


FEATURE_NAMES = (
    "name_exact", "name_token_jaccard", "name_bigram_dice", "name_trigram_dice",
    "name_length_ratio", "name_prefix_similarity", "name_rare_token_dice",
    "s1_name_missing", "candidate_name_missing",
    "address_exact", "address_token_jaccard", "address_bigram_dice",
    "shared_numeric_token_count", "numeric_token_jaccard", "address_rare_token_dice",
    "s1_address_missing", "candidate_address_missing", "both_addresses_missing",
    "address_length_ratio", "country_equal", "candidate_is_s2", "candidate_is_s3",
    "candidate_ranking_score", "candidate_rank", "ranking_available",
) + (("name_rapidfuzz_ratio", "name_rapidfuzz_token_set_ratio") if fuzz is not None else ())


def split_s1_entities(entity_ids, validation_fraction=.2, seed=42):
    """Shuffle sorted unique S1 IDs, never candidate pairs."""
    ids = sorted(entity_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("S1 entity IDs must be unique")
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    validation_count = round(len(ids) * validation_fraction)
    if not 0 < validation_count < len(ids):
        raise ValueError("Both splits need at least one S1 entity")
    random.Random(seed).shuffle(ids)
    return {entity_id: "validation" if i < validation_count else "train"
            for i, entity_id in enumerate(ids)}


def label_pair(source1_entity_id, candidate_entity_id, truth):
    """One-to-many membership label; missing S1 truth is an error."""
    return int(candidate_entity_id in truth[source1_entity_id])


def dice(left, right):
    total = len(left) + len(right)
    return 2 * len(left & right) / total if total else 0.0


def length_ratio(left, right):
    maximum = max(len(left), len(right))
    return min(len(left), len(right)) / maximum if maximum else 0.0


def rare_token_dice(left_weights, right_weights, left_total, right_total):
    """Same-country IDF Dice matches Phase 2D.

    For different countries, use the smaller shared-token weight to keep the
    feature symmetric and bounded even when country-specific IDFs differ.
    """
    total = left_total + right_total
    shared = left_weights.keys() & right_weights.keys()
    overlap = math.fsum(min(left_weights[t], right_weights[t]) for t in shared)
    return 2 * overlap / total if total else 0.0


@lru_cache(maxsize=8192)
def address_bigrams(address):
    """Bounded cache; repeated addresses do not create unbounded state."""
    return char_ngrams(address, 2)


def pair_features(left, right, frequencies, ranking_score=None, candidate_rank=None,
                  prepared_left=None, prepared_right=None):
    """Return a fixed-order numeric feature dictionary, without IDs or labels.

    Records use the existing normalization. Pass cached WeightedRecords for
    bulk processing. Scores/ranks must either both be supplied or both absent.
    Rank zero plus ranking_available=0 represents unavailable ranking metadata.
    """
    source = right.entity_id.split("-", 1)[0]
    if source not in ("S2", "S3"):
        raise ValueError("Candidate source must be S2 or S3")
    if (ranking_score is None) != (candidate_rank is None):
        raise ValueError("Supply both ranking score and rank, or neither")
    if candidate_rank is not None and (not isinstance(candidate_rank, int) or candidate_rank < 1):
        raise ValueError("Candidate rank must be a positive integer")
    if ranking_score is not None and (not math.isfinite(ranking_score) or not 0 <= ranking_score <= 1):
        raise ValueError("Candidate ranking score must be finite and in [0, 1]")
    a = prepared_left if prepared_left is not None else prepare_weighted(left, frequencies)
    b = prepared_right if prepared_right is not None else prepare_weighted(right, frequencies)
    prefix = 0
    for x, y in zip(left.name[:4], right.name[:4]):
        if x != y:
            break
        prefix += 1
    result = {
        "name_exact": int(bool(left.name) and left.name == right.name),
        "name_token_jaccard": jaccard(a.basic.name_tokens, b.basic.name_tokens),
        "name_bigram_dice": dice(a.basic.name_bigrams, b.basic.name_bigrams),
        "name_trigram_dice": dice(a.grams, b.grams),
        "name_length_ratio": length_ratio(left.name, right.name),
        "name_prefix_similarity": prefix / 4,
        "name_rare_token_dice": rare_token_dice(a.name_weights, b.name_weights, a.name_total, b.name_total),
        "s1_name_missing": int(not left.name), "candidate_name_missing": int(not right.name),
        "address_exact": int(bool(left.address) and left.address == right.address),
        "address_token_jaccard": jaccard(a.basic.address_tokens, b.basic.address_tokens),
        "address_bigram_dice": dice(address_bigrams(left.address), address_bigrams(right.address)),
        "shared_numeric_token_count": len(a.basic.numbers & b.basic.numbers),
        "numeric_token_jaccard": jaccard(a.basic.numbers, b.basic.numbers),
        "address_rare_token_dice": rare_token_dice(a.address_weights, b.address_weights, a.address_total, b.address_total),
        "s1_address_missing": int(not left.address), "candidate_address_missing": int(not right.address),
        "both_addresses_missing": int(not left.address and not right.address),
        "address_length_ratio": length_ratio(left.address, right.address),
        "country_equal": int(bool(left.country) and left.country == right.country),
        "candidate_is_s2": int(source == "S2"), "candidate_is_s3": int(source == "S3"),
        "candidate_ranking_score": 0.0 if ranking_score is None else ranking_score,
        "candidate_rank": 0 if candidate_rank is None else candidate_rank,
        "ranking_available": int(candidate_rank is not None),
    }
    if fuzz is not None:
        present = bool(left.name and right.name)
        result["name_rapidfuzz_ratio"] = fuzz.ratio(left.name, right.name) / 100 if present else 0.0
        result["name_rapidfuzz_token_set_ratio"] = fuzz.token_set_ratio(left.name, right.name) / 100 if present else 0.0
    # Stable serialization precision; integer features retain their type.
    return {key: round(value, 8) if isinstance(value, float) else value for key, value in result.items()}
