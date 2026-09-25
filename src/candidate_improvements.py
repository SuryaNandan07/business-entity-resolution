"""Country-neutral, frequency-based candidate and ranking experiments.

Frequencies come only from the retained training target pool, without labels.
No changes to the original normalization or baseline algorithms are needed.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
import math
import unicodedata

from src.ranking import prepare


def script_profile(text):
    """Approximate scripts using Unicode letter-name prefixes, not languages.

    This is a diagnostic heuristic, not a complete Unicode Script property
    implementation. Combining marks, punctuation, and digits are ignored.
    """
    counts = Counter()
    for character in text:
        if unicodedata.category(character).startswith("L"):
            name = unicodedata.name(character, "UNKNOWN")
            script = name.split()[0]
            if script == "CJK":
                script = "HAN"
            counts[script] += 1
    dominant = min(counts, key=lambda s: (-counts[s], s)) if counts else None
    return {"scripts": sorted(counts), "dominant": dominant}


def char_ngrams(name, n=3):
    compact = "".join(name.split())
    return frozenset(compact[i:i+n] for i in range(len(compact)-n+1))


@dataclass(slots=True)
class Frequencies:
    documents: Counter
    name: Counter
    address: Counter
    grams: Counter

    @classmethod
    def fit(cls, records):
        result = cls(Counter(), Counter(), Counter(), Counter())
        for r in records:
            result.documents[r.country] += 1
            result.name.update((r.country, t) for t in set(r.name.split()))
            result.address.update((r.country, t) for t in set(r.address.split()))
            result.grams.update((r.country, g) for g in char_ngrams(r.name))
        return result

    def idf(self, country, token, field):
        """Smoothed inverse document frequency; common tokens weigh less."""
        return 1 + math.log((self.documents[country] + 1) /
                            (getattr(self, field)[country, token] + 1))

    def common(self, country, token, field="name"):
        return getattr(self, field)[country, token] >= .01 * max(1, self.documents[country])


@dataclass(slots=True)
class WeightedRecord:
    basic: object
    name_weights: dict
    address_weights: dict
    name_total: float
    address_total: float
    grams: frozenset


def prepare_weighted(record, frequencies):
    basic = prepare(record)
    names = {t: frequencies.idf(record.country, t, "name") for t in basic.name_tokens}
    addresses = {t: frequencies.idf(record.country, t, "address") for t in basic.address_tokens}
    return WeightedRecord(basic, names, addresses, math.fsum(names.values()),
                          math.fsum(addresses.values()), char_ngrams(record.name))


def weighted_overlap(left, right, left_total, right_total):
    """Return IDF-weighted Dice and containment; empty evidence is zero."""
    shared = left.keys() & right.keys()
    mass = math.fsum(left[t] for t in shared)
    total = left_total + right_total
    minimum = min(left_total, right_total)
    return (2 * mass / total if total else 0,
            mass / minimum if minimum else 0)


def improved_signals(left, right, frequencies):
    """Return name evidence, address evidence, and rare-number agreement."""
    a, b = left.basic, right.basic
    if not a.country or a.country != b.country:
        return None
    name_dice, _ = weighted_overlap(left.name_weights, right.name_weights,
                                     left.name_total, right.name_total)
    gram_size = len(a.name_bigrams) + len(b.name_bigrams)
    char_dice = 2 * len(a.name_bigrams & b.name_bigrams) / gram_size if gram_size else 0
    exact = float(bool(a.name) and a.name == b.name)
    name = max(exact, .65 * name_dice + .35 * char_dice)
    address_dice, containment = weighted_overlap(left.address_weights, right.address_weights,
                                                 left.address_total, right.address_total)
    shared_words = {t for t in a.address_tokens & b.address_tokens if not t.isdecimal()}
    # Containment helps truncated addresses, but one city or number is not enough.
    distinctive = any(not frequencies.common(a.country, t, "address") for t in shared_words)
    address = .65 * address_dice + .35 * containment if len(shared_words) >= 2 and distinctive else address_dice
    shared_numbers = a.numbers & b.numbers
    # A frequent number contributes much less than a rare house/postal number.
    maximum_idf = 1 + math.log(frequencies.documents[a.country] + 1)
    numeric = max((left.address_weights[t] / maximum_idf for t in shared_numbers), default=0)
    return name, address, numeric


def improved_score(signals, version="address_rescue"):
    if signals is None:
        return float("-inf")
    name, address, numeric = signals
    weighted = .65 * name + .30 * address + .05 * numeric
    if version == "weighted":
        return round(weighted, 12)
    if version == "address_rescue":
        # Quantize harmless float noise, then the caller breaks ties by ID.
        return round(max(weighted, .85 * address + .10 * name + .05 * numeric), 12)
    raise ValueError(f"Unknown score version: {version}")


class AdditionalIndex:
    """Bounded postings, added to (never substituted for) broad baseline C.

    Token rules: any shared name token occurring in <=200 pool records, OR
    at least two shared nonnumeric address tokens each occurring in <=300.
    N-gram rule: at least two hits among the query's four rarest character
    trigrams, each with <=200 pool postings. All keys include country.
    Common keys are omitted explicitly, not silently truncated.
    """

    def __init__(self, records, frequencies):
        self.frequencies = frequencies
        self.name = defaultdict(list)
        self.address = defaultdict(list)
        self.grams = defaultdict(list)
        for r in records:
            if not r.country:
                continue
            for token in set(r.name.split()):
                if frequencies.name[r.country, token] <= 200:
                    self.name[r.country, token].append(r.entity_id)
            for token in set(r.address.split()):
                if not token.isdecimal() and frequencies.address[r.country, token] <= 300:
                    self.address[r.country, token].append(r.entity_id)
            for gram in char_ngrams(r.name):
                if frequencies.grams[r.country, gram] <= 200:
                    self.grams[r.country, gram].append(r.entity_id)

    def candidates(self, record):
        if not record.country:
            return set(), set()
        names = set(chain.from_iterable(self.name.get((record.country, t), ()) for t in set(record.name.split())))
        votes = Counter(chain.from_iterable(self.address.get((record.country, t), ()) for t in set(record.address.split()) if not t.isdecimal()))
        tokens = names | {entity_id for entity_id, count in votes.items() if count >= 2}
        grams = [g for g in char_ngrams(record.name) if (record.country, g) in self.grams]
        grams.sort(key=lambda g: (len(self.grams[record.country, g]), g))
        votes = Counter(chain.from_iterable(self.grams[record.country, g] for g in grams[:4]))
        ngrams = {entity_id for entity_id, count in votes.items() if count >= 2}
        return tokens, ngrams


def failure_categories(left, right, frequencies):
    """Overlapping observable proxies; do not imply known semantic causes."""
    a, b = prepare(left), prepare(right)
    sa, sb = script_profile(left.name), script_profile(right.name)
    at, bt = left.name.split(), right.name.split()
    words_a = a.address_tokens - a.numbers
    words_b = b.address_tokens - b.numbers
    n = len(a.name_bigrams) + len(b.name_bigrams)
    gram_dice = 2 * len(a.name_bigrams & b.name_bigrams) / n if n else 0
    flags = {
        "country_missing_or_different": not a.country or a.country != b.country,
        "different_dominant_scripts": bool(sa["dominant"] and sb["dominant"] and sa["dominant"] != sb["dominant"]),
        "disjoint_letter_scripts": bool(sa["scripts"] and sb["scripts"] and not set(sa["scripts"]) & set(sb["scripts"])),
        "first_name_token_changed": bool(at and bt and at[0] != bt[0]),
        "name_tokens_reordered": bool(at and bt and at != bt and set(at) == set(bt)),
        "no_shared_name_tokens": not a.name_tokens & b.name_tokens,
        "spelling_change_proxy": bool(sa["dominant"] == sb["dominant"] and a.name != b.name and gram_dice >= .5),
        "abbreviation_proxy": any(len(x) >= 2 and len(y) > len(x) and y.startswith(x) for x in a.name_tokens for y in b.name_tokens) or any(len(y) >= 2 and len(x) > len(y) and x.startswith(y) for x in a.name_tokens for y in b.name_tokens),
        "address_missing": not a.address_tokens or not b.address_tokens,
        "nonempty_addresses_differ": bool(a.address_tokens and b.address_tokens and left.address != right.address),
        "no_shared_address_words": not words_a & words_b,
        "address_numbers_disjoint": bool(a.numbers and b.numbers and not a.numbers & b.numbers),
        "address_numbers_missing_on_either_side": not a.numbers or not b.numbers,
        "short_name_on_either_side": len(at) <= 2 or len(bt) <= 2,
        "all_name_tokens_common_on_either_side": any(bool(tokens) and all(frequencies.common(a.country, t) for t in tokens) for tokens in (a.name_tokens, b.name_tokens)),
        "weak_character_name_overlap": gram_dice < .2,
    }
    return [key for key, value in flags.items() if value]
