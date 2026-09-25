"""Entity-level macro F0.5, including entities with no predicted links."""


def entity_f05(true_ids, predicted_ids):
    """Duplicates do not count twice. Both sets empty is a perfect result."""
    truth, predicted = set(true_ids), set(predicted_ids)
    if not truth and not predicted:
        return 1.0
    # Equivalent to the precision/recall formula, without division by zero.
    return 1.25 * len(truth & predicted) / (len(predicted) + 0.25 * len(truth))


def macro_f05(truth, predictions):
    """Evaluate EVERY entity in truth, including those absent in predictions."""
    if not truth:
        raise ValueError("At least one evaluation entity is required")
    if set(predictions) - set(truth):
        raise ValueError("Predictions contain unknown evaluation entities")
    return sum(entity_f05(ids, predictions.get(entity_id, ()))
               for entity_id, ids in truth.items()) / len(truth)
