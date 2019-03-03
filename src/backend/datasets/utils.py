def get_weights(labels):
    total = len(labels)
    weight_negative = float((labels>=0).sum())/total
    return [weight_negative if l<0 else 1-weight_negative for l in labels]