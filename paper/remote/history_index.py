"""History convention for producing x[t] from a buffer ending at x[t-1]."""


def effective_delay(label, convention="theory"):
    if type(label) is not int or label < 0:
        raise ValueError("delay must be a non-negative integer")
    if convention not in ("theory", "legacy"):
        raise ValueError("unknown delay convention")
    return label if convention == "theory" else max(0, label - 1)


def history_index(t, label, convention="theory"):
    if type(t) is not int or t < 1:
        raise ValueError("t must identify an update, starting at 1")
    return max(0, t - 1 - effective_delay(label, convention))
