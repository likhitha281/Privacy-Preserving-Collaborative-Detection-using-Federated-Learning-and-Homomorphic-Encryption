"""Attack experiments against an honest-but-curious aggregation server (see README.md in this folder).

    python -m experiments.attacks.label_flip [--quick]        # one malicious bank flips training labels
    python -m experiments.attacks.gradient_leakage [--quick]  # server inverts a client's update (DLG-style)
"""
