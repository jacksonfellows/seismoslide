import evolve

train = evolve.load_evolver("train")
valid = evolve.load_evolver("valid")


def dumb_evolve(max_depth=5, terminal_proba=0.2, initial_pop=None):
    pop = [] if initial_pop is None else initial_pop
    best_score = 0
    while True:
        while True:
            f = train.random_feature(max_depth, terminal_proba)
            if train.feature_rank(f) == 0:
                break
        f = train.simplify_feature(f)
        print(f"{pop=}")
        print(f"{f=}")
        score = evolve.score_model(train, valid, pop + [f])
        if score > best_score:
            print("adding feature!")
            pop.append(f)
            best_score = score


# Ran for a while.
pop = [
    ("median", ("sum", ("abs_fft", ("mean", ("abs_fft", "N"))))),
    ("max", ("abs_fft", ("envelope", ("skew", ("max", "N"))))),
    ("abs", ("min", "N3")),
    ("max", ("median", ("corr", "N3", ("second_half", ("sum", "N"))))),
    ("sum", "N"),
    ("sum", ("max", ("max", ("argmin", ("first_half", "N2"))))),
    (
        "first_half",
        ("argmax", ("corr", ("second_half", ("mean", "E1")), ("min", ("mean", "E")))),
    ),
    ("mean", "E2"),
    ("kurtosis", ("kurtosis", "N")),
    ("median", ("skew", ("skew", ("min", "E2")))),
    ("second_half", ("second_half", ("kurtosis", ("median", ("argmin", "N2"))))),
    ("median", ("envelope", ("second_half", "E"))),
    ("kurtosis", "E2"),
    ("first_half", ("abs", ("abs", ("max", "E3")))),
    ("argmax", "Z"),
    ("median", ("abs", "E1")),
    ("second_half", ("first_half", ("argmax", ("mean", ("abs", "E1"))))),
    ("argmin", ("sum", ("mean", ("median", ("max", "N2"))))),
    ("argmin", ("second_half", ("mean", ("envelope", "Z2")))),
    ("min", ("max", ("argmax", "N"))),
    ("median", ("abs", "Z")),
    ("envelope", ("min", "Z")),
    (
        "mean",
        (
            "corr",
            ("corr", ("skew", "N3"), ("median", ("second_half", "Z3"))),
            ("+", ("abs", ("skew", "E1")), "E3"),
        ),
    ),
    ("max", ("second_half", ("skew", "N2"))),
    ("min", ("min", "E1")),
    (
        "+",
        ("envelope", ("envelope", ("argmax", ("+", "Z1", "Z")))),
        ("min", ("abs_fft", ("skew", ("envelope", "Z2")))),
    ),
    ("argmin", ("median", ("max", ("median", "E")))),
    ("first_half", ("mean", ("max", ("abs_fft", ("envelope", "Z1"))))),
    ("abs", ("median", ("argmax", ("envelope", ("abs_fft", "Z2"))))),
    ("kurtosis", ("skew", ("envelope", ("mean", ("min", "N3"))))),
    (
        "sum",
        (
            "max",
            ("corr", ("first_half", ("corr", "N1", "Z1")), ("envelope", ("abs", "Z3"))),
        ),
    ),
    ("argmax", ("min", ("corr", ("abs", ("argmax", "E")), ("max", ("+", "E3", "Z1"))))),
    ("second_half", ("first_half", ("argmax", ("abs_fft", ("first_half", "N"))))),
    ("abs_fft", ("median", ("abs_fft", ("abs", "Z2")))),
    ("min", ("median", ("skew", ("first_half", ("median", "E"))))),
    (
        "max",
        (
            "min",
            ("+", ("kurtosis", ("+", "Z2", "Z")), ("+", ("mean", "N"), ("skew", "Z3"))),
        ),
    ),
    ("argmax", ("envelope", ("argmin", "Z"))),
    ("skew", "N2"),
    ("argmin", ("skew", "E")),
    ("second_half", ("argmin", ("second_half", ("min", ("corr", "E3", "E1"))))),
    ("mean", ("first_half", ("sum", ("abs_fft", "Z1")))),
    ("skew", ("+", "E2", ("min", ("second_half", ("second_half", "E3"))))),
    ("kurtosis", "Z1"),
    ("sum", ("mean", ("argmax", ("min", ("+", "E1", "E3"))))),
    ("first_half", ("max", ("first_half", ("max", ("abs_fft", "E"))))),
    ("kurtosis", ("second_half", "Z1")),
    ("min", ("median", ("+", "Z3", ("sum", ("abs", "N3"))))),
    (
        "min",
        ("mean", ("corr", ("skew", ("abs_fft", "N")), ("argmax", ("+", "Z2", "E1")))),
    ),
    (
        "+",
        ("envelope", ("sum", "Z2")),
        ("kurtosis", ("second_half", ("max", ("median", "E")))),
    ),
    ("argmin", ("abs_fft", ("argmax", ("kurtosis", ("abs_fft", "Z1"))))),
    ("envelope", ("argmin", ("envelope", ("max", "E")))),
    (
        "kurtosis",
        ("mean", ("+", ("argmin", ("sum", "N2")), ("abs_fft", ("argmin", "E2")))),
    ),
    ("min", ("argmin", ("corr", ("second_half", "N"), ("abs_fft", ("median", "E2"))))),
]

pop_simpl = [train.simplify_feature(f) for f in pop]


def dumb_reduce(pop, best_score=0.93):
    i = 0
    while i < len(pop):
        print(f"{pop=}")
        print(f"f={pop[i]}")
        minus = pop[:i] + pop[i + 1 :]
        score = evolve.score_model(train, valid, minus)
        if score > best_score:
            print(f"removing feature {pop[i]}!")
            pop = minus
        else:
            i += 1


pop_reduced = [
    ("max", "N"),
    ("kurtosis", "N"),
    ("mean", ("abs", "E1")),
    ("max", "N2"),
    ("mean", ("envelope", "Z2")),
    ("argmax", "N"),
    ("median", ("abs", "Z")),
    ("median", "E"),
    ("abs", ("argmax", ("envelope", ("abs_fft", "Z2")))),
    ("argmax", ("abs_fft", ("first_half", "N"))),
    ("median", "E"),
    ("+", ("kurtosis", ("+", "Z2", "Z")), ("+", ("mean", "N"), ("skew", "Z3"))),
    ("argmin", "Z"),
    ("skew", "N2"),
    ("min", ("corr", "E3", "E1")),
    ("sum", ("abs_fft", "Z1")),
    ("skew", ("+", "E2", ("min", ("second_half", ("second_half", "E3"))))),
    ("kurtosis", "Z1"),
    ("max", ("abs_fft", "E")),
    ("kurtosis", ("second_half", "Z1")),
    ("median", ("+", "Z3", ("sum", ("abs", "N3")))),
    ("skew", ("abs_fft", "N")),
    ("+", ("sum", "Z2"), ("median", "E")),
    ("kurtosis", ("abs_fft", "Z1")),
    ("+", ("sum", "N2"), ("argmin", "E2")),
    ("argmin", ("corr", ("second_half", "N"), ("median", "E2"))),
]


gp_run = [
    ("kurtosis", "Z1"),
    ("+", ("argmax", "E1"), ("max", "Z2")),
    ("median", ("corr", ("corr", "N", ("kurtosis", "N")), ("kurtosis", "N"))),
    ("skew", ("abs_fft", "Z")),
    ("sum", ("first_half", ("first_half", "E2"))),
    ("argmax", ("envelope", "Z")),
    ("mean", "N1"),
    ("mean", "Z1"),
    ("sum", ("corr", ("first_half", "E1"), ("kurtosis", "E1"))),
    ("argmin", ("envelope", ("envelope", ("abs_fft", "Z1")))),
    ("mean", ("corr", "E", ("corr", "N1", "E"))),
    ("median", ("corr", ("sum", ("second_half", "E")), "N3")),
    ("sum", ("corr", ("kurtosis", ("second_half", "Z1")), "Z")),
    ("sum", ("first_half", ("second_half", ("corr", "N2", "E2")))),
    ("sum", ("first_half", "N")),
    ("max", ("corr", "N1", ("skew", "N1"))),
    ("median", ("corr", "E3", ("sum", "E2"))),
    (
        "median",
        ("first_half", ("+", ("skew", "Z1"), ("corr", "N2", ("corr", "N2", "N")))),
    ),
    ("max", ("corr", ("skew", "Z"), ("+", ("second_half", "Z1"), "Z"))),
    ("mean", ("corr", ("mean", "Z1"), "Z2")),
]

pop_gp_reduced = [
    ("mean", ("envelope", "Z2")),
    ("kurtosis", "Z1"),
    ("max", ("abs_fft", "E")),
    ("kurtosis", ("second_half", "Z1")),
    ("+", ("sum", "Z2"), ("median", "E")),
    ("+", ("sum", "N2"), ("argmin", "E2")),
    ("kurtosis", "Z1"),
    ("+", ("argmax", "E1"), ("max", "Z2")),
    ("skew", ("abs_fft", "Z")),
    ("argmax", ("envelope", "Z")),
    ("mean", "Z1"),
    ("sum", ("corr", ("first_half", "E1"), ("kurtosis", "E1"))),
    ("argmin", ("envelope", ("envelope", ("abs_fft", "Z1")))),
    ("sum", ("corr", ("kurtosis", ("second_half", "Z1")), "Z")),
]

pop_gp_reduced_2 = [
    ("mean", ("envelope", "Z2")),
    ("max", ("abs_fft", "E")),
    ("kurtosis", ("second_half", "Z1")),
    ("+", ("sum", "N2"), ("argmin", "E2")),
    ("kurtosis", "Z1"),
    ("+", ("argmax", "E1"), ("max", "Z2")),
    ("skew", ("abs_fft", "Z")),
    ("argmax", ("envelope", "Z")),
    ("sum", ("corr", ("first_half", "E1"), ("kurtosis", "E1"))),
    ("argmin", ("envelope", ("envelope", ("abs_fft", "Z1")))),
    ("sum", ("corr", ("kurtosis", ("second_half", "Z1")), "Z")),
]

# Expanded pop_gp_reduced_2.
pop_2 = [
    ("mean", ("envelope", "Z2")),
    ("max", ("abs_fft", "E")),
    ("kurtosis", ("second_half", "Z1")),
    ("+", ("sum", "N2"), ("argmin", "E2")),
    ("kurtosis", "Z1"),
    ("+", ("argmax", "E1"), ("max", "Z2")),
    ("skew", ("abs_fft", "Z")),
    ("argmax", ("envelope", "Z")),
    ("sum", ("corr", ("first_half", "E1"), ("kurtosis", "E1"))),
    ("argmin", ("envelope", ("envelope", ("abs_fft", "Z1")))),
    ("sum", ("corr", ("kurtosis", ("second_half", "Z1")), "Z")),
    ("mean", ("first_half", "E")),
    ("abs", ("max", "N1")),
    ("kurtosis", ("second_half", ("abs_fft", "Z"))),
    ("abs", ("median", "E")),
    ("kurtosis", "N2"),
    ("argmax", "N"),
    ("abs", ("min", "E3")),
    ("abs", ("argmax", "N")),
    ("median", "E"),
    ("min", ("+", ("kurtosis", "Z"), "E")),
    ("min", "E1"),
    ("argmax", ("envelope", "N1")),
    ("argmin", ("corr", ("argmax", "N1"), ("corr", "Z2", "E3"))),
    ("+", ("median", ("envelope", ("envelope", "E"))), ("kurtosis", "E2")),
    ("abs", ("argmin", "Z1")),
    ("argmax", "Z2"),
    ("kurtosis", "N1"),
    ("argmax", "N3"),
    ("sum", ("+", ("envelope", "E1"), "N2")),
    ("median", "N3"),
    ("median", "Z1"),
    ("argmin", "Z"),
]
