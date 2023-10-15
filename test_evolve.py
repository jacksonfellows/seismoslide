import evolve


def test_simplify_feature():
    # Simplifying a feature shouldn't change the results.
    evolver = evolve.load_evolver("train")
    for _ in range(10):
        rf = evolver.random_feature(10, 0.3)
        a = evolver.eval_feature(rf, simplify_first=True)
        b = evolver.eval_feature(rf, simplify_first=False)
        assert (a == b).all()
