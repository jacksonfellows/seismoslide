import evolve


def test_simplify_feature():
    # Simplifying a feature shouldn't change the results.
    for _ in range(10):
        rf = evolve.random_feature(10)
        a = evolve.eval_feature(rf, simplify_first=True)
        b = evolve.eval_feature(rf, simplify_first=False)
        assert (a == b).all()
