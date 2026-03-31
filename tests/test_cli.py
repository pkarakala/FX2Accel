from fx2accel import cli


def test_main_returns_zero():
    assert cli.main([]) == 0
