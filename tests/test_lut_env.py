import importlib


def test_lut_env_overrides(monkeypatch):
    monkeypatch.setenv("NEURALKAPPA_KAPPA_MIN", "0.02")
    monkeypatch.setenv("NEURALKAPPA_KAPPA_MAX", "200.0")
    monkeypatch.setenv("NEURALKAPPA_LUT_RESOLUTION", "12345")

    import neuralkappa._lut as lut_module

    reloaded = importlib.reload(lut_module)
    assert reloaded._GLOBAL_LUT.kappa_min == 0.02
    assert reloaded._GLOBAL_LUT.kappa_max == 200.0
    assert reloaded._GLOBAL_LUT.resolution == 12345
