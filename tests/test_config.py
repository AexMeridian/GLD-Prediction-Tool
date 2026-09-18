from gold_edge.config import get_settings


def test_settings_load_from_yaml():
    settings = get_settings()
    assert settings.kalshi.series_ticker == "KXGOLD15M"
    assert settings.engine.enter_edge == 0.03
    assert settings.engine.persist_s == 1.5
    assert settings.fees.base_rate == 0.07


def test_settings_default_secrets_are_blank_without_env():
    settings = get_settings()
    assert isinstance(settings.kalshi_api_key_id, str)
    assert isinstance(settings.pyth_api_key, str)
