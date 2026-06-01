import json

import pytest

from utils import preferences


class _FakeConfigManager:
    def __init__(self, config_path):
        self._config_path = config_path

    def ensure_config_directory(self):
        return None

    def get_config_path(self, name):
        return self._config_path.parent / name


def _patch_preferences_storage(monkeypatch, tmp_path):
    prefs_file = tmp_path / "user_preferences.json"
    fake_cm = _FakeConfigManager(prefs_file)
    monkeypatch.setattr(preferences, "_config_manager", fake_cm, raising=True)
    monkeypatch.setattr(preferences, "PREFERENCES_FILE", str(prefs_file), raising=False)
    return prefs_file


def test_global_conversation_settings_roundtrip_mouse_tracking(monkeypatch, tmp_path):
    prefs_file = _patch_preferences_storage(monkeypatch, tmp_path)

    payload = {
        "mouseTrackingEnabled": False,
        "mouseTrackingSensitivity": 2.25,
        "subtitleEnabled": True,
        "userLanguage": "zh-CN",
        "ignoredField": "nope",
    }

    assert preferences.save_global_conversation_settings(payload) is True

    stored = json.loads(prefs_file.read_text(encoding="utf-8"))
    assert stored[0]["model_path"] == preferences.GLOBAL_CONVERSATION_KEY
    assert stored[0]["mouseTrackingEnabled"] is False
    assert stored[0]["mouseTrackingSensitivity"] == pytest.approx(2.25)
    assert "ignoredField" not in stored[0]

    loaded = preferences.load_global_conversation_settings()
    assert loaded["mouseTrackingEnabled"] is False
    assert loaded["mouseTrackingSensitivity"] == pytest.approx(2.25)
    assert loaded["subtitleEnabled"] is True
    assert loaded["userLanguage"] == "zh-CN"
    assert "ignoredField" not in loaded


@pytest.mark.parametrize("invalid_sensitivity", [0.05, 3.5])
def test_global_conversation_settings_rejects_out_of_range_mouse_tracking_sensitivity(
    monkeypatch,
    tmp_path,
    invalid_sensitivity,
):
    prefs_file = _patch_preferences_storage(monkeypatch, tmp_path)

    assert preferences.save_global_conversation_settings(
        {
            "mouseTrackingEnabled": True,
            "mouseTrackingSensitivity": invalid_sensitivity,
        }
    ) is True

    stored = json.loads(prefs_file.read_text(encoding="utf-8"))
    assert stored[0]["model_path"] == preferences.GLOBAL_CONVERSATION_KEY
    assert "mouseTrackingSensitivity" not in stored[0]
    assert stored[0]["mouseTrackingEnabled"] is True

    loaded = preferences.load_global_conversation_settings()
    assert "mouseTrackingSensitivity" not in loaded
    assert loaded["mouseTrackingEnabled"] is True
