import json
from types import SimpleNamespace

from utils.live2d_resources import (
    compose_expression_refs,
    derive_emotion_mapping,
    load_emotion_overlay,
    load_vtube_expression_refs,
    normalize_relative_asset_path,
    resolve_live2d_context,
    sanitize_emotion_mapping,
    save_overlay_mapping,
    scan_live2d_assets,
)


def test_scan_yui_style_model_with_trailing_space(tmp_path):
    model_dir = tmp_path / "0403原皮YUI-导出"
    expressions_dir = model_dir / "expressions"
    texture_dir = model_dir / "YUI0403yuanpi .8192"
    expressions_dir.mkdir(parents=True)
    texture_dir.mkdir()
    (model_dir / "yui0403yuanpi .model3.json").write_text(
        json.dumps(
            {
                "Version": 3,
                "FileReferences": {
                    "Moc": "YUI0403yuanpi .moc3",
                    "Textures": ["YUI0403yuanpi .8192/texture_00.png"],
                    "Expressions": [
                        {"Name": "001", "File": "expressions/001.exp3.json"},
                        {"Name": "by", "File": "expressions/by.exp3.json"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (expressions_dir / "001.exp3.json").write_text("{}", encoding="utf-8")
    (expressions_dir / "by.exp3.json").write_text("{}", encoding="utf-8")

    assets = scan_live2d_assets(str(model_dir))

    assert assets["motion_files"] == []
    assert assets["expression_files"] == [
        "expressions/001.exp3.json",
        "expressions/by.exp3.json",
    ]
    assert normalize_relative_asset_path("YUI0403yuanpi .moc3") == "YUI0403yuanpi .moc3"


def test_xinghai_style_vtube_refs_fill_unregistered_expressions(tmp_path):
    model_dir = tmp_path / "星海伊束小天"
    expressions_dir = model_dir / "expressions"
    expressions_dir.mkdir(parents=True)
    (model_dir / "xinghai.model3.json").write_text(
        json.dumps({"Version": 3, "FileReferences": {"Moc": "xinghai.moc3"}}),
        encoding="utf-8",
    )
    (expressions_dir / "happy.exp3.json").write_text("{}", encoding="utf-8")
    (expressions_dir / "angry.exp3.json").write_text("{}", encoding="utf-8")
    (model_dir / "profile.vtube.json").write_text(
        json.dumps(
            {
                "Actions": [
                    {"Type": "ToggleExpression", "File": "happy.exp3.json"},
                    {"Type": "ToggleExpression", "File": "expressions/angry.exp3.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assets = scan_live2d_assets(str(model_dir))
    vtube_refs = load_vtube_expression_refs(str(model_dir))
    refs = compose_expression_refs({"FileReferences": {}}, assets["expression_files"], vtube_refs)

    assert assets["motion_files"] == []
    assert vtube_refs == ["happy.exp3.json", "expressions/angry.exp3.json"]
    assert [(ref["source"], ref["file"]) for ref in refs[:2]] == [
        ("vtube", "expressions/happy.exp3.json"),
        ("vtube", "expressions/angry.exp3.json"),
    ]


def test_yui_empty_registered_expressions_falls_back_to_disk_candidates(tmp_path):
    model_dir = tmp_path / "yui"
    expressions_dir = model_dir / "expressions"
    motions_dir = model_dir / "motions"
    expressions_dir.mkdir(parents=True)
    motions_dir.mkdir()
    (expressions_dir / "smile.exp3.json").write_text("{}", encoding="utf-8")
    (expressions_dir / "wink.exp3.json").write_text("{}", encoding="utf-8")
    (motions_dir / "idle.motion3.json").write_text("{}", encoding="utf-8")

    assets = scan_live2d_assets(str(model_dir))
    refs = compose_expression_refs(
        {"FileReferences": {"Expressions": [], "Motions": {"Idle": [{"File": "motions/idle.motion3.json"}]}}},
        assets["expression_files"],
        [],
    )

    assert assets["motion_files"] == ["motions/idle.motion3.json"]
    assert [ref["source"] for ref in refs] == ["disk", "disk"]
    assert [ref["file"] for ref in refs] == ["expressions/smile.exp3.json", "expressions/wink.exp3.json"]


def test_empty_model_returns_empty_assets_refs_and_mapping(tmp_path):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    assets = scan_live2d_assets(str(model_dir))
    refs = compose_expression_refs({"Version": 3}, assets["expression_files"], [])
    mapping = derive_emotion_mapping({"Version": 3})

    assert assets == {"expression_files": [], "motion_files": []}
    assert refs == []
    assert mapping == {"motions": {}, "expressions": {}}


def test_compose_expression_refs_preserves_duplicate_registered_files():
    config = {
        "FileReferences": {
            "Expressions": [
                {"Name": "happy_a", "File": "expressions/a.exp3.json"},
                {"Name": "sad_a", "File": "expressions/a.exp3.json"},
            ]
        }
    }

    refs = compose_expression_refs(config, ["expressions/a.exp3.json"], [])

    assert [ref["name"] for ref in refs] == ["happy_a", "sad_a"]
    assert [ref["file"] for ref in refs] == [
        "expressions/a.exp3.json",
        "expressions/a.exp3.json",
    ]


def test_compose_expression_refs_resolves_unique_basename_and_marks_ambiguous():
    refs = compose_expression_refs(
        {"FileReferences": {"Expressions": []}},
        [
            "expressions/unique.exp3.json",
            "a/dupe.exp3.json",
            "b/dupe.exp3.json",
        ],
        ["unique.exp3.json", "dupe.exp3.json"],
    )

    unique = next(ref for ref in refs if ref["source"] == "vtube" and ref["name"] == "unique")
    ambiguous = next(ref for ref in refs if ref["source"] == "vtube" and ref["file"] == "dupe.exp3.json")

    assert unique["file"] == "expressions/unique.exp3.json"
    assert unique["exists"] is True
    assert ambiguous["ambiguous"] is True
    assert ambiguous["exists"] is False


def test_compose_expression_refs_marks_missing_registered_expression():
    refs = compose_expression_refs(
        {"FileReferences": {"Expressions": [{"Name": "missing", "File": "missing.exp3.json"}]}},
        [],
        [],
    )

    assert refs == [
        {
            "name": "missing",
            "file": "missing.exp3.json",
            "source": "model",
            "exists": False,
            "ambiguous": False,
        }
    ]


def test_sanitize_emotion_mapping_rejects_bad_paths_and_persistent_motions():
    mapping = sanitize_emotion_mapping(
        {
            "motions": {
                "happy": ["motions/happy.motion3.json", "../bad.motion3.json"],
                "常驻": ["motions/idle.motion3.json"],
            },
            "expressions": {
                "常驻": ["expressions/idle.exp3.json", "/bad.exp3.json"],
            },
            "hotkeys": {"h": "happy"},
        }
    )

    assert mapping["motions"] == {"happy": ["motions/happy.motion3.json"]}
    assert mapping["expressions"] == {"常驻": ["expressions/idle.exp3.json"]}
    assert mapping["hotkeys"] == {"h": "happy"}


def test_overlay_save_and_delete(tmp_path):
    config_mgr = SimpleNamespace(config_dir=tmp_path)
    context = SimpleNamespace(
        model_identity="documents:yui:yui0403yuanpi .model3.json",
        requested_name="yui",
        source="documents",
        item_id="",
        model_config_file="yui0403yuanpi .model3.json",
        fingerprint="abc",
    )

    changed = save_overlay_mapping(
        config_mgr,
        context,
        {"motions": {}, "expressions": {"happy": ["expressions/by.exp3.json"]}, "hotkeys": {}},
    )
    overlay = load_emotion_overlay(config_mgr)

    assert changed is True
    assert context.model_identity in overlay["models"]

    changed = save_overlay_mapping(config_mgr, context, {"motions": {}, "expressions": {}, "hotkeys": {}})
    overlay = load_emotion_overlay(config_mgr)

    assert changed is True
    assert context.model_identity not in overlay["models"]


def test_corrupt_overlay_returns_empty_and_logs_warning(tmp_path, caplog):
    config_mgr = SimpleNamespace(config_dir=tmp_path)
    (tmp_path / "live2d_emotion_overrides.json").write_text("{broken", encoding="utf-8")

    caplog.set_level("WARNING", logger="utils.live2d_resources")
    overlay = load_emotion_overlay(config_mgr)

    assert overlay == {"version": 1, "models": {}}
    assert "Failed to read Live2D emotion overlay" in caplog.text


def test_resolve_live2d_context_rejects_unsafe_item_id():
    try:
        resolve_live2d_context(model_name="yui", item_id="../bad")
    except FileNotFoundError as e:
        assert "invalid item id" in str(e)
    else:
        raise AssertionError("unsafe item id should be rejected")
