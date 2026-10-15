from tests.helpers import normalise


def test_normalise_replaces_absolute_path():
    assert normalise({"model_path": "/tmp/edgemason/model.onnx"}) == {"model_path": "<PATH>"}


def test_normalise_replaces_uuid():
    assert normalise({"id": "a3f8b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6"}) == {"id": "<UUID>"}


def test_normalise_replaces_system_info():
    assert normalise({"system_info": {"os": "Linux", "cpu": "x86_64"}}) == {
        "system_info": "<SYSTEM_INFO>"
    }


def test_normalise_replaces_version():
    assert normalise("onnxruntime==1.27.0") == "<VERSION>"


def test_normalise_replaces_timestamp():
    assert normalise("2026-10-14T23:05:12.123456") == "<TIMESTAMP>"


def test_normalise_nested_list_replaces_all_and_does_not_mutate():
    original = [
        {"path": "/tmp/model.onnx", "id": "a3f8b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6"},
        {
            "system_info": {"os": "Linux"},
            "version": "onnx==1.16.0",
            "time": "2026-10-14T23:05:12.123456",
        },
    ]
    expected = [
        {"path": "<PATH>", "id": "<UUID>"},
        {
            "system_info": "<SYSTEM_INFO>",
            "version": "<VERSION>",
            "time": "<TIMESTAMP>",
        },
    ]

    result = normalise(original)

    assert result == expected
    assert original[0]["path"] == "/tmp/model.onnx"
    assert original[1]["version"] == "onnx==1.16.0"