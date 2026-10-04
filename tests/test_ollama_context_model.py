from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_16k_modelfile_exists_and_sets_context() -> None:
    modelfile = ROOT / "ollama" / "Modelfile.qwen3.5-9b-16k"
    text = modelfile.read_text(encoding="utf-8")
    assert "FROM qwen3.5:9b" in text
    assert "PARAMETER num_ctx 16384" in text


def test_windows_setup_script_updates_both_model_settings() -> None:
    script = (ROOT / "scripts" / "setup_ollama_qwen16k.ps1").read_text(encoding="utf-8")
    assert "qwen3.5-9b-16k" in script
    assert "OPENAI_MODEL" in script
    assert "GAME_VISION_MODEL" in script


def test_env_example_uses_16k_model() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_MODEL=qwen3.5-9b-16k" in text
    assert "GAME_VISION_MODEL=qwen3.5-9b-16k" in text
