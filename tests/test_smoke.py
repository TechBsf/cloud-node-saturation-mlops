from pathlib import Path


def test_project_structure():
    assert Path("Dockerfile").exists()
    assert Path("requirements.txt").exists()
    assert Path("pyproject.toml").exists()
    assert Path("scripts").exists()
    assert Path("src/schema.py").exists()
