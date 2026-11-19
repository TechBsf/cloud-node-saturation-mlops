from pathlib import Path


def test_project_structure():
    assert Path("Dockerfile").exists()
    assert Path("requirements.txt").exists()
    assert Path("scripts").exists()
