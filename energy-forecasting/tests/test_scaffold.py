"""Scaffold smoke test — verifies project structure and imports work."""


def test_project_structure_exists() -> None:
    """Verify that the project src package is importable."""
    import src  # noqa: F401
