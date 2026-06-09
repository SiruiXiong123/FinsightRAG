from pathlib import Path


PROJECT_MARKERS = ("config.yaml", "config.example.yaml", ".git")


def package_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_project_root(path: Path) -> bool:
    return any((path / marker).exists() for marker in PROJECT_MARKERS)


def default_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if is_project_root(cwd):
        return cwd

    source_root = package_project_root().resolve()
    if is_project_root(source_root):
        return source_root

    return cwd
