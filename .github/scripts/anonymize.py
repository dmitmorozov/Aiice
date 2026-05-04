from pathlib import Path

REPLACEMENTS = {
    "github.com/ITMO-NSS-team/Aiice/tree/main": "anonymous.4open.science/r/Aiice-0BF8",
    "github.com/ITMO-NSS-team": "anonymous.4open.science/r/Aiice-0BF8",
    "itmo-nss-team.github.io/Aiice": "prismatic-baklava-6691d5.netlify.app",
    "ITMO-NSS/Aiice": "anon-aiice/Aiice",
    "ITMO-NSS-team/Aiice": "anon-aiice/Aiice",

    "kdduha": "Anonymous Author",
    "just.andrew.kd@gmail.com": "anonymous@anonymous.com",
}

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".yaml", ".yml",
    ".toml", ".html", ".xml", ".js",
}

SKIP_PATHS = {
    "anonymize.py", 
    ".git",
    "__pycache__",
    "outputs",
    ".venv",
    "venv",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PATHS for part in path.parts)


def anonymize_file(path: Path) -> int:
    """Replace all sensitive strings in a single file. Returns number of replacements."""
    try:
        original = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0

    result = original
    for real, placeholder in REPLACEMENTS.items():
        result = result.replace(real, placeholder)

    count = sum(original.count(real) for real in REPLACEMENTS)
    if result != original:
        path.write_text(result, encoding="utf-8")
    return count


def anonymize_repo(root: str = ".") -> None:
    root_path = Path(root)
    total_files = 0
    total_replacements = 0

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path.relative_to(root_path)):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        replacements = anonymize_file(path)
        if replacements:
            print(f"  [+] {path}  ({replacements} replacements)")
            total_files += 1
            total_replacements += replacements

    print(f"\nDone: {total_replacements} replacements across {total_files} files.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Anonymizing: {Path(root).resolve()}\n")
    anonymize_repo(root)
