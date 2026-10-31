from pathlib import Path


PRIVATE_OUTPUT_FILENAMES = frozenset({"metadata.json", "prompt.txt", "reference_face.png"})


def is_private_output_file(path: str | Path) -> bool:
    filename = Path(path).name.lower()
    return filename in PRIVATE_OUTPUT_FILENAMES or filename.startswith("reference_")


def list_public_output_images(outputs_dir: Path, limit: int) -> list[Path]:
    if not outputs_dir.exists():
        return []

    image_paths = [
        path
        for directory in outputs_dir.iterdir()
        if directory.is_dir()
        for path in directory.glob("*.png")
        if not is_private_output_file(path)
    ]
    image_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return image_paths[:limit]
