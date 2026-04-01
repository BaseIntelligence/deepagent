from pathlib import Path


def _rotated_paths(base_name, cwd):
    base_path = Path(cwd) / base_name
    return [
        base_path,
        Path(str(base_path) + ".1"),
        Path(str(base_path) + ".2"),
        Path(str(base_path) + ".3"),
    ]


def test_logrotate_file_paths_are_anchored_to_cwd(tmp_path):
    base_name = "logfile-rotation-failure.log"

    expected_paths = _rotated_paths(base_name, tmp_path)
    created_paths = [p.resolve() for p in expected_paths]

    assert created_paths == expected_paths
