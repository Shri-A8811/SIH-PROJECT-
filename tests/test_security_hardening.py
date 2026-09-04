from pathlib import Path

from config.settings import OUTPUT_DIR
from src.tools.file_tools import FileTools
from src.tools.sandbox import CodeSandbox


def test_file_tools_reject_paths_outside_approved_roots(tmp_path):
    outside = tmp_path / "confidential.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        FileTools.read_file(str(outside))
        assert False, "read outside approved roots must be rejected"
    except PermissionError:
        pass


def test_sandbox_never_uses_host_process_when_docker_is_unavailable(monkeypatch):
    monkeypatch.setattr(CodeSandbox, "_check_docker_available", lambda _: False)
    result = CodeSandbox().execute_python_code("print('safe')")
    assert result.exit_code == -3
    assert result.sandbox_backend == "unavailable_fail_closed"


def test_output_write_stays_inside_outputs_directory():
    target = OUTPUT_DIR / "test_security_hardening.txt"
    result = FileTools.write_file(str(target), "approved output")
    assert Path(result["file_path"]).parent == OUTPUT_DIR.resolve()
