import re
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from gradio.cli import cli


LLAMA_CPP_RELEASE_URL = os.environ.get(
    "LLAMA_CPP_RELEASE_URL",
    "https://github.com/ggml-org/llama.cpp/releases/download/b10091/"
    "llama-b10091-bin-ubuntu-x64.tar.gz",
)


def _can_write(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".write-test"
        marker.write_text("", encoding="utf-8")
        marker.unlink()
        return True
    except OSError:
        return False


def _is_path_like(binary: str):
    return os.sep in binary or (os.altsep is not None and os.altsep in binary)


def _write_llama_server_wrapper(wrapper_path: Path, llama_server_dir: Path):
    wrapper = """#!/usr/bin/env sh
set -eu
LLAMA_CPP_HOME="{llama_server_dir}"
export LD_LIBRARY_PATH="${{LLAMA_CPP_HOME}}:${{LD_LIBRARY_PATH:-}}"
exec "${{LLAMA_CPP_HOME}}/llama-server" "$@"
""".format(
        llama_server_dir=llama_server_dir
    )

    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(wrapper, encoding="utf-8")
    wrapper_path.chmod(0o755)
    os.environ["LLAMA_CPP_BINARY"] = str(wrapper_path)


def _download_llama_server(install_root: Path):
    extract_dir = install_root / "release"
    archive_path = install_root / "llama.cpp.tar.gz"
    llama_server = (
        next(extract_dir.rglob("llama-server"), None) if extract_dir.exists() else None
    )

    if llama_server and llama_server.is_file():
        llama_server.chmod(llama_server.stat().st_mode | 0o755)
        return llama_server

    install_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading llama.cpp release from {LLAMA_CPP_RELEASE_URL} to {archive_path}", flush=True)
    urllib.request.urlretrieve(LLAMA_CPP_RELEASE_URL, archive_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    extract_root = extract_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target_path = (extract_root / member.name).resolve()
            try:
                target_path.relative_to(extract_root)
            except ValueError:
                raise RuntimeError(f"Unsafe path in llama.cpp archive: {member.name}")
        tar.extractall(extract_dir)

    llama_server = next(extract_dir.rglob("llama-server"), None)
    if not llama_server or not llama_server.is_file():
        raise RuntimeError("llama-server not found in downloaded llama.cpp release")
    print(f"Using {llama_server} as the inference backend.", flush=True)
    
    llama_server.chmod(llama_server.stat().st_mode | 0o755)
    return llama_server


def _ensure_llama_server():
    if os.environ.get("SURYA_INFERENCE_BACKEND") != "llamacpp":
        return

    binary = os.environ.get("LLAMA_CPP_BINARY", "llama-server")
    if shutil.which(binary):
        return

    if _is_path_like(binary):
        wrapper_path = Path(binary).expanduser()
        install_root = Path(
            os.environ.get("LLAMA_CPP_INSTALL_DIR", wrapper_path.parent / ".llama.cpp")
        )
    else:
        install_dir = Path.home() / ".local" / "bin"
        os.environ["PATH"] = f"{install_dir}:{os.environ.get('PATH', '')}"
        wrapper_path = install_dir / binary
        install_root = Path(
            os.environ.get("LLAMA_CPP_INSTALL_DIR", Path.home() / ".cache" / "llama.cpp")
        )

    if not _can_write(wrapper_path.parent):
        raise RuntimeError(f"Cannot write llama-server wrapper to {wrapper_path}")

    if not _can_write(install_root):
        install_root = Path.home() / ".cache" / "llama.cpp"
        install_root.mkdir(parents=True, exist_ok=True)

    llama_server = _download_llama_server(install_root)
    _write_llama_server_wrapper(wrapper_path, llama_server.parent)


if __name__ == '__main__':
    _ensure_llama_server()
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli())
