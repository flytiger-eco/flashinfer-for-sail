#!/usr/bin/env python3
"""按源码版本对应的官方 tag 还原 CUDA wheel 的构建配置。"""

import argparse
import configparser
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

UPSTREAM_REPO = "https://github.com/flashinfer-ai/flashinfer.git"
# 只还原构建基础设施，不覆盖 fork 的算子、内核和 Python API。
CONFIG_FILES = (
    ".gitmodules",
    "pyproject.toml",
    "requirements.txt",
    "build_backend.py",
    "build_utils.py",
    ".python-version",
    "ci/cuda-versions.json",
    "flashinfer/compilation_context.py",
    "flashinfer/aot.py",
    "flashinfer/jit/core.py",
    "flashinfer/jit/cpp_ext.py",
    "flashinfer/jit/env.py",
    "flashinfer/jit/attention/modules.py",
)
REQUIRED_FILES = {
    ".gitmodules",
    "pyproject.toml",
    "requirements.txt",
    "build_backend.py",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def read_tree(repo: Path, revision: str) -> dict[str, tuple[str, str]]:
    entries = {}
    for entry in git(repo, "ls-tree", "-r", "-z", revision).split("\0"):
        if entry:
            metadata, filename = entry.split("\t", 1)
            mode, _, commit = metadata.split()
            entries[filename] = (mode, commit)
    return entries


def source_version(source: Path) -> str:
    version = (source / "version.txt").read_text().strip()
    if not re.fullmatch(r"\d[0-9A-Za-z.+-]*", version):
        raise SystemExit(f"无效的 FlashInfer 版本：{version!r}")
    return version


def safe_path(source: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SystemExit(f"不安全的上游路径：{filename!r}")
    target = source / path
    if target.resolve() != source.resolve() / path:
        raise SystemExit(f"拒绝覆盖符号链接路径：{target}")
    return target


def restore_config(source: Path, upstream: Path, revision: str, version: str) -> None:
    upstream_version = git(upstream, "show", f"{revision}:version.txt").strip()
    if upstream_version != version:
        raise SystemExit(
            f"官方 tag 的版本 {upstream_version!r} 与源码版本 {version!r} 不一致"
        )
    tree = read_tree(upstream, revision)
    missing = REQUIRED_FILES - tree.keys()
    if missing:
        raise SystemExit(f"官方 tag 缺少构建配置：{sorted(missing)}")
    converted = {}
    for filename in CONFIG_FILES:
        if filename not in tree:
            continue
        if tree[filename][0] not in {"100644", "100755"}:
            raise SystemExit(f"官方配置不是普通文本文件：{filename}")
        content = git(upstream, "show", f"{revision}:{filename}")
        if filename.endswith(".py"):
            compile(content, filename, "exec")
        converted[safe_path(source, filename)] = content

    project = tomllib.loads(converted[source / "pyproject.toml"])
    modules = configparser.ConfigParser(interpolation=None)
    modules.read_string(converted[source / ".gitmodules"])
    submodules = {
        modules[section]["path"]: modules[section]["url"]
        for section in modules.sections()
    }
    # 包目录来自官方配置；gitlink 来自同一 tag，不使用固定的 CUTLASS SHA。
    package_dirs = project["tool"]["setuptools"]["package-dir"].values()
    dependencies = []
    for filename in sorted(set(package_dirs) & submodules.keys()):
        if filename not in tree or tree[filename][0] != "160000":
            raise SystemExit(f"官方配置缺少子模块 gitlink：{filename}")
        target = safe_path(source, filename)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise SystemExit(f"需要全新源码目录，拒绝覆盖已有子模块：{target}")
        url = submodules[filename]
        # 官方仓库使用绝对 HTTPS URL；拒绝 file/SSH/相对路径等隐式来源。
        if not url.startswith("https://"):
            raise SystemExit(f"不支持的官方子模块 URL：{url!r}")
        dependencies.append((target, url, tree[filename][1]))

    # 部分 fork API 仍引用该探测器；CUDA wheel 不应因 PPU 环境变量切换工具链。
    ppu_env = safe_path(source, "flashinfer/_ppu_env.py")
    if ppu_env.exists():
        converted[ppu_env] = (
            '"""CUDA-only wheel 的工具链探测器。"""\n\n\n'
            "def ppu_sdk_root() -> None:\n"
            "    return None\n"
        )

    for target, url, commit in dependencies:
        target.mkdir(parents=True, exist_ok=True)
        git(target, "init", "--quiet")
        subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "fetch",
                "--depth",
                "1",
                "--no-tags",
                url,
                commit,
            ],
            check=True,
        )
        git(target, "checkout", "--quiet", "--detach", commit)
        if git(target, "rev-parse", "HEAD").strip() != commit:
            raise SystemExit(f"子模块提交校验失败：{target}")
        print(f"子模块：{target.relative_to(source)} {url} @ {commit}", flush=True)
    for path, content in converted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        filename = path.relative_to(source).as_posix()
        if filename in tree:
            path.chmod(int(tree[filename][0], 8) & 0o777)


def cudafy(source: Path, upstream_repo: str = UPSTREAM_REPO) -> None:
    source = source.resolve()
    if git(source, "status", "--porcelain", "--untracked-files=normal").strip():
        raise SystemExit("需要全新且干净的源码 checkout，拒绝覆盖已有改动")
    version = source_version(source)
    tag = f"v{version}"
    with tempfile.TemporaryDirectory(
        prefix="flashinfer-upstream-", dir=source.parent
    ) as temp:
        upstream = Path(temp)
        git(upstream, "init", "--quiet")
        result = subprocess.run(
            [
                "git",
                "-C",
                str(upstream),
                "fetch",
                "--depth",
                "1",
                "--no-tags",
                upstream_repo,
                f"refs/tags/{tag}",
            ]
        )
        if result.returncode:
            raise SystemExit(
                f"无法获取官方 tag {tag}（{upstream_repo}）；不回退到其他版本"
            )
        # 支持轻量和附注 tag，后续读取全部固定在解析出的 commit。
        revision = git(upstream, "rev-parse", "FETCH_HEAD^{commit}").strip()
        restore_config(source, upstream, revision, version)
    summary = f"CUDA 配置来源：{upstream_repo}，tag `{tag}`，commit `{revision}`\n"
    print(summary, end="", flush=True)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as output:
            output.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="全新且干净的源码 checkout")
    parser.add_argument("--upstream-repo", default=UPSTREAM_REPO, help="官方配置仓库")
    args = parser.parse_args()
    cudafy(args.source, args.upstream_repo)
