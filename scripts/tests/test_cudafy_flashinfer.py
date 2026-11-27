"""无需 GPU 或网络的 CUDA 配置还原回归测试。"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

SCRIPT = Path(__file__).resolve().parents[1] / "cudafy_flashinfer.py"
spec = importlib.util.spec_from_file_location("cudafy_flashinfer", SCRIPT)
cudafy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cudafy)


class RestoreConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.upstream = Path(self.temp.name) / "upstream"
        self.upstream.mkdir()
        self.files = {
            "version.txt": "0.7.1\n",
            ".gitmodules": (
                '[submodule "cuda-headers"]\n'
                "    path = 3rdparty/cutlass\n"
                "    url = https://github.com/NVIDIA/cutlass.git\n"
                '[submodule "not-packaged"]\n'
                "    path = 3rdparty/nixl\n"
                "    url = https://github.com/ai-dynamo/nixl.git\n"
            ),
            "pyproject.toml": (
                "[build-system]\n"
                'requires = ["apache-tvm-ffi>=0.1.11,<0.2"]\n'
                "[tool.setuptools.package-dir]\n"
                '"flashinfer.data" = "."\n'
                '"flashinfer.data.cutlass" = "3rdparty/cutlass"\n'
            ),
            "requirements.txt": "apache-tvm-ffi>=0.1.11,<0.2\n",
            "build_backend.py": 'CUDA_HEADERS = "3rdparty/cutlass"\n',
            "build_utils.py": 'CONFIG = "ci/cuda-versions.json"\n',
            "ci/cuda-versions.json": '{"dependency_policy": {}}\n',
            "flashinfer/jit/core.py": 'COMPILER = "nvcc"\n',
        }
        self.commit = "a" * 40
        self.tree = {filename: ("100644", "b" * 40) for filename in self.files}
        self.tree["3rdparty/cutlass"] = ("160000", self.commit)
        self.tree["3rdparty/nixl"] = ("160000", "c" * 40)
        self.git_calls = []
        self.checkouts = {}

    def fake_git(self, repo, *args):
        self.git_calls.append((repo, args))
        if args[0] == "show":
            return self.files[args[1].split(":", 1)[1]]
        if args[0] == "checkout":
            self.checkouts[repo] = args[-1]
        if args[0] == "rev-parse":
            return self.checkouts.get(repo, self.commit) + "\n"
        return ""

    def restore(self, version="0.7.1"):
        with (
            patch.object(cudafy, "git", side_effect=self.fake_git),
            patch.object(cudafy, "read_tree", return_value=self.tree),
            patch.object(cudafy.subprocess, "run") as run,
        ):
            cudafy.restore_config(self.source, self.upstream, "upstream-sha", version)
            return run

    def test_restores_tag_configuration_and_preserves_kernels(self):
        kernel = self.source / "include/flashinfer/custom.cuh"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("custom kernel\n")
        run = self.restore()
        for filename in cudafy.CONFIG_FILES:
            if filename in self.files:
                self.assertEqual(
                    (self.source / filename).read_text(), self.files[filename]
                )
        self.assertEqual(kernel.read_text(), "custom kernel\n")
        self.assertEqual(run.call_count, 1)
        self.assertIn(self.commit, run.call_args.args[0])
        self.assertFalse((self.source / "3rdparty/nixl").exists())

    def test_preserves_upstream_executable_mode(self):
        self.tree["flashinfer/jit/core.py"] = ("100755", "b" * 40)
        self.restore()
        self.assertEqual(
            (self.source / "flashinfer/jit/core.py").stat().st_mode & 0o777, 0o755
        )

    def test_accepts_another_version_and_its_gitlink(self):
        self.files["version.txt"] = "0.6.18\n"
        self.commit = "d" * 40
        self.tree["3rdparty/cutlass"] = ("160000", self.commit)
        run = self.restore("0.6.18")
        self.assertIn(self.commit, run.call_args.args[0])

    def test_discovers_submodule_path_and_url_from_tag(self):
        self.files["pyproject.toml"] = self.files["pyproject.toml"].replace(
            "3rdparty/cutlass", "thirdparty/cuda-headers"
        )
        self.files[".gitmodules"] = (
            self.files[".gitmodules"]
            .replace("3rdparty/cutlass", "thirdparty/cuda-headers")
            .replace("NVIDIA/cutlass.git", "example/headers.git")
        )
        self.tree["thirdparty/cuda-headers"] = self.tree.pop("3rdparty/cutlass")
        run = self.restore()
        self.assertIn("https://github.com/example/headers.git", run.call_args.args[0])
        self.assertTrue((self.source / "thirdparty/cuda-headers").is_dir())

    def test_disables_remaining_ppu_detector(self):
        detector = self.source / "flashinfer/_ppu_env.py"
        detector.parent.mkdir()
        detector.write_text("def ppu_sdk_root():\n    return 'ppu'\n")
        self.restore()
        namespace = {}
        exec(detector.read_text(), namespace)
        self.assertIsNone(namespace["ppu_sdk_root"]())

    def test_version_mismatch_leaves_source_untouched(self):
        with self.assertRaisesRegex(SystemExit, "不一致"):
            self.restore("0.6.18")
        self.assertEqual(list(self.source.iterdir()), [])

    def test_missing_required_config_fails(self):
        self.tree.pop("build_backend.py")
        with self.assertRaisesRegex(SystemExit, "缺少构建配置"):
            self.restore()

    def test_missing_gitlink_fails_before_writing(self):
        self.tree.pop("3rdparty/cutlass")
        with self.assertRaisesRegex(SystemExit, "gitlink"):
            self.restore()
        self.assertEqual(list(self.source.iterdir()), [])

    def test_existing_dependency_is_not_overwritten(self):
        dependency = self.source / "3rdparty/cutlass/local.txt"
        dependency.parent.mkdir(parents=True)
        dependency.write_text("keep me\n")
        with self.assertRaisesRegex(SystemExit, "拒绝覆盖已有子模块"):
            self.restore()
        self.assertEqual(dependency.read_text(), "keep me\n")

    def test_symlink_config_is_rejected(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("keep me\n")
        (self.source / "pyproject.toml").symlink_to(outside)
        with self.assertRaisesRegex(SystemExit, "符号链接"):
            self.restore()
        self.assertEqual(outside.read_text(), "keep me\n")

    def test_invalid_python_config_fails_before_fetch(self):
        self.files["build_backend.py"] = "this is invalid python!\n"
        with self.assertRaises(SyntaxError):
            self.restore()
        self.assertEqual(list(self.source.iterdir()), [])

    def test_missing_tag_has_no_fallback(self):
        (self.source / "version.txt").write_text("0.7.1\n")
        with (
            patch.object(cudafy, "git", return_value=""),
            patch.object(
                cudafy.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1),
            ) as run,
            self.assertRaisesRegex(SystemExit, "不回退"),
        ):
            cudafy.cudafy(self.source)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-1], "refs/tags/v0.7.1")

    def test_resolves_annotated_tag_to_commit(self):
        (self.source / "version.txt").write_text("0.7.1\n")
        with (
            patch.object(
                cudafy, "git", side_effect=["", "", "resolved-commit\n"]
            ) as git,
            patch.object(
                cudafy.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            patch.object(cudafy, "restore_config") as restore,
        ):
            cudafy.cudafy(self.source)
        self.assertEqual(git.call_args.args[-1], "FETCH_HEAD^{commit}")
        self.assertEqual(restore.call_args.args[2:], ("resolved-commit", "0.7.1"))

    def test_dirty_checkout_is_rejected(self):
        with (
            patch.object(cudafy, "git", return_value=" M include/custom.cuh\n"),
            self.assertRaisesRegex(SystemExit, "拒绝覆盖已有改动"),
        ):
            cudafy.cudafy(self.source)

    def test_invalid_version_is_rejected(self):
        (self.source / "version.txt").write_text("../main\n")
        with self.assertRaisesRegex(SystemExit, "无效"):
            cudafy.source_version(self.source)

    @unittest.skipUnless(
        os.environ.get("FLASHINFER_TEST_UPSTREAM"), "未指定本地官方 tag 仓库"
    )
    def test_real_upstream_tag_configurations(self):
        upstream = Path(os.environ["FLASHINFER_TEST_UPSTREAM"])
        for version in ("0.6.18", "0.7.1"):
            with self.subTest(version=version):
                revision = cudafy.git(
                    upstream, "rev-parse", f"v{version}^{{commit}}"
                ).strip()
                self.tree = cudafy.read_tree(upstream, revision)
                self.files = {
                    filename: cudafy.git(upstream, "show", f"{revision}:{filename}")
                    for filename in (*cudafy.CONFIG_FILES, "version.txt")
                    if filename in self.tree
                }
                self.source = Path(self.temp.name) / version
                self.source.mkdir()
                self.restore(version)
                for filename in cudafy.CONFIG_FILES:
                    if filename in self.files:
                        self.assertEqual(
                            (self.source / filename).read_text(), self.files[filename]
                        )
                for target, commit in self.checkouts.items():
                    if target.is_relative_to(self.source):
                        self.assertEqual(
                            commit,
                            self.tree[target.relative_to(self.source).as_posix()][1],
                        )

    def test_workflow_has_no_fixed_version_or_dependency(self):
        workflow = (
            SCRIPT.parent.parent / ".github/workflows/flashinfer-cudafy-upload-tmp.yaml"
        )
        content = workflow.read_text()
        self.assertNotIn("0.6.18", content)
        self.assertNotIn("apache-tvm-ffi>=", content)
        self.assertIn('project["build-system"]["requires"]', content)
        self.assertIn('wheels=("$RUNNER_TEMP/flashinfer-wheels/"*.whl)', content)
        blocks = re.findall(r"<<'PY'\n(.*?)\n            PY", content, flags=re.DOTALL)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            compile(textwrap.dedent(block), str(workflow), "exec")
        # 同时检查 YAML 中 shell block 的 heredoc 和数组语法。
        lines = content.splitlines()
        checked = 0
        for index, line in enumerate(lines):
            if line != "        run: |":
                continue
            body = []
            for following in lines[index + 1 :]:
                if following.strip() and not following.startswith("          "):
                    break
                body.append(following)
            result = subprocess.run(
                ["bash", "-n"],
                input=textwrap.dedent("\n".join(body)) + "\n",
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            checked += 1
        self.assertGreaterEqual(checked, 4)


class WheelValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = "a" * 40
        self.wheel = self.root / (
            f"flashinfer_python-0.7.1-1cudafy{self.commit[:12]}-py3-none-any.whl"
        )
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "flashinfer-python"\n'
        )
        (self.root / "version.txt").write_text("0.7.1\n")
        workflow = (
            SCRIPT.parent.parent / ".github/workflows/flashinfer-cudafy-upload-tmp.yaml"
        )
        blocks = re.findall(
            r"<<'PY'\n(.*?)\n            PY", workflow.read_text(), flags=re.DOTALL
        )
        self.assertEqual(len(blocks), 2)
        self.validator = textwrap.dedent(blocks[1])

    def validate(self, extra_files):
        # 执行工作流原有校验代码，使用真实 ZIP 结构而不模拟归档读取。
        with ZipFile(self.wheel, "w") as archive:
            archive.writestr(
                "flashinfer_python-0.7.1.dist-info/METADATA",
                "Name: flashinfer-python\nVersion: 0.7.1\n",
            )
            archive.writestr("flashinfer/_build_meta.py", self.commit)
            for name in (
                "flashinfer/data/cutlass/include/cutlass/cutlass.h",
                "flashinfer/data/csrc/sampling.cu",
                "flashinfer/data/include/flashinfer/attention/prefill.cuh",
                *extra_files,
            ):
                archive.writestr(name, "")
        return subprocess.run(
            [sys.executable, "-c", self.validator, str(self.wheel), self.commit],
            cwd=self.root,
            text=True,
            capture_output=True,
        )

    def test_allows_ptx_sources(self):
        result = self.validate(
            [
                "flashinfer/data/csrc/experimental/deepgemm_fp8_gemm/generated/"
                f"{arch}/kernel_{index}.ptx"
                for arch in ("sm_100a", "sm_103a")
                for index in range(2)
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_still_rejects_binaries_and_aot_data(self):
        names = [
            f"flashinfer/data/csrc/kernel{suffix}"
            for suffix in (
                ".a",
                ".cubin",
                ".dll",
                ".dylib",
                ".fatbin",
                ".o",
                ".pyd",
                ".so",
            )
        ]
        names.append("flashinfer/data/aot/kernel.ptx")
        for name in names:
            with self.subTest(name=name):
                result = self.validate([name])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("compiled files or AOT data", result.stderr)


if __name__ == "__main__":
    unittest.main()
