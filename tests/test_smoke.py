import subprocess
import sys
import unittest
from pathlib import Path


class PackageSmokeTest(unittest.TestCase):
    def test_local_code_package_imports_from_release_root(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from code.common.config import SystemConfig; print(SystemConfig.__name__)",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "SystemConfig")


class CliSmokeTest(unittest.TestCase):
    def test_every_public_script_has_working_help(self):
        root = Path(__file__).resolve().parents[1]
        names = (
            "run_demo.py",
            "reproduce_fig2.py",
            "reproduce_fig3.py",
            "reproduce_fig4.py",
            "reproduce_fig5.py",
            "reproduce_fig6.py",
        )
        for name in names:
            with self.subTest(name=name):
                proc = subprocess.run(
                    [sys.executable, str(root / "scripts" / name), "--help"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_plot_data_returns_nonzero(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "scripts/reproduce_fig5.py", "--data", "missing.json"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("missing.json", proc.stderr)


if __name__ == "__main__":
    unittest.main()
