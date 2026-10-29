# SPDX-License-Identifier: Apache-2.0
"""Hardware identity fixtures for the development-only original benchmark."""
import unittest
from unittest.mock import patch
import originals


class CpuIdentityTests(unittest.TestCase):
    def test_linux_uses_model_instead_of_architecture(self):
        with patch.object(originals.sys, "platform", "linux"), patch.object(originals.platform, "machine", return_value="x86_64"), patch.object(originals.platform, "processor", return_value="x86_64"), patch.object(originals.Path, "read_text", return_value="processor : 0\nmodel name : AMD EPYC 9V74\n"):
            result = originals.cpu_identity()
        self.assertEqual(result["cpuModel"], "AMD EPYC 9V74")
        self.assertEqual(result["cpuModelSource"], "/proc/cpuinfo:model name")
        self.assertEqual(result["cpuModelStatus"], "identified")

    def test_linux_unavailable_does_not_claim_architecture_as_model(self):
        for content in ("processor : 0\n", "model name : x86_64\n", "model name : \n"):
            with self.subTest(content=content), patch.object(originals.sys, "platform", "linux"), patch.object(originals.platform, "machine", return_value="x86_64"), patch.object(originals.Path, "read_text", return_value=content):
                result = originals.cpu_identity()
                self.assertIsNone(result["cpuModel"])
                self.assertEqual(result["cpuModelStatus"], "unavailable")
        with patch.object(originals.sys, "platform", "linux"), patch.object(originals.Path, "read_text", side_effect=PermissionError()):
            self.assertEqual(originals.cpu_identity()["cpuModelStatus"], "unavailable")

    def test_arm_hardware_field_and_macos_sysctl(self):
        with patch.object(originals.sys, "platform", "linux"), patch.object(originals.platform, "machine", return_value="aarch64"), patch.object(originals.Path, "read_text", return_value="Hardware : Example ARM CPU\n"):
            self.assertEqual(originals.cpu_identity()["cpuModel"], "Example ARM CPU")
        with patch.object(originals.sys, "platform", "darwin"), patch.object(originals.subprocess, "check_output", return_value="Apple M5 Max\n"):
            self.assertEqual(originals.cpu_identity()["cpuModel"], "Apple M5 Max")


if __name__ == "__main__":
    unittest.main()
