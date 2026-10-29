# SPDX-License-Identifier: Apache-2.0
"""Development-only fixtures. Never imported by the production runtime."""
from pathlib import Path
import json
import pandas as pd
import polars as pl
import pyarrow as pa

assert pd.__version__ == "3.1.0"
assert pl.__version__ == "2.1.0"
assert pa.__version__ == "23.0.1"
root = Path(__file__).parent
frame = pl.DataFrame({
    "signed": pl.Series([-(2**63), None, 2**63-1], dtype=pl.Int64),
    "unsigned": pl.Series([0, None, 2**64-1], dtype=pl.UInt64),
    "small": pl.Series([-128, None, 127], dtype=pl.Int8),
    "real": pl.Series([1.5, None, float("nan")], dtype=pl.Float32),
    "flag": pl.Series([True, None, False], dtype=pl.Boolean),
})
frame.write_ipc(root / "polars.arrow", compression="uncompressed")
frame.write_parquet(root / "polars.parquet", compression="snappy")
frame.write_ipc(root / "compressed.arrow", compression="zstd")
pl.DataFrame({"text": ["a",None,"b"]}).write_ipc(root / "strings.arrow", compression="uncompressed")
pd.DataFrame({"signed": pd.Series([-(2**63),None,2**63-1],dtype="Int64")}).to_parquet(root / "pandas.parquet",engine="pyarrow",compression="snappy")
(root / "manifest.json").write_text(json.dumps({"versions":{"pandas":pd.__version__,"polars":pl.__version__,"pyarrow":pa.__version__},"license":"Apache-2.0","source":"Synthetic values authored in generate.py","expected":{"rows":3,"signed":[str(-(2**63)),None,str(2**63-1)],"unsigned":["0",None,str(2**64-1)]},"explicitErrors":["pandas schema/index metadata", "string dtype", "compressed IPC"]}, indent=2)+"\n")
