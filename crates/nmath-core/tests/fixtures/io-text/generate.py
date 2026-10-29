# SPDX-License-Identifier: Apache-2.0
"""Synthetic numeric text fixtures from the exact development-only originals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl
assert (np.__version__, pd.__version__, pl.__version__) == ('2.5.4', '3.1.0', '2.1.0')
out = Path(__file__).parent
signed = [-9223372036854775808, None, 9223372036854775807]
unsigned = [18446744073709551615, 0, None]
frame = pd.DataFrame({'signed': pd.array(signed,dtype='Int64'), 'unsigned': pd.array(unsigned,dtype='UInt64'), 'small': pd.array([1,None,-2],dtype='Int8'), 'flag': pd.array([True,None,False],dtype='boolean')})
frame.to_csv(out / 'pandas.csv', index=False)
frame.to_json(out / 'pandas.json', orient='records')
pframe = pl.DataFrame({'signed':pl.Series(signed,dtype=pl.Int64),'unsigned':pl.Series(unsigned,dtype=pl.UInt64),'small':pl.Series([1,None,-2],dtype=pl.Int8),'flag':pl.Series([True,None,False],dtype=pl.Boolean)})
pframe.write_csv(out / 'polars.csv')
pframe.write_json(out / 'polars.json')
(out / 'manifest.json').write_text(json.dumps({'versions':{'numpy':np.__version__,'pandas':pd.__version__,'polars':pl.__version__},'schema':[['signed','int64'],['unsigned','uint64'],['small','int8'],['flag','bool']], 'values':[[str(v) if v is not None else None for v in signed],[str(v) if v is not None else None for v in unsigned],[1,None,-2],[True,None,False]]},indent=2)+'\n')
# Upstream readers are also checked with explicit nullable schemas.
for name in ['pandas','polars']:
    p = out / f'{name}.csv'
    got = pd.read_csv(p,dtype={'signed':'Int64','unsigned':'UInt64','small':'Int8','flag':'boolean'})
    assert got['signed'].tolist() == frame['signed'].tolist()
    got = pl.read_csv(p,schema_overrides={'signed':pl.Int64,'unsigned':pl.UInt64,'small':pl.Int8,'flag':pl.Boolean})
    assert got['unsigned'].to_list() == unsigned
print('Exact-version pandas/Polars numeric CSV and JSON fixtures generated.')
