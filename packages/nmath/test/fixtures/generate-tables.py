# SPDX-License-Identifier: Apache-2.0
"""Run only with the fixed test oracles; production never imports Python."""
import json
from pathlib import Path
import pandas as pd
import polars as pl
assert pd.__version__ == '3.1.0'
assert pl.__version__ == '2.1.0'
k = [2, 1, 1, None]
v = [20, 10, None, 40]
rk = [1, 1, None]
rv = [100, 101, 400]
p = pd.DataFrame({'k': pd.array(k, dtype='Int64'), 'v': pd.array(v, dtype='Int64')})
r = pd.DataFrame({'k': pd.array(rk, dtype='Int64'), 'w': pd.array(rv, dtype='Int64')})
l = pl.DataFrame({'k': k, 'v': v}, schema={'k':pl.Int64, 'v':pl.Int64})
rp = pl.DataFrame({'k': rk, 'w': rv}, schema={'k':pl.Int64, 'w':pl.Int64})
def pandas_rows(frame):
    return [[None if pd.isna(x) else int(x) for x in row] for row in frame.itertuples(index=False, name=None)]
fixture = {'versions': {'pandas':pd.__version__,'polars':pl.__version__}, 'input': {'k':k,'v':v,'rk':rk,'rv':rv},
 'pandasJoin':pandas_rows(p.merge(r,on='k',how='left',sort=False)),
 'polarsJoin':[list(row) for row in l.join(rp,on='k',how='left',maintain_order='left').rows()],
 'pandasGroup':pandas_rows(p.groupby('k',dropna=True,sort=True,as_index=False).agg({'v':'sum'})),
 'polarsGroup':[list(row) for row in l.group_by('k',maintain_order=True).agg(pl.col('v').sum()).rows()],
 'pandasSort':pandas_rows(p.sort_values('k',kind='stable')),
 'polarsSort':[list(row) for row in l.sort('k',maintain_order=True).rows()],
 'emptyGroup':pandas_rows(p.iloc[:0].groupby('k',as_index=False).agg({'v':'sum'}))}
Path(__file__).with_name('tables.json').write_text(json.dumps(fixture,indent=2)+'\n')
dtypes=[]
for name in ['int8','int16','int32','int64','uint8','uint16','uint32','uint64','float32','float64','bool']:
    ptype={'bool':pl.Boolean}.get(name,getattr(pl,name.title().replace('Uint','UInt'),None))
    vals=[True,True] if name=='bool' else [1,1]
    pandas_frame=pd.DataFrame({'k':[1,1],'v':pd.Series(vals,dtype=name)})
    polars_frame=pl.DataFrame({'k':[1,1],'v':pl.Series(vals,dtype=ptype)})
    for op in ['sum','mean','min','max','count']:
        pandas_result=pandas_frame.groupby('k').agg({'v':op})
        polars_result=polars_frame.group_by('k').agg(getattr(pl.col('v'),op)())
        dtype=str(polars_result.schema['v']).lower().replace('boolean','bool')
        dtypes.append({'input':name,'op':op,'pandas':str(pandas_result.v.dtype),'polars':dtype,'value':float(pandas_result.v.iloc[0])})
fixture['aggregationDtypes']=dtypes
fixture['allNullGroups']={op:pandas_rows(pd.DataFrame({'k':[1,1,2],'v':pd.array([None,None,4],dtype='Int64')}).groupby('k',as_index=False).agg({'v':op})) for op in ['sum','mean','min','max','count']}
Path(__file__).with_name('tables.json').write_text(json.dumps(fixture,indent=2)+'\n')
