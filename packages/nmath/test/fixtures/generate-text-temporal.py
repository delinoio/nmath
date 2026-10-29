# SPDX-License-Identifier: Apache-2.0
"""Exact NumPy 2.5.4 string/temporal cases; Python is an oracle only."""
import base64
import hashlib
import io
import json
from pathlib import Path
import warnings
import numpy as np
assert np.__version__ == '2.5.4'
warnings.simplefilter('ignore')
np.seterr(all='ignore')

def record(a):
    a = np.asarray(a)
    if a.dtype.kind in 'Mm':
        unit, step = np.datetime_data(a.dtype)
        return dict(dtype='datetime64' if a.dtype.kind == 'M' else 'timedelta64', unit=(str(step) if step != 1 else '') + unit, shape=list(a.shape), values=[str(x) for x in a.view('i8').ravel()])
    if a.dtype.kind == 'U':
        return dict(dtype='str', stringWidth=a.itemsize//4, shape=list(a.shape), values=a.ravel().tolist())
    return dict(dtype=str(a.dtype), shape=list(a.shape), values=[str(x) if a.dtype.kind in 'iu' else ('NaN' if np.isnan(x) else float(x)) for x in a.ravel()])

cases=[]
def case(op,a,b=None,**extra):
    c=dict(op=op,a=record(a),**extra)
    if b is not None: c['b']=record(b)
    try:
        if op=='cast': result=a.astype(extra['target'])
        else:
            f={'sub':np.subtract,'mul':np.multiply,'div':np.divide,'mod':np.remainder,'neg':np.negative,'pos':np.positive}.get(op,getattr(np,op,None))
            result=f(a,b) if b is not None else f(a,axis=extra.get('axis')) if op in ['sum','mean','min','max'] else f(a)
        c['result']=record(result)
    except (TypeError,ValueError,OverflowError) as e: c['error']=type(e).__name__
    cases.append(c)

s=np.array([['한글','😀','a\0b'],['','é','𝄞']],dtype='U3')
case('add',s,np.array(['!','🦊','z'],dtype='U2'))
case('cast',s,target='U1')
for op in ['sub','mul','div','mod']: case(op,s,s)
for op in ['neg','pos','sum','mean','min','max']: case(op,s)
units=['Y','M','W','D','h','m','s','ms','us','ns','ps','fs','as','3s','2ms']
for kind in ['datetime64','timedelta64']:
    for unit in units:
        a=np.array([-13,-1,0,1,13,-9223372036854775808],dtype=f'{kind}[{unit}]')
        for target in ['Y','M','D','s','ms','ns','as']:
            case('cast',a,target=f'{kind}[{target}]')
    a=np.array([[-3,4,6],[1,2,-9223372036854775808]],dtype=f'{kind}[ms]')
    for op in ['sum','mean','min','max']:
        for axis in [None,0,-1]: case(op,a,axis=axis)
    for op in ['pos','neg']: case(op,a)
    for op in ['add','sub','mul','div','mod']:
        case(op,a,np.array([1,2,-1],dtype='timedelta64[s]'))
    for v in [9223372036854775807,-9223372036854775807,10000000000]:
        x=np.array([v],dtype=f'{kind}[s]')
        case('cast',x,target=f'{kind}[ns]')
        case('add',x,np.array([v],dtype='timedelta64[s]'))
        case('cast',np.array([v],dtype=f'{kind}[ns]'),target=f'{kind}[ms]')
    case('sub',a,a)
for op in ['mul','div']:
    case(op,np.array([-5,4,-9223372036854775808],dtype='timedelta64[3s]'),np.array([2,-2,0],dtype='int64'))
for op in ['sum','mean','min','max']:
    case(op,np.array([],dtype='timedelta64[ns]'))
    case(op,np.array([9223372036854775807,1],dtype='timedelta64[ns]'))

# The pinned temporal loop reinterprets uint64 bits as signed int64. Cover
# representable results, negative reinterpretations, NaT and result overflow.
for integer in [2**63, 2**63 + 1, 2**64 - 1]:
    for ticks in [0, 10, -10, 1, -1, -2**63]:
        for op in ['mul', 'div']:
            case(op, np.array([ticks], dtype='timedelta64[ns]'),
                 np.array([integer], dtype='uint64'))
        case('mul', np.array([integer], dtype='uint64'),
             np.array([ticks], dtype='timedelta64[ns]'))

files=[]
for a in [s,np.array([[-1,2],[9007199254740993,-9223372036854775808]],dtype='datetime64[ns]'),np.array([[-3,4],[0,-9223372036854775808]],dtype='timedelta64[3s]')]:
    for variant in [a,a.astype(a.dtype.newbyteorder('>')),np.asfortranarray(a)]:
        stream=io.BytesIO();np.save(stream,variant)
        files.append(dict(npy=base64.b64encode(stream.getvalue()).decode(),result=record(a)))
source=Path('/tmp/nmath-pinned-sources/numpy/numpy/_core/src/multiarray')
result=dict(numpy=np.__version__,criterion='exact strings, dtype metadata, signed integer ticks, shape, NaT and exception category; NaN by classification',sourceHashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'_datetime.h',source/'datetime.c']},cases=cases,files=files)
Path(__file__).with_name('text-temporal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
