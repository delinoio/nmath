# SPDX-License-Identifier: Apache-2.0
"""Development-only NumPy 2.5.4 oracle. Runtime tests read JSON, not Python."""
import hashlib
import json
from pathlib import Path
import numpy as np

assert np.__version__ == '2.5.4'
root = Path('/tmp/nmath-pinned-sources/numpy')
refs = ['numpy/linalg/tests/test_linalg.py', 'numpy/fft/tests/test_pocketfft.py', 'numpy/testing/_private/utils.py']
def packed(a):
    a = np.asarray(a)
    return {'shape': list(a.shape), 'dtype': a.dtype.name,
            'values': [[float(z.real),float(z.imag)] for z in a.ravel()]}
cases=[]
for dtype in ['float32','float64','complex64','complex128']:
    complex_ = dtype.startswith('complex')
    # Exact square cases from test_linalg.py's LinalgCase definitions.
    a=np.array([[1+2j,2+3j],[3+4j,4+5j]] if complex_ else [[1.,2.],[3.,4.]],dtype=dtype)
    b=np.array([2+1j,1+2j] if complex_ else [2.,1.],dtype=dtype)
    for op in ['solve','inv','det']:
        result=getattr(np.linalg,op)(a,b) if op=='solve' else getattr(np.linalg,op)(a)
        cases.append({'family':'linalg','op':op,'a':packed(a),'b':packed(b) if op=='solve' else None,'expected':packed(result),
                      'criterion':'residual_decimal' if op!='det' else 'eigenvalue_decimal',
                      'decimal':6 if dtype in ['float32','complex64'] else 12,
                      'reference':packed(np.prod(np.linalg.eigvals(a.astype('complex128' if complex_ else 'float64')))) if op=='det' else None})
# Source test_identity_long_short: bounded random length-16 inputs, all n=1..31,
# dtype-dependent absolute bound, rtol=0. Values fixed to make generation reproducible.
rng=np.random.default_rng(20261029)
for dtype in ['float32','float64']:
    real=rng.random(16).astype(dtype)
    z=real+1j*rng.random(16).astype(dtype)
    for n in range(1,32):
        for op,x in [('fft',z),('rfft',real)]:
            y=getattr(np.fft,op)(x,n=n)
            cases.append({'family':'fft_identity','op':op,'a':packed(x),'n':n,'expected':packed(y),
                          'atol':float(5*np.spacing(np.array(1.,dtype=dtype))), 'rtol':0})
# Shape/dtype and adaptation fixtures are distinct from numerical acceptance.
for dtype in ['bool','int8','uint16','int64','float16','float32','float64','complex64','complex128']:
    for op in ['fft','ifft','rfft','irfft']:
        if op=='rfft' and dtype.startswith('complex'): continue
        x=np.array([[1,2,3],[4,5,6]],dtype=dtype)
        if dtype.startswith('complex'): x=x+np.array([[1j,-2j,3j],[-4j,5j,-6j]],dtype=dtype)
        for norm in ['backward','forward','ortho']:
            for axis in [0,-1]:
                y=getattr(np.fft,op)(x,n=5,axis=axis,norm=norm)
                cases.append({'family':'fft_adaptation','op':op,'a':packed(x),'n':5,'axis':axis,'norm':norm,'expected':packed(y),
                              'criterion':'dtype_shape_only; numerical criterion pending'})
out={'numpy':'2.5.4','source_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in refs},'cases':cases}
for family in ['linalg', 'fft_identity', 'fft_adaptation']:
    part = dict(out, cases=[c for c in cases if c['family'] == family])
    Path(__file__).with_name(family + '.json').write_text(json.dumps(part,indent=2)+'\n')
