# SPDX-License-Identifier: Apache-2.0
# Fixture regeneration only; Python is not needed to build or run tests.
from pathlib import Path
import numpy as np
assert np.__version__ == '2.5.4'
rng=np.random.default_rng(23019)
edge=[0,0x80000000,1,2,3,0x7fffff,0x800000,0x800001,0x1000000,0x3f000000,0x3f7fffff,0x3f800000,0x3f800001,0x40000000,0x4b800000,0x7f7fffff,0x7f800000,0x7fc00000]
edge += [x | 0x80000000 for x in edge[2:]]
pairs=[(a,b) for a in edge for b in edge]
pairs += [(int(a),int(b)) for a,b in rng.integers(0,2**32,size=(1024,2),dtype=np.uint32)]
a=np.array([p[0] for p in pairs],dtype=np.uint32).view(np.float32)
b=np.array([p[1] for p in pairs],dtype=np.uint32).view(np.float32)
with np.errstate(all='ignore'):
    outs=[f(a,b).view(np.uint32) for f in [np.add,np.subtract,np.multiply,np.divide,np.remainder]]
with open(Path(__file__).with_name('float32_edges.rs'), 'w') as f:
    f.write('// Generated with NumPy 2.5.4; seed 23019. Exact binary32 words.\n// Columns: lhs, rhs, add, sub, mul, div, remainder. NaN payloads are not compared.\n')
    f.write('pub const CASES: &[[u32; 7]] = &[\n')
    for i,(x,y) in enumerate(pairs):
        f.write('    ['+', '.join(f'0x{int(w):08x}' for w in [x,y]+[o[i] for o in outs])+'],\n')
    f.write('];\n')
print(np.__version__,len(pairs),'pairs')
