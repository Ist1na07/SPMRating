"""Enumerate the complete LN overlap support with O(rows) temporary memory.

The old pair-count safety fallback changed dtmax and therefore the model.
Here pair terms are streamed into the same two event sums as _step_scatter.
No time window, valid-pair definition, or per-pair price is approximated.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def overlap_steps(t,tg,chart,ln_end,rows,cols,matrix,dtmax,cap,jlo,jhi):
    n=len(t)
    plus=np.zeros(n+1);minus=np.zeros(n+1)
    for i in range(len(rows)):
        source=rows[i]
        # Search the sorted head list without allocating tg[rows] per source.
        lo=i+1;hi=len(rows);limit=tg[source]+dtmax
        while lo<hi:
            mid=(lo+hi)//2
            if tg[rows[mid]]<=limit:lo=mid+1
            else:hi=mid
        for j in range(i+1,lo):
            target=rows[j]
            if chart[source]!=chart[target]:continue
            ti=t[source];tj=t[target];ei=ln_end[source];ej=ln_end[target]
            if not (tj>ti and ei>tj and ej>ei):continue
            ov=min(ei-tj,cap)
            value=(ov/1000.)*matrix[cols[i],cols[j]]
            if value==0.:continue
            stop=np.searchsorted(tg,(tj+ov)+chart[target]*1073741824,side='left')
            stop=max(jlo[target],min(stop,jhi[target]));stop=max(stop,target)
            plus[target]+=value;minus[stop]+=value
    return np.cumsum(plus-minus)[:n]
