"""Exact shared-phase continuous extension of a discrete millisecond model.

For U uniform on[0,1), evaluate floor(t*rate+U)/rate. Its expectation is
computed exactly between successive fractional-time breakpoints, not by
random sampling. Native integer-clock inputs call the old model unchanged.
"""
from dataclasses import replace
import math
import numpy as np

def _snap(x,tolerance):
    nearest=np.rint(x)
    return np.where(abs(x-nearest)<=tolerance,nearest,x)

def plan(notes,extra_times=()):
    rate=float(notes.rate)
    if not math.isfinite(rate) or rate<=0:raise ValueError('Positive finite clock rate required.')
    head=np.asarray(notes.time,float)*rate;tail=np.asarray(notes.ln_end,float)*rate
    extra=np.array([float(t)*rate for t in extra_times if math.isfinite(t)])
    values=np.r_[head,tail[notes.is_ln],extra]
    if not len(values) or not np.all(np.isfinite(values)):return None
    tolerance=32.*np.finfo(float).eps*max(1.,float(np.max(abs(values))))
    normalized=_snap(values,tolerance)
    fractional=normalized-np.floor(normalized)
    if not np.any(fractional):return None
    breaks=np.sort(1.-fractional[fractional>0.]);breaks=breaks[np.r_[True,np.diff(breaks)>tolerance]]
    return dict(rate=rate,head=_snap(head,tolerance),tail=_snap(tail,tolerance),tolerance=tolerance,breaks=np.r_[0.,breaks,1.])

def quantize(points,phase,p):
    return np.floor(_snap(np.asarray(points,float)*p['rate'],p['tolerance'])+phase)/p['rate']

def states(notes,p):
    for left,right in zip(p['breaks'][:-1],p['breaks'][1:]):
        if right<=left:continue
        phase=.5*(left+right);time=np.floor(p['head']+phase)/p['rate'];tail=np.where(notes.is_ln,np.floor(p['tail']+phase)/p['rate'],notes.ln_end);order=np.lexsort((notes.col,time));col=notes.col[order];time=time[order]
        child=replace(notes,time=time,col=col,ln_end=tail[order],is_ln=notes.is_ln[order],col_idx=tuple(np.flatnonzero(col==k) for k in range(7)),duration=float(time[-1]-time[0]))
        yield float(right-left),child,phase,order
