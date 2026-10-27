"""Positive action-work atoms and an exact, query-independent rate readout.

Only an interface primitive: costs are supplied by the caller. This does
not choose physical action grouping, price LN holding, or produce ratings.
Reuse the mature raised-cosine CDF, existing synchrony/window scales.
"""
import numpy as np
from .geometry import cumulative

def validate(time,cost,radius):
    t=np.asarray(time,dtype=float);c=np.asarray(cost,dtype=float)
    if t.ndim!=1 or c.shape!=t.shape or not np.isfinite(radius) or radius<=0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(c)) or np.any(c<0):raise ValueError('Finite1D event times, matching nonnegative costs, positive radius required.')
    order=np.argsort(t,kind='stable');return t[order],c[order]

def mass(time,cost,start,end,radius):
    t,c=validate(time,cost,radius);a,b=np.broadcast_arrays(start,end)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(b<a):raise ValueError('Finite ordered query intervals required.')
    if not len(t):return np.zeros_like(a,dtype=float)
    left=cumulative(t,c,np.asarray(a,dtype=float).ravel(),radius);right=cumulative(t,c,np.asarray(b,dtype=float).ravel(),radius);value=(right-left).reshape(a.shape)
    if np.any(value<-1e-8*(1.+c.sum())):raise ValueError('Positive source mass violated.')
    return np.maximum(value,0.)

def field(time,cost,query,width,radius):
    if not np.isfinite(width) or width<=0:raise ValueError('Positive finite time window required.')
    q=np.asarray(query,dtype=float);return mass(time,cost,q-width/2.,q+width/2.,radius)/(width/1000.)
