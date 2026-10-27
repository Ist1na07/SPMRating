"""One task-conditioned finger-role resource, without a fitting dependency."""
from dataclasses import replace
import numpy as np
from numba import njit

def canonical(notes):
    order=np.lexsort((notes.col,notes.time));t=notes.time[order];c=notes.col[order];e=notes.ln_end[order];ln=notes.is_ln[order]
    duplicate=np.r_[False,(np.diff(t)==0)&(np.diff(c)==0)]
    if np.any(e[duplicate]!=e[np.maximum(np.flatnonzero(duplicate)-1,0)]):raise ValueError('Conflicting simultaneous objects on one physical key.')
    keep=~duplicate;t=t[keep];c=c[keep];e=e[keep];ln=ln[keep]
    return replace(notes,time=t,col=c,ln_end=e,is_ln=ln,n_notes=len(t),col_idx=tuple(np.flatnonzero(c==k) for k in range(7)))

def episodes(notes):
    result=[]
    for lane in range(7):
        take=np.flatnonzero(notes.is_ln&(notes.col==lane));order=take[np.argsort(notes.time[take],kind='stable')];spans=[]
        for i in order:
            h=float(notes.time[i]);e=float(notes.ln_end[i])
            if e<=h:continue
            if spans and h<spans[-1][1]:spans[-1]=(spans[-1][0],max(e,spans[-1][1]))
            else:spans.append((h,e))
        result.extend((lane,h,e) for h,e in spans)
    return result

def interval_mean(times,values,query,width):
    if len(times)<2:return np.zeros(len(query))
    cumulative=np.r_[0.,np.cumsum(values*np.diff(times))]
    def integral(q):
        index=np.clip(np.searchsorted(times,q,side='right')-1,0,len(times)-2);result=cumulative[index]+values[index]*(q-times[index])
        return np.where(q<=times[0],0.,np.where(q>=times[-1],cumulative[-1],result))
    return np.maximum((integral(query+width/2.)-integral(query-width/2.))/width,0.)

@njit(cache=True)
def pulse_max(time,col,weight,query,radius):
    result=np.zeros(len(query));acc=np.empty(7)
    for i in range(len(query)):
        acc[:]=0.;q=query[i];lo=np.searchsorted(time,q-radius);hi=np.searchsorted(time,q+radius,side='right')
        for j in range(lo,hi):
            delta=q-time[j]
            factor=1000./radius*np.cos(np.pi*delta/(2.*radius))**2 if abs(delta)<radius else 0.
            acc[col[j]]+=weight[j]*factor
        result[i]=np.max(acc)
    return result

def field(notes,p,query):
    query=np.asarray(query,float)
    if not np.any(notes.is_ln):return np.zeros(len(query))
    nt=canonical(notes);time=nt.time;col=nt.col;width=float(p['w_jm']);spans=episodes(nt);duty=np.zeros(len(time))
    for lane in range(7):
        own=[(h,e) for c,h,e in spans if c==lane]
        if not own:continue
        h,e=np.asarray(own).T;events=np.r_[h,e];changes=np.r_[np.ones(len(h)),-np.ones(len(e))];order=np.argsort(events,kind='stable');events=events[order];changes=changes[order]
        starts,ix=np.unique(events,return_index=True);values=np.cumsum(np.add.reduceat(changes,ix))[:-1];take=np.flatnonzero(col==lane)
        duty[take]=interval_mean(starts,values,time[take]-width/2.,width)
    duty=np.clip(duty,0.,1.);gap=np.full(len(time),width/1000.)
    for lane in range(7):
        ix=np.flatnonzero(col==lane);g=np.maximum(np.diff(time[ix])/1000.,.0001)
        gap[ix[:-1]]=np.minimum(gap[ix[:-1]],g);gap[ix[1:]]=np.minimum(gap[ix[1:]],g)
    q=(64.5-float(np.ceil(3.0*nt.od)))/500.;x=.3*np.sqrt(max(q,1e-6));precision=float(min(x,.6*(x-.09)+.09));k=p['j_c1']*max(precision**p['od_mult_jm'],1e-6)**.25
    return pulse_max(time,col,(1./(gap+k))*duty**2,query,width/2.)
