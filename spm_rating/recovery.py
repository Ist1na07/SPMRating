"""Continuous head-count recovery context, independent of query mesh.

C(t)=number of actual heads in a centred1000ms box. Short/long means
are exact trapezoid convolutions. The positive burst ratio is piecewise
linear-over-linear; its outer mean uses a positive endpoint log primitive.
This specifies zero event measure outside the chart, rather than holding
the last sampled rate until another game object appears. Not a rating.
"""
import numpy as np

def mean_rate(heads,query,width):
    h=np.sort(np.asarray(heads,dtype=float));q=np.asarray(query,dtype=float)
    if width<=0 or not np.isfinite(width):raise ValueError('Positive mean width required.')
    if not len(h):return np.zeros_like(q)
    origin=h[0];h=h-origin;q=q-origin;prefix=np.r_[0.,np.cumsum(h)]
    outer=(1000.+width)/2.;inner=abs(1000.-width)/2.;plateau=min(1000.,width)/width
    lo=np.searchsorted(h,q-outer,side='left');a=np.searchsorted(h,q-inner,side='left');b=np.searchsorted(h,q+inner,side='right');hi=np.searchsorted(h,q+outer,side='right')
    left=(prefix[a]-prefix[lo])-(a-lo)*(q-outer);right=(hi-b)*(q+outer)-(prefix[hi]-prefix[b])
    return np.maximum((b-a)*plateau+(left+right)/width,0.)

def ratio_area(a0,a1,b0,b1,length):
    """Integral of positive linear numerator / positive linear denominator.

    Positive endpoint weights avoid cancellation of signed antiderivatives.
    Small slope uses the corresponding power series, not a0/den shortcuts.
    """
    a0,a1,b0,b1,length=np.broadcast_arrays(a0,a1,b0,b1,length);d=(b1-b0)/b0;small=abs(d)<1e-4;w0=np.zeros_like(d);w1=np.zeros_like(d)
    z=-d[small];u0=np.zeros_like(z);u1=np.zeros_like(z);power=np.ones_like(z)
    for k in range(7):u0+=power/((k+1.)*(k+2.));u1+=power/(k+2.);power*=z
    w0[small]=u0;w1[small]=u1;z=d[~small];log=np.log1p(z)
    w0[~small]=((1.+z)*log-z)/(z*z);w1[~small]=(z-log)/(z*z)
    return np.maximum(length*(np.maximum(a0,0.)*np.maximum(w0,0.)+np.maximum(a1,0.)*np.maximum(w1,0.))/b0,0.)

def roots(time,y):
    cross=(y[:-1]<0.)&(y[1:]>0.) | (y[:-1]>0.)&(y[1:]<0.)
    i=np.flatnonzero(cross);return time[i]-y[i]*(time[i+1]-time[i])/(y[i+1]-y[i])

def prepare(heads,short_width,long_width,floor=1e-6):
    h=np.sort(np.asarray(heads,dtype=float));w=max(float(short_width),50.);wl=max(float(long_width),1000.)
    if not len(h) or np.any(~np.isfinite(h)) or not np.isfinite(w) or not np.isfinite(wl):raise ValueError('Finite nonempty heads and widths required.')
    unique=np.unique(h);knots=[]
    for width in [w,wl]:
        a=(1000.+width)/2.;b=abs(1000.-width)/2.
        for shift in [-a,-b,b,a]:knots.append(unique+shift)
    t=np.unique(np.concatenate(knots));l=mean_rate(h,t,wl);t=np.unique(np.r_[t,roots(t,l-floor)])
    s=mean_rate(h,t,w);l=np.maximum(mean_rate(h,t,wl),floor);t=np.unique(np.r_[t,roots(t,s-l)])
    s=mean_rate(h,t,w);l=np.maximum(mean_rate(h,t,wl),floor);a=np.maximum(s-l,0.);area=ratio_area(a[:-1],a[1:],l[:-1],l[1:],np.diff(t));prefix=np.r_[0.,np.cumsum(area)]
    return dict(time=t,numerator=a,denominator=l,prefix=prefix,width=w,long_width=wl)

def integral(model,query):
    q=np.asarray(query,dtype=float);t=model['time'];i=np.searchsorted(t,q,side='right')-1;before=i<0;after=i>=len(t)-1;i=np.clip(i,0,len(t)-2)
    length=np.clip(q-t[i],0.,t[i+1]-t[i]);fraction=length/(t[i+1]-t[i]);a=model['numerator'];b=model['denominator'];aq=a[i]+fraction*(a[i+1]-a[i]);bq=b[i]+fraction*(b[i+1]-b[i])
    value=model['prefix'][i]+ratio_area(a[i],aq,b[i],bq,length)
    return np.where(before,0.,np.where(after,model['prefix'][-1],value))

def interval_mass(model,start,end):
    """Two positive partial cells plus a monotone full-cell prefix gap."""
    lo,hi=np.broadcast_arrays(start,end);t=model['time'];lo=np.maximum(lo,t[0]);hi=np.minimum(hi,t[-1]);empty=hi<=lo
    i=np.clip(np.searchsorted(t,lo,side='right')-1,0,len(t)-2);j=np.clip(np.searchsorted(t,hi,side='right')-1,0,len(t)-2);same=i==j
    a=model['numerator'];b=model['denominator']
    def at(index,q):
        f=np.clip((q-t[index])/(t[index+1]-t[index]),0.,1.);return (1.-f)*a[index]+f*a[index+1],(1.-f)*b[index]+f*b[index+1]
    al,bl=at(i,lo);ah,bh=at(j,hi)
    one=ratio_area(al,ah,bl,bh,np.maximum(hi-lo,0.))
    left=ratio_area(al,a[i+1],bl,b[i+1],np.maximum(t[i+1]-lo,0.));right=ratio_area(a[j],ah,b[j],bh,np.maximum(hi-t[j],0.));middle=np.maximum(model['prefix'][j]-model['prefix'][i+1],0.)
    return np.where(empty,0.,np.where(same,one,left+middle+right))

def curve(heads,query,short_width,long_width):
    q=np.asarray(query,dtype=float);m=prepare(heads,short_width,long_width);w=m['width'];out=interval_mass(m,q-w/2.,q+w/2.)/w
    if np.any(out<-1e-10):raise ValueError('Positive recovery mass invariant violated.')
    return np.maximum(out,0.)
