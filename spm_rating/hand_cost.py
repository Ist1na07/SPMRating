"""Price an additional common-hand task, rather than marginal total pressure.

Two distinct priors: f(H)-f(I) versus f((H-I)+). The first charges the
cross-interaction with the already-present finger clock. The second treats
the unexplained hand clock as its own task. Both share the Sunny f.
"""
import numpy as np


def state(rates):
    return dict(finger_rate=np.array([[h['independent_max'] for h in p] for p in rates['profiles']]),
                hand_rate=np.array([[h['hand_rate'] for h in p] for p in rates['profiles']]))


def price(values,x,p,mode='extra_task'):
    i=values['finger_rate'];h=values['hand_rate'];k=p['j_c1']*max(float(x)**p['od_mult_jm'],1e-6)**.25
    def cost(r):return r*r/(1.+k*r)
    if mode=='extra_task':per_hand=cost(np.maximum(h-i,0.))
    elif mode=='marginal_total':per_hand=np.maximum(cost(h)-cost(i),0.)
    else:raise ValueError('Choose extra_task or marginal_total.')
    profiles=np.maximum(per_hand[:,0],per_hand[:,1])
    return np.sqrt(np.mean(np.square(profiles),axis=0))
