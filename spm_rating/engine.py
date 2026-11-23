"""Completed model inference with packaged numerical components."""
from dataclasses import replace
from types import SimpleNamespace,FunctionType
from functools import lru_cache
import numpy as np
from . import role,geometry,hand,hand_cost,recovery
from ._core import batch,model,parser

def mirror(notes):
    col=6-notes.col;order=np.lexsort((col,notes.time));col=col[order]
    return replace(notes,time=notes.time[order],col=col,ln_end=notes.ln_end[order],is_ln=notes.is_ln[order],col_idx=tuple(np.flatnonzero(col==k) for k in range(7)))

class Engine:
    M=model
    parser=parser
    batch=batch
    def build(self,notes,p):
        original=batch.build_batch
        builder=FunctionType(original.__code__,{**original.__globals__,'parse_file':lambda path,rate:notes},original.__name__,original.__defaults__)
        builder.__kwdefaults__=original.__kwdefaults__
        data=builder([SimpleNamespace(id='chart',path='chart')],parse_cache=False)
        return model.build_struct(data,p)
    def _components(self,notes,p,role_price):
        other=mirror(notes)
        if (other.col.tobytes(),other.ln_end.tobytes())<(notes.col.tobytes(),notes.ln_end.tobytes()):notes,other=other,notes
        sources=[]
        for nt in [notes,other]:
            local={**p,'x_jump_w':0.}
            s=self.build(nt,local)
            cur=geometry.compute_curves(s.batch,s,{**local,'d_rec':0.})
            cur={**cur,'Recov':recovery.curve(nt.time,s.t,local['recov_w'],local['recov_wl'])}
            data=hand.extract(nt,30.)
            rates=hand.field(data,s.t,float(local['w_jm']))
            state=hand_cost.state(rates)
            unit=hand_cost.price(state,s.x[0],p)
            sources.append((nt,s,cur,state,unit))
        reference=sources[0][1]
        extra=role_price*role.field(notes,p,reference.t)
        release=.5*(sources[0][2]['Rbar']+sources[1][2]['Rbar'])
        return sources,reference,extra,release
    def predict(self,notes,p,role_price,*,with_zero=False):
        sources,reference,extra,release=self._components(notes,p,role_price)
        def combine(s,cur,unit):return model.combine(s,{**cur,'Rbar':release,'Jm_context':cur['Jm'],'Jm':cur['Jm']+unit},p)
        parts=[combine(s,cur,unit+extra) for _,s,cur,_,unit in sources]
        field=.5*(parts[0]+parts[1]);raw=float(model.postprocess(reference,model.aggregate(reference,field,p,hold_w=True),p)[0])
        result=dict(raw_score=raw,query=reference.t,field=field,struct=reference)
        if with_zero:
            zero=[combine(s,cur,unit) for _,s,cur,_,unit in sources];zero=.5*(zero[0]+zero[1])
            result.update(zero_field=zero,raw_zero=float(model.postprocess(reference,model.aggregate(reference,zero,p,hold_w=True),p)[0]))
        return result
@lru_cache(maxsize=1)
def get():return Engine()
