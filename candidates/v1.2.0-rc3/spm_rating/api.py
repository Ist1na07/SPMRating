"""Whole-chart and context/isolated segment scoring, independent of research."""
from dataclasses import dataclass,replace
from pathlib import Path
from types import MappingProxyType
import hashlib,json,math
import numpy as np
from . import engine,role
from . import clock

@dataclass(frozen=True)
class Model:
    name:str
    parameters:object
    role_price:float
    alignment:object
    identity:str
    physical_objects:bool=True

def load_model(name):
    if name not in ['detail','compact']:raise ValueError('Unknown model name.')
    path=Path(__file__).resolve().parent/'models'/(name+'.json')
    if not path.exists():raise ValueError('Model file is missing.')
    raw=path.read_bytes();data=json.loads(raw)
    if not data.get('completed_validation',False):raise ValueError('Model validation is unfinished.')
    return Model(name,MappingProxyType(data['parameters']),float(data['role_price']),MappingProxyType(data.get('alignment',{})),hashlib.sha256(raw).hexdigest(),bool(data.get('physical_objects',True)))

def _crop(notes,start,end):
    take=(notes.time>=start)&(notes.time<end)
    if np.count_nonzero(take)<2:raise ValueError('A rated segment requires at least two physical heads; headless holding has no validated scalar output.')
    t=notes.time[take].copy();c=notes.col[take].copy();e=np.minimum(notes.ln_end[take],end);ln=notes.is_ln[take]&(e>t)
    return replace(notes,time=t,col=c,ln_end=e,is_ln=ln,n_notes=len(t),col_idx=tuple(np.flatnonzero(c==k) for k in range(7)),duration=float(t[-1]-t[0]))

def _rate_notes_grid(notes,model,*,start_ms=None,end_ms=None,scope='context',scale='raw',include_curve=False):
    if isinstance(model,str):model=load_model(model)
    if scope not in ['context','isolated']:raise ValueError('Choose context or isolated.')
    if scale not in ['raw','reference']:raise ValueError('Choose raw or reference.')
    if notes is None or notes.n_notes<2:raise ValueError('A chart requires at least two physical heads.')
    if notes.key_count not in [4,6,7]:raise ValueError('This version supports4K,6K and7K.')
    if not np.all(np.isfinite(notes.time)) or not np.all(np.isfinite(notes.ln_end)):raise ValueError('Finite note times required.')
    physical=role.canonical(notes)
    if model.physical_objects:notes=physical
    segment=start_ms is not None or end_ms is not None;start=float(start_ms) if start_ms is not None else -math.inf;end=float(end_ms) if end_ms is not None else math.inf
    if not start<end or math.isnan(start) or math.isnan(end):raise ValueError('Use a nonempty half-open interval[start,end).')
    e=engine.get();p=dict(model.parameters)
    if segment and scope=='isolated':
        rated=_crop(notes,start,end);result=e.predict(rated,p,model.role_price)
    else:
        rated=notes;result=e.predict(notes,p,model.role_price)
        if segment:
            rated=_crop(notes,start,end);s=e.build(rated,p);positions=np.searchsorted(result['query'],s.t)
            if np.any(positions>=len(result['query'])) or not np.array_equal(result['query'][positions],s.t):raise RuntimeError('Context query identity mismatch.')
            d=result['field'][positions];raw=float(e.M.postprocess(s,e.M.aggregate(s,d,p,hold_w=True),p)[0]);result={**result,'raw_score':raw,'field':d,'query':s.t}
    raw=float(result['raw_score']);aligned=None;calibration=model.alignment.get(str(notes.key_count))
    if calibration is not None:aligned=float(calibration['scale']*raw**calibration['power'])
    if scale=='reference' and aligned is None:raise ValueError('This model has no validated reference scale for this key count.')
    out=dict(model=model.name,model_identity=model.identity,score=raw if scale=='raw' else aligned,raw_score=raw,reference_score=aligned,units='raw' if scale=='raw' else calibration['domain'],key_count=notes.key_count,rate=notes.rate,heads=rated.n_notes,ln_heads=rated.n_ln,scope=scope if segment else 'whole',start_ms=start_ms,end_ms=end_ms)
    if include_curve:out.update(query=result['query'].copy(),field=result['field'].copy())
    return out

def rate_notes(notes,model,*,start_ms=None,end_ms=None,scope='context',scale='raw',include_curve=False,cancel=None):
    if cancel is not None and cancel():raise InterruptedError('计算已停止。')
    if isinstance(model,str):model=load_model(model)
    options=dict(start_ms=start_ms,end_ms=end_ms,scope=scope,scale=scale,include_curve=include_curve)
    if notes is None or notes.n_notes<2 or scope not in ['context','isolated'] or scale not in ['raw','reference']:
        return _rate_notes_grid(notes,model,**options)
    end=float(end_ms) if end_ms is not None else math.inf
    pclock=clock.plan(notes,[end] if start_ms is not None or end_ms is not None else ())
    if pclock is None:return _rate_notes_grid(notes,model,**options)
    # Preserve validation, input metadata and the observed interval contract.
    # The unextended scalar is not used as the new result.
    baseline=_rate_notes_grid(notes,model,**{**options,'scale':'raw'})
    observed=role.canonical(notes) if model.physical_objects else notes
    pclock=clock.plan(observed,[end] if start_ms is not None or end_ms is not None else ())
    if pclock is None:return _rate_notes_grid(notes,model,**options)
    start=float(start_ms) if start_ms is not None else -math.inf;segment=start_ms is not None or end_ms is not None
    selected=(observed.time>=start)&(observed.time<end);e=engine.get();parameters=dict(model.parameters);raw=0.;curve=np.zeros(len(baseline.get('query',[]))) if include_curve else None
    for weight,state,phase,order in clock.states(observed,pclock):
        if cancel is not None and cancel():raise InterruptedError('计算已停止。')
        full=role.canonical(state) if model.physical_objects else state
        if segment:
            take=selected[order];t=state.time[take];c=state.col[take];limit=float(clock.quantize([end],phase,pclock)[0]) if math.isfinite(end) else math.inf;tail=np.minimum(state.ln_end[take],limit);ln=state.is_ln[take]&(tail>t)
            child=replace(state,time=t,col=c,ln_end=tail,is_ln=ln,n_notes=len(t),col_idx=tuple(np.flatnonzero(c==k) for k in range(7)),duration=float(t[-1]-t[0]))
            if model.physical_objects:child=role.canonical(child)
            if child.n_notes<2:raise ValueError('A quantized segment requires at least two physical heads.')
            if scope=='isolated':result=e.predict(child,parameters,model.role_price)
            else:
                mother=e.predict(full,parameters,model.role_price);s=e.build(child,parameters);positions=np.searchsorted(mother['query'],s.t)
                if np.any(positions>=len(mother['query'])) or not np.array_equal(mother['query'][positions],s.t):raise RuntimeError('Context phase query identity mismatch.')
                d=mother['field'][positions];result=dict(query=s.t,field=d,raw_score=float(e.M.postprocess(s,e.M.aggregate(s,d,parameters,hold_w=True),parameters)[0]))
        else:result=e.predict(full,parameters,model.role_price)
        raw+=weight*result['raw_score']
        if include_curve:
            target=clock.quantize(baseline['query'],phase,pclock);curve+=weight*np.interp(target,result['query'],result['field'])
    calibration=model.alignment.get(str(notes.key_count));aligned=float(calibration['scale']*raw**calibration['power']) if calibration is not None else None
    if scale=='reference' and aligned is None:raise ValueError('This model has no validated reference scale for this key count.')
    out={**baseline,'raw_score':float(raw),'reference_score':aligned,'score':float(raw) if scale=='raw' else aligned,'units':'raw' if scale=='raw' else calibration['domain']}
    if include_curve:out['field']=curve
    return out

def rate_file(path,model,*,rate=1.,**options):
    if not math.isfinite(rate) or rate<=0:raise ValueError('Positive finite speed rate required.')
    return dict(path=str(Path(path).resolve()),**rate_notes(engine.get().parser.parse_file(str(path),rate=rate),model,**options))

def rate_content(content,model,*,rate=1.,**options):
    if not math.isfinite(rate) or rate<=0:raise ValueError('Positive finite speed rate required.')
    return rate_notes(engine.get().parser.parse_content(content,rate=rate),model,**options)
