"""Native v1.3.0 demand field and M4 scoring.

Static port of the frozen Stage 3B kernels. Keep operation order in the
numerical path: the release vectors were validated on that exact field.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
import math

import numpy as np
from numba import njit
from . import engine, geometry, hand, hand_cost, role
from ._core import model as M
from ._core.components import hb
from ._core.components.lncoord_v2 import CbarV2
from .engine import mirror
from .phase_average import integrate
from ._core.model import (
    AIN_GROUP,
    HAND,
    K,
    _abar_val,
    _chain_scatter,
    _chord_coupling,
    _coord_weight,
    _edge_ramp,
    _jack_kernel,
    _row_gesture_cached,
    _row_of_time,
    _run_lengths,
    _step_scatter,
    _wmean,
    build_jack_stats,
)


# Frozen row-geometry priors; not adjustable model coordinates.
P = MappingProxyType({'a_c': 25.83867288316744,
 'a_cb': 4.234752287195439,
 'a_cb_4k': 11.60113975415644,
 'a_cbc': 2.506126891341002,
 'a_p': 0.4016531544080434,
 'a_p_4k': 0.7336548088255943,
 'a_r': 108.50536487992183,
 'a_r_4k': 58.25324080278074,
 'abar_active': 0.02959105293159399,
 'abar_c0': 0.6575435661594596,
 'abar_c1': 1.3302402303785177,
 'abar_k': 5.052287546779305,
 'abar_mx': 0.451610888785001,
 'abar_mx_thr': 0.05208686478341369,
 'abar_mx_w': 0.4754900770279769,
 'abar_scale': 0.9908465677870455,
 'abar_thr_hi': 0.11599034279776298,
 'abar_thr_lo': 0.01389142404428909,
 'abar_weld': 1.0,
 'aggC_gap': 2270.295764558157,
 'aggC_gap_4k': 1222.4669501466997,
 'aggC_gc': 387.1237698484442,
 'aggC_gw': 1.2909164079307989,
 'aggC_hw': 1.0,
 'aggC_kap': 0.8250936661252044,
 'aggC_ms': 4397.439342626505,
 'aggC_ms_4k': 3078.207539838553,
 'aggC_pw1': 2.516135362334202,
 'aggC_pw2': 4.4266132719189,
 'aggC_tau': 4027.709861084976,
 'aggC_wc': 2.5507155298795627,
 'agg_C': 5.7765188650680015,
 'agg_gamma': 0.3142734186203886,
 'agg_gap_w': 1.0873927743990874,
 'agg_k': 3.923290274033005,
 'agg_mode': 4.0,
 'agg_nseg': 30.0,
 'agg_wc': 1.145022164188944,
 'aggw_gap_cap': 0.0,
 'aggw_src': 0.0,
 'aj': 1.430584921574431,
 'an_a0': 0.20919909312275142,
 'an_a1': 0.37286666751258996,
 'an_cubic': 8.862618779101767,
 'an_on_p': -0.32562582197542633,
 'ap': 3.5424464326854332,
 'at': 12.654197587970756,
 'c_chord2': 0.0,
 'c_fj': 1.291464498859916,
 'c_fj_4k': 1.1891597740767534,
 'c_ja': 0.038468075195409575,
 'c_jc': 0.0,
 'c_jm': 0.356418909837854,
 'c_s': 0.2848251338028861,
 'c_stam': 0.0,
 'calib_a': 0.9877134644832144,
 'calib_b': 0.6823998306913243,
 'cap_a': 22.11098017221394,
 'cap_b': 0.5877082264222335,
 'cb_holdtap': 0.0,
 'cb_holdtap_pow': 1.0,
 'cb_lock_pow': 1.0,
 'cb_lockanchor': 0.0,
 'cb_lockjack': 0.0,
 'cb_par': 0.0,
 'cb_shield': 0.0,
 'cb_shield_lock': 0.8062,
 'cb_shield_tau': 56.24,
 'cbv_churn': 3.6654074897742612,
 'cbv_churn_4k': 4.298621698858253,
 'cbv_holdage': 18.945855743518063,
 'cbv_holdage_4k': 10.303537336083915,
 'cbv_holdtap': -3.2366101454102796,
 'cbv_lock_pow': 0.9791217299061639,
 'cbv_lockanchor': 1.4544619305858701,
 'cbv_lockstack': 1.4797072901013206,
 'cbv_ls_1h_split': 0.59,
 'cbv_ls_2h_split': 0.37,
 'cbv_ls_cross': 0.2,
 'cbv_ls_sh_adj': 0.82,
 'cbv_ls_sh_split': 1.0,
 'cbv_ov_1h_split': 0.59,
 'cbv_ov_2h_split': 0.37,
 'cbv_ov_cap': 3446.987025249341,
 'cbv_ov_cross': 0.2,
 'cbv_ov_dt_max': 2276.1965593534787,
 'cbv_ov_sh_adj': 0.82,
 'cbv_ov_sh_split': 1.0,
 'cbv_overlap': -1.8394647223580556,
 'cbv_sh_dt_max': 330.6326350747474,
 'cbv_sh_ln_tau': 387.1884382046885,
 'cbv_sh_ln_w': 3.0,
 'cbv_sh_lock': 0.22,
 'cbv_sh_tau': 498.1744137671231,
 'cbv_shield': -1.976620227141156,
 'cbv_str_1h_split': 1.3155287246626448,
 'cbv_str_2h_split': 0.9916617411662032,
 'cbv_str_bal': 4.0,
 'cbv_str_cross': -0.6289598036143927,
 'cbv_str_h0': 0.3224021808746196,
 'cbv_str_h1': -2.2868450899924073,
 'cbv_str_h2': 2.6136511203207173,
 'cbv_str_h3': 2.340428184412323,
 'cbv_str_maxd': 2.6813954311063055,
 'cbv_str_sh_adj': -0.8710391482004185,
 'cbv_str_sh_split': 0.3664690221870731,
 'cbv_straddle': 9.213308770964225,
 'cbv_w': 421.0782656735729,
 'chord2_hard': 1.0,
 'chord2_w': 998.3068456460403,
 'chord_no_dilute': 1.0,
 'chord_soft_att': 1.0,
 'chord_tau_c': 24.345887557552615,
 'churn_event_rate': 0.0,
 'churn_sat': 20.70743322086763,
 'churn_w': 1167.6875064558421,
 'cj_norm': 3.9223792605276984,
 'cj_size_exp': 2.3090892612901857,
 'd_b1': 1.2519368453250523,
 'd_b2': 0.6276354594810581,
 'd_ds': 0.5851191625546722,
 'd_dt': 0.7540471413982772,
 'd_eye': 0.03491913373673157,
 'd_eye_4k': -0.022838270713520077,
 'd_hc': 1.0188595536994098,
 'd_read': 0.0,
 'd_read_4k': 0.0,
 'd_rec': 12.279350811907214,
 'd_sh': 0.0,
 'env_ms': 296.6324480607108,
 'env_w': 0.5401340304888701,
 'eye_tau': 0.2,
 'eye_w_ref': 4000.0,
 'eye_w_s': 500.0,
 'fj_bounded': 0.0,
 'ga_alpha': 0.0,
 'ga_gam': 1.0,
 'ga_pw': 1.0,
 'ga_q': 1.0,
 'group_jump': 0.0,
 'hand_average': 1.0,
 'hand_beta': 0.0,
 'hand_entry': 1.0,
 'hand_floor': 0.0,
 'hand_lookback': 1200.0,
 'hand_null': 0.1,
 'hand_power': 1.0,
 'hand_predecessor': 6.0,
 'hand_projection': 500.0,
 'hand_sync': 30.0,
 'hand_tau': 400.0,
 'hc_e': 1.7049026540688783,
 'holdage_cap': 0.42468477879654104,
 'holdage_th': 0.15716538407588404,
 'hw_a': -4.833333333333334,
 'hw_half': 51.932695284873475,
 'j_agg_pow': 2.8052526777466564,
 'j_c1': 0.16126641911276896,
 'j_nerf_a': 0.0,
 'j_nerf_c': 0.0924591760779166,
 'j_nerf_off': 0.09265897011180577,
 'j_triple_gain': 0.0,
 'j_triple_gain_4k': 0.0,
 'j_triple_pow': 2.0072942534242566,
 'j_triple_tau': 140.9631484638465,
 'ja_coord_exp': 0.012636794260452278,
 'ja_gap_thr': 250.97771855895374,
 'ja_len_exp': 0.873949786346038,
 'ja_len_floor': 1.0,
 'ja_norm': 4.912805983390194,
 'ja_scale': 0.9792209480655096,
 'jc_scale': 1.4332747701403683,
 'ln_eff_tail_ms': 11.348134111824663,
 'ln_note_level': 0.0,
 'mean_pow': 6.465267031930503,
 'mj_dilute': -0.022067037291399555,
 'mj_dilute_exp': 1.7147578892981583,
 'note_chord': 1.0,
 'od_mult_ja': -1.0,
 'od_mult_jc': -0.293176786084059,
 'od_mult_jm': 1.2040048799564311,
 'od_mult_p': 1.1286927800588342,
 'od_mult_r': 1.058858405816053,
 'od_mult_x': 1.1974891491877375,
 'out_a': 0.8686939253627315,
 'out_b': -0.9591165705858606,
 'p83_c': 2.667686959362696,
 'p93_c': 2.505172683584436,
 'p_boost': -1e-05,
 'p_boost_hi': 296.9197535881636,
 'p_boost_lo': 117.67575960429505,
 'p_burst3': 0.0,
 'p_burst4': 0.0,
 'p_bv_max': 0.7121791011599433,
 'p_chw2': 0.6997350782823039,
 'p_chw3': 1.4710731528472405,
 'p_chw4': 0.82149995851254,
 'p_chw5': 0.8676070127257947,
 'p_chw6': 0.7474127532548505,
 'p_chw7': 1.03284160083206,
 'p_lam2': 0.005077940680001312,
 'p_lam3': 60.0,
 'p_sat_a': 12.967278061527875,
 'p_sat_b': 2.3046546300449706,
 'p_scale': 0.0573287291354225,
 'p_v_norm': -0.03652823337937215,
 'pp_div': 1.559992046815954,
 'pp_n0': 0.0,
 'pp_scale': 0.9866220368284715,
 'pp_thr': 8.416405509249257,
 'r_I_off': 1.6936500452160337,
 'r_I_steep': 7.138399583260547,
 'r_I_w': 1.3201531009318856,
 'r_ain0': 0.0,
 'r_ain1': 2.466978961221889,
 'r_ain2': 2.5020134256224296,
 'r_ain3': 1.2702424995871497,
 'r_coord_e': 2.2703135423553786,
 'r_cw_cross': 0.7315226193212329,
 'r_cw_hand': 0.46562083856474745,
 'r_cw_same': 1.5375480228960465,
 'r_cw_thumb': 0.011960920198257674,
 'r_dt_max': 6749.562099456267,
 'r_dtr_min': 0.004050313466366775,
 'r_lock': 2.064385891165511,
 'r_order_pen': 0.23401924932585802,
 'r_order_tau': 702.5373743325986,
 'r_same_col': 0.19578505600824977,
 'r_seq': 0.06635864238365655,
 'r_short_red': 0.0,
 'r_short_thr': 232.63366268425537,
 'r_sim_tau': 11.405453069087805,
 'r_simul': 0.48507152260049274,
 'r_soft_edges': 1.0,
 'r_straddle': 0.4485333102831889,
 'r_tail': 0.030479803326367228,
 'r_tt': 0.0,
 'read_clip_hi': 1.5,
 'read_clip_lo': 0.5,
 'read_n_win': 16.0,
 'read_tol': 0.18,
 'rec_half': 0.3671841225596574,
 'recov_w': 968.8229953754064,
 'recov_wl': 38307.95803840174,
 's_p': 1.1959916923340246,
 'shd_w': 1007.2471613945766,
 'stam_grp': 30.0,
 'stam_tau': 8000.0,
 't_jack_mix': -0.3805743609330355,
 't_s_off': 3.0179360184812842,
 'use_cbar_v2': 1.0,
 'use_cbv_holdtap': 1.0,
 'use_cbv_lockanchor': 1.0,
 'use_cbv_lockstack': 1.0,
 'use_cbv_overlap': 1.0,
 'use_cbv_shield': 1.0,
 'use_cbv_straddle': 1.0,
 'use_eye': 1.0,
 'use_hb': 1.0,
 'use_read': 0.0,
 'w83': 0.4815139944602175,
 'w93': 0.3605157897072774,
 'w_a': 75.76213228110679,
 'w_cb': 1716.856156378797,
 'w_ja': 0.0,
 'w_jc': -0.4552854599596368,
 'w_jm': 934.2448253030639,
 'w_p': 902.5681256843202,
 'w_r': 1972.5181634749874,
 'w_x': 927.941415540503,
 'wmean': 2.1613377064577026,
 'x_adj_w': -1.4830291674247595,
 'x_amp': 0.09636644976733964,
 'x_cw0': 0.03724653823590479,
 'x_cw1': 0.8550132389171754,
 'x_cw2': 0.4722195790253056,
 'x_cw3': 0.023511327398546056,
 'x_din': 0.21329641269001137,
 'x_dir_in': 0.7935947499263032,
 'x_dir_out': -0.46052690940589464,
 'x_dout': 0.3884059305040078,
 'x_fc_a': 0.5315844156004722,
 'x_fc_floor': 0.07402097926631954,
 'x_fc_off': 62.51708153410152,
 'x_fc_w': 1.68396911611816,
 'x_jd_e': 2.070662416719641,
 'x_jd_p': 0.038157319982258006,
 'x_jh_e': 0.7460102426771034,
 'x_jh_p': 0.761399733967284,
 'x_jt': 0.5379957175084137,
 'x_jump_w': 0.0})


P_TRIP = {**P, "j_triple_gain": 1.0, "j_triple_gain_4k": 1.0, "c_ja": 0.0, "c_jc": 0.0, "c_fj": 0.0, "c_fj_4k": 0.0}


SHAPE1, SHAPE2 = ((1.14, 1.1), (1.62, 1.48))


NAMES = ('c_jm',
 'c_ja',
 'c_s',
 'aj',
 'cap_a',
 'cap_b',
 's_p',
 'a_p',
 'a_p_4k',
 'a_r',
 'a_r_4k',
 'a_c',
 'a_cb',
 'a_cb_4k',
 'a_cbc',
 'ap',
 'at',
 'd_hc',
 'hc_e',
 'ga_alpha',
 'ga_shape',
 'p_burst3',
 'x_base',
 'x_fast',
 'x_context',
 'k_jt',
 't_s_off',
 'd_b1',
 'd_ds',
 'd_dt',
 'd_b2',
 'role_price',
 'hold_floor',
 'ln_gate',
 'churn',
 'holdage',
 'agg_p1',
 'agg_p2',
 'mass_pow',
 'gap_pow',
 'hold_mass',
 'env_w',
 'j_triple_4k',
 'pp_scale',
 'pp_thr',
 'pp_div',
 'pp_bias',
 'c_stam',
 'tau_stam',
 'stam_theta',
 'stam_half',
 'xh_holdtap',
 'sh_holdtap_extra',
 'lockstack',
 'lockanchor',
 'shield',
 'overlap',
 'rhythm',
 'roll',
 'reversal')


for _i, _name in enumerate(NAMES):
    globals()["I_" + _name.upper()] = _i


FIELDS = ('jm_raw',
 'jm_hand',
 'ja',
 'pressure',
 'chord1',
 'chord2',
 'burst',
 'x_base',
 'x_fast',
 'x_context',
 'abar',
 'ks',
 'rbar',
 'amount',
 'hc1',
 'hc22',
 'role',
 'churn',
 'holdage',
 'xh_holdtap',
 'sh_holdtap_extra',
 'lockstack',
 'lockanchor',
 'shield',
 'overlap',
 'rhythm',
 'roll',
 'reversal',
 'jm_trip',
 'hold_support')


# The plain path prunes the dormant triple branch; 4K prices only the positive increment.
def _jack_stats(b, s: Struct, p: dict) -> list:
    """Per-column jack-gap arrays.

    Extracted verbatim from `compute_curves` so the Cbar v2 component
    (`core.components.lncoord_v2`) can build per-column lock-modulated
    curves without duplicating this loop.  Pure move -- no arithmetic change.
    """
    xrow = s.x[s.chart]
    gap_thr = float(p['ja_gap_thr']) / 1000.0
    tau_c = float(p.get('chord_tau_c', 0.0))
    att_cum = None
    att_row = None
    if tau_c > 0.0:
        wf, wb = _chord_coupling(s, tau_c)
        if p.get('chord_soft_att', 0.0) >= 0.5:
            att_row = 1.0 - wb
        else:
            att_cum = np.concatenate(([0.0], np.cumsum((wb == 0.0).astype(np.float64))))
    att_soft_cum = None if att_row is None else np.concatenate(([0.0], np.cumsum(att_row)))
    chord_row = None
    if p.get('chord_no_dilute', 0.0) >= 0.5 and p.get('note_chord', 0.0) >= 0.5:
        chord_row = _row_gesture_cached(b, s, p)[2] >= 2.0
    stats = []
    for k in range(K):
        c = s.col_chain[k]
        pos = c['pos']
        if pos.size < 2:
            stats.append(None)
            continue
        g = c['g']
        valid = np.isfinite(g) & (np.diff(c['cid']) == 0)
        gs = np.maximum(np.where(valid, g, 1.0), 0.0001)
        if att_cum is not None:
            nrows = np.maximum(att_cum[pos[1:]] - att_cum[pos[:-1]], 1.0)
        elif att_soft_cum is not None:
            nrows = np.maximum(att_soft_cum[pos[1:]] - att_soft_cum[pos[:-1]], 1.0)
        else:
            nrows = np.maximum(pos[1:] - pos[:-1], 1).astype(np.float64)
        total = s.cumsize[pos[1:]] - s.cumsize[pos[:-1]]
        other = np.maximum(total - 1.0, 0.0)
        same_step = np.diff(c['cid']) == 0
        runlen = _run_lengths((g <= gap_thr) & same_step)
        cid_gap = c['cid'][:-1]
        if p.get('fj_bounded', 0.0) >= 0.5:
            fj = np.where(same_step & (gs < 0.15), np.clip(1.0 - gs / 0.15, 0.0, 1.0), 0.0)
        else:
            fj = np.where(same_step & (gs < 0.15), 1.0 / gs * np.maximum(1.0 - gs / 0.15, 0.0), 0.0)
        stats.append(dict(pos=pos, gs=gs, valid=valid, nrows=nrows, other=other, load=total / nrows, runlen=runlen, fj=fj, xr=xrow[pos[:-1]], chord_row=None if chord_row is None else chord_row[pos[:-1]], kern=_jack_kernel(gs, xrow[pos[:-1]], p, float(p['od_mult_jm']))))
    return stats


def JACK(b, s, p):
    n = s.n
    xrow = s.x[s.chart]
    agg_pow = max(float(p['j_agg_pow']), 0.25)
    ja_len_exp = float(p['ja_len_exp'])
    ja_coord_exp = float(p['ja_coord_exp'])
    ja_floor = float(p['ja_len_floor'])
    ja_norm = max(float(p['ja_norm']), 0.001)
    w_jm = float(p['w_jm'])
    od_jm = float(p['od_mult_jm'])
    w_ja = float(p.get('w_ja', 0.0))
    w_ja = w_jm if w_ja <= 0.0 else w_ja
    od_ja = float(p.get('od_mult_ja', -1.0))
    od_ja = od_jm if od_ja < 0.0 else od_ja
    stats = _jack_stats(b, s, p)

    def jack_channel(kind: str, w_ms: float, od_mult: float) -> np.ndarray:
        num = np.zeros(n)
        den = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            if kind == 'plain':
                mod = np.ones(st['pos'].size - 1)
            elif kind == 'mj':
                dil = np.ones_like(st['gs'])
                if st.get('chord_row') is not None:
                    dil = np.where(st['chord_row'], 1.0, dil)
                mod = dil
            else:
                e = np.maximum(st['runlen'] + 2.0 - ja_floor, 0.0)
                runf = np.power(e / (e + ja_norm), ja_len_exp)
                mod = runf * np.power(1.0 + st['other'] / st['nrows'], ja_coord_exp)
            kern = st['kern'] if od_mult == od_jm else _jack_kernel(st['gs'], st['xr'], p, od_mult)
            v = np.where(st['valid'], kern * mod, 0.0)
            cur = np.maximum(_wmean(s, _chain_scatter(st['pos'], v, n, st['valid']), w_ms), 0.0)
            w = _chain_scatter(st['pos'], np.where(st['valid'], 1.0 / st['gs'], 0.0), n, st['valid'])
            w = _wmean(s, w, w_ms) if kind == 'mj' else w
            num += np.power(cur, agg_pow) * w
            den += w
        val = np.where(den > 1e-09, num / np.where(den > 1e-09, den, 1.0), 0.0)
        return np.power(np.maximum(val, 0.0), 1.0 / agg_pow)
    need = set()
    if p['c_jm'] or p['t_jack_mix'] or p['cb_lockjack']:
        need.add('mj')
    if p['c_ja'] or p['cb_lockanchor']:
        need.add('ja')
    if p['c_s'] or p['a_p']:
        need.add('plain') if False else None
    need.add('plain')
    Jm = jack_channel('mj', w_jm, od_jm)
    c_fj = float(p.get('c_fj', 0.0))
    c_fj_4k = float(p.get('c_fj_4k', 0.0))
    if c_fj != 0.0 or c_fj_4k != 0.0:
        FJ = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            FJ += _chain_scatter(st['pos'], np.where(st['valid'], st['fj'], 0.0), n, st['valid'])
        FJ = _wmean(s, FJ, w_jm)
        c_fj_row = np.where(s.is_4k[s.chart], c_fj_4k, c_fj)
        Jm = Jm + c_fj_row * FJ
    Ja = jack_channel('ja', w_ja, od_ja) if 'ja' in need else Jm
    return (Jm, Ja)


def JACK_TRIP(b, s, p):
    n = s.n
    xrow = s.x[s.chart]
    agg_pow = max(float(p['j_agg_pow']), 0.25)
    triple_gain = float(p['j_triple_gain'])
    mj_dilute = float(p['mj_dilute'])
    mj_exp = float(p['mj_dilute_exp'])
    cj_exp = float(p['cj_size_exp'])
    ja_len_exp = float(p['ja_len_exp'])
    ja_coord_exp = float(p['ja_coord_exp'])
    ja_floor = float(p['ja_len_floor'])
    cj_norm = max(float(p['cj_norm']), 0.001)
    ja_norm = max(float(p['ja_norm']), 0.001)
    w_jm = float(p['w_jm'])
    od_jm = float(p['od_mult_jm'])
    w_jc = float(p.get('w_jc', 0.0))
    w_jc = w_jm if w_jc <= 0.0 else w_jc
    w_ja = float(p.get('w_ja', 0.0))
    w_ja = w_jm if w_ja <= 0.0 else w_ja
    od_jc = float(p.get('od_mult_jc', -1.0))
    od_jc = od_jm if od_jc < 0.0 else od_jc
    od_ja = float(p.get('od_mult_ja', -1.0))
    od_ja = od_jm if od_ja < 0.0 else od_ja
    stats = build_jack_stats(b, s, p)

    def jack_channel(kind: str, w_ms: float, od_mult: float) -> np.ndarray:
        num = np.zeros(n)
        den = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            if kind == 'plain':
                mod = np.ones(st['pos'].size - 1)
            elif kind == 'mj':
                dil = np.ones_like(st['gs'])
                if st.get('chord_row') is not None:
                    dil = np.where(st['chord_row'], 1.0, dil)
                mod = dil * (1.0 + st['tgain'] * st['trip'])
            elif kind == 'cj':
                mod = np.power(np.maximum(st['load'], 1e-06) / cj_norm, cj_exp)
            else:
                e = np.maximum(st['runlen'] + 2.0 - ja_floor, 0.0)
                runf = np.power(e / (e + ja_norm), ja_len_exp)
                mod = runf * np.power(1.0 + st['other'] / st['nrows'], ja_coord_exp)
            kern = st['kern'] if od_mult == od_jm else _jack_kernel(st['gs'], st['xr'], p, od_mult)
            v = np.where(st['valid'], kern * mod, 0.0)
            cur = np.maximum(_wmean(s, _chain_scatter(st['pos'], v, n, st['valid']), w_ms), 0.0)
            w = _chain_scatter(st['pos'], np.where(st['valid'], 1.0 / st['gs'], 0.0), n, st['valid'])
            w = _wmean(s, w, w_ms) if kind == 'mj' else w
            num += np.power(cur, agg_pow) * w
            den += w
        val = np.where(den > 1e-09, num / np.where(den > 1e-09, den, 1.0), 0.0)
        return np.power(np.maximum(val, 0.0), 1.0 / agg_pow)
    need = set()
    if p['c_jm'] or p['t_jack_mix'] or p['cb_lockjack']:
        need.add('mj')
    if p['c_jc']:
        need.add('cj')
    if p['c_ja'] or p['cb_lockanchor']:
        need.add('ja')
    if p['c_s'] or p['a_p']:
        need.add('plain') if False else None
    need.add('plain')
    Jbar = jack_channel('plain', w_jm, od_jm)
    Jm = jack_channel('mj', w_jm, od_jm) if 'mj' in need else Jbar
    c_fj = float(p.get('c_fj', 0.0))
    c_fj_4k = float(p.get('c_fj_4k', 0.0))
    if c_fj != 0.0 or c_fj_4k != 0.0:
        FJ = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            FJ += _chain_scatter(st['pos'], np.where(st['valid'], st['fj'], 0.0), n, st['valid'])
        FJ = _wmean(s, FJ, w_jm)
        c_fj_row = np.where(s.is_4k[s.chart], c_fj_4k, c_fj)
        Jm = Jm + c_fj_row * FJ
    Jc = jack_channel('cj', w_jc, od_jc) if 'cj' in need else Jm
    Ja = jack_channel('ja', w_ja, od_ja) if 'ja' in need else Jm
    return Jm


def abar(s, p, n):
    A = np.ones(n)
    dk = s.dcol
    act = s.act
    if p['abar_active'] >= 0.5:
        prev_d = np.full(n, -1.0)
        prev_a = np.full(n, -1.0)
        for k in range(K):
            d0, a0 = (prev_d, prev_a)
            on = (a0 >= 0.0) & act[k]
            d1 = dk[k]
            mx = np.maximum(np.where(on, d0, 0.0), d1)
            dd = np.abs(np.where(on, d0, d1) - d1) + p['abar_mx_w'] * np.maximum(0.0, mx - p['abar_mx_thr'])
            val = _abar_val(dd, mx, p)
            A *= np.where(on, val, 1.0)
            prev_d = np.where(act[k], d1, d0)
            prev_a = np.where(act[k], 1.0, a0)
    else:
        for k in range(K - 1):
            on = act[k] & act[k + 1]
            d0, d1 = (dk[k], dk[k + 1])
            mx = np.maximum(d0, d1)
            dd = np.abs(d0 - d1) + p['abar_mx_w'] * np.maximum(0.0, mx - p['abar_mx_thr'])
            val = _abar_val(dd, mx, p)
            A *= np.where(on, val, 1.0)
    Abar = _wmean(s, A, float(p['w_a'])) * p['abar_scale']
    return Abar


def R_POSITIVE(b, s: Struct, p: dict, xrow: np.ndarray) -> np.ndarray:
    n = s.n
    t = s.tail
    if t is None:
        return np.zeros(n)
    xr = np.power(xrow, float(p['od_mult_r']))
    xs = xr[t['j0']]
    dur = t['t'] - t['h']
    I_h = 0.001 * np.abs(dur - 80.0) / xs
    nh = t['nh']
    I_t = np.where(np.isfinite(nh), 0.001 * np.abs(nh - t['t'] - 80.0) / xs, 10.0)
    I = 2.0 / (2.0 + np.exp(-p['r_I_steep'] * (I_h - p['r_I_off'])) + np.exp(-p['r_I_steep'] * (I_t - p['r_I_off'])))
    R = np.zeros(n)
    dtr_min = p.get('r_dtr_min', 0.0001)
    soft = p.get('r_soft_edges', 0.0)
    wfloor = dtr_min * 1000.0 if soft else 0.0
    sim_tau = p.get('r_sim_tau', 0.0)
    nxt = t['nxt']
    nxt_is_tail = t['is_tail']
    nxt_col = t['nxt_col']
    ok = np.isfinite(nxt) & (nxt - t['t'] <= p['r_dt_max'])
    if ok.any():
        dtr = np.where(ok, (nxt - t['t']) / 1000.0, 1.0)
        rv = p['r_tail'] * np.power(np.maximum(dtr, dtr_min), -0.5) / xs * (1.0 + p['r_I_w'] * I)
        same_col = (nxt_col == t['col']) & ~nxt_is_tail
        rv = rv * np.where(same_col, p['r_same_col'], 1.0)
        cw = _coord_weight(t['col'], nxt_col, p)
        if sim_tau > 0.0:
            with np.errstate(invalid='ignore'):
                d_ev = t['nt'] - nh
            beta = np.clip(0.5 - d_ev / (2.0 * sim_tau), 0.0, 1.0)
            beta = np.where(np.isfinite(d_ev), beta, np.where(nxt_is_tail, 1.0, 0.0))
            ex = p['r_coord_e'] * (beta + (1.0 - beta) * p['r_tt'])
        else:
            ex = p['r_coord_e'] * np.where(nxt_is_tail, 1.0, p['r_tt'])
        rv = rv * np.power(np.maximum(cw, 0.001), ex)
        thr = max(p['r_short_thr'], 1.0)
        red = p['r_short_red'] + (1.0 - p['r_short_red']) * np.minimum(dur, thr) / thr
        rv = rv * np.where(dur > 0, red, 1.0)
        lock = np.zeros(t['col'].size)
        hm = s.heldmask[t['j0']]
        col = t['col']
        if p['r_lock']:
            for j in range(K):
                bit = (hm >> j & 1).astype(bool) & (j != col)
                if bit.any():
                    lock = lock + np.where(bit, _coord_weight(col, np.full(col.size, j), p), 0.0)
        rv = rv * (1.0 + p['r_lock'] * lock)
        if p['r_straddle']:
            ain = np.array([p['r_ain0'], p['r_ain1'], p['r_ain2'], p['r_ain3']])
            ag = AIN_GROUP
            S0 = np.zeros(t['col'].size)
            for j in range(K):
                bit = (hm >> j & 1).astype(bool)
                if bit.any():
                    S0 = S0 + np.where(bit, ain[ag[j]], 0.0)
            S0 = S0 - np.where((hm >> col & 1).astype(bool), ain[ag[col]], 0.0)
            outer = ((hm & 1) != 0).astype(float) + ((hm & 64) != 0).astype(float)
            outer -= (((hm >> col) & 1) != 0) & (ag[col] == 0)
            drive = 1.0 + p['r_straddle'] * S0
            rv = rv * drive / (1.0 + 1.14 * outer / np.maximum(drive, 0.3))
        if soft:
            rv = rv * _edge_ramp(nxt - t['t'], p['r_dt_max'])
        rv = np.where(ok, rv, 0.0)
        end_t = np.where(ok, np.maximum(t['nxt'], t['t'] + wfloor), t['t'])
        j1 = _row_of_time(s, t, end_t)
        if soft:
            span_i = np.maximum(end_t - t['t'], 0.0)
            j1c = np.minimum(np.maximum(j1, t['j0']), n - 1)
            grid_span = s.t[j1c] - s.t[t['j0']]
            den = np.maximum(grid_span, np.maximum(span_i, 1e-09))
            scale = np.where(span_i > 0.0, span_i / den, 1.0)
            rv = rv * scale
        R = R + _step_scatter(t['j0'], np.maximum(j1, t['j0']), rv, n)
    m = t['col'].size - 1
    if m > 0:
        idx = np.arange(m)
        idx = idx[t['chart'][idx] == t['chart'][idx + 1]]
        ch = t['chart'][idx]
        dtr = (t['t'][idx + 1] - t['t'][idx]) / 1000.0
        dtr = np.maximum(dtr, dtr_min)
        cw = _coord_weight(t['col'][idx], t['col'][idx + 1], p)
        sv = p['r_seq'] * np.power(dtr, -0.5) / xs[idx] * (1.0 + p['r_I_w'] * (I[idx] + I[idx + 1])) * np.power(np.maximum(cw, 0.001), p['r_coord_e'])
        j0 = t['j0'][idx]
        end_B = np.maximum(t['t'][idx + 1], t['t'][idx] + wfloor)
        j1 = np.maximum(_row_of_time(s, t, end_B, idx + 1), j0)
        if soft:
            span_i = np.maximum(end_B - t['t'][idx], 0.0)
            j1c = np.minimum(j1, n - 1)
            grid_span = s.t[j1c] - s.t[j0]
            sv = sv * (span_i / np.maximum(grid_span, span_i))
        R = R + _step_scatter(j0, j1, sv, n)
    if p['r_order_pen']:
        m = t['col'].size - 2
        if m > 0:
            i = np.arange(m)
            ch = t['chart'][i]
            ok3 = ch == t['chart'][i + 2]
            span = t['t'][i + 2] - t['t'][i]
            if not soft:
                ok3 &= span <= p['r_order_tau']
            c1, c2, c3 = (t['col'][i], t['col'][i + 1], t['col'][i + 2])
            hh = HAND[c1]
            ok3 &= (hh == HAND[c2]) & (hh == HAND[c3]) & (hh != 1)
            ok3 &= (c1 != c2) & (c2 != c3) & (c1 != c3)
            jump = np.abs(c1.astype(np.int64) - c2) + np.abs(c2.astype(np.int64) - c3)
            lo = np.minimum(np.minimum(c1, c2), c3).astype(np.int64)
            hi_ = np.maximum(np.maximum(c1, c2), c3).astype(np.int64)
            excess = np.maximum(jump - (hi_ - lo), 0.0).astype(np.float64)
            early = np.maximum(t['t'][i + 1] - t['t'][i], 0.)
            late = np.maximum(t['t'][i + 2] - t['t'][i + 1], 0.)
            contrast = abs(early - late) / np.maximum(early + late, 1.)
            val = np.where(ok3, p['r_order_pen'] * excess / xs[i] * (0.45 + 0.85 * contrast), 0.)
            if soft:
                val = val * _edge_ramp(span, p['r_order_tau'])
            j0 = t['j0'][i]
            end_C = np.maximum(t['t'][i + 2], t['t'][i] + wfloor)
            j1 = np.maximum(_row_of_time(s, t, end_C, i + 2), j0)
            if soft:
                span_i = np.maximum(end_C - t['t'][i], 0.0)
                j1c = np.minimum(j1, n - 1)
                grid_span = s.t[j1c] - s.t[j0]
                val = val * (span_i / np.maximum(grid_span, span_i))
            R = R + _step_scatter(j0, j1, val, n)
    return np.maximum(_wmean(s, R, float(p['w_r'])), 0.0)


def PRESSURE(b,s,p,xrow,n):
    """Unit-price pressure atoms for the active mature note_chord path."""
    xp=np.power(xrow,float(p["od_mult_p"]))
    d=s.dt
    tau=float(p["chord_tau_c"])
    ul,ur,_,_,wb=M._row_gesture_cached(b,s,p)
    att=1.-wb
    att_idx=np.flatnonzero(wb<=0.)
    rows=np.arange(n)
    pos=np.searchsorted(att_idx,rows,side="right")
    nxt=att_idx[np.minimum(pos,att_idx.size-1)]
    valid=(pos<att_idx.size)&(s.chart[nxt]==s.chart)
    d_fp=np.where(valid,(s.t[nxt]-s.t)/1000.,d)
    d_fp=np.where(d_fp>0.,d_fp,d)
    q=np.minimum(d_fp-xp/2.,xp/6.)
    fp=np.power(np.maximum(p["p_scale"]/xp*(1.-p["p_lam3"]/xp*q*q),0.),.25)
    vv=1.+p["p_lam2"]*s.ln_body_ms/np.power(np.maximum(s.dt_ms,1.),float(p["p_v_norm"]))
    inc=1./d*fp*(att+np.maximum(vv-1.,0.))
    anchor=1.+p["an_on_p"]*np.minimum(s.anchor_raw-p["an_a0"],p["an_cubic"]*np.power(s.anchor_raw-p["an_a1"],3.))
    inc=np.minimum(inc*anchor,np.maximum(inc,inc*float(p["p_sat_b"])-float(p["p_sat_a"])))
    return inc,ul,ur,tau


def boundary(b,s,p,xrow,tau):
    """Established boundary base, with a C1 genuine-cross-only acceleration."""
    total=np.zeros(s.n);fcs=[]
    xp=xrow**p["od_mult_x"]
    cw=np.array([p[f"x_cw{i}"] for i in range(4)])
    for bd in range(M.N_BOUND):
        raw=s.bnd_chain[bd]["pos"]
        if len(raw)<2:
            fcs.append(np.zeros(s.n));continue
        pos=raw[np.r_[True,np.diff(s.t[raw])>=tau]]
        if len(pos)<2:
            fcs.append(np.zeros(s.n));continue
        e=M.coordinate(s,bd,pos,tau)
        g=np.diff(s.t[pos])/1000.
        a,z=pos[:-1],pos[1:]
        dd=abs(e[:-1]-3)-abs(e[1:]-3)
        gx=np.maximum(g,xp[a])
        amp=p["x_amp"]*gx**-2*(1+p["x_din"]*np.maximum(dd,0)+p["x_dout"]*np.maximum(-dd,0))
        good=np.ones(len(amp),bool)
        total+=cw[M.BOUND_GROUP[bd]]*M._chain_scatter(pos,amp,s.n,good)
        # Saturating smooth odds. The OD-dependent physical scale replaces
        # the single global x_fc_floor; a near-zero time gap remains finite.
        base=np.sqrt(g*g+(.57*xp[a])**2)
        drive=p["x_fc_a"]/base**2-p["x_fc_off"]
        soft=7.0*np.logaddexp(0,drive/7.)
        different=(s.mask[a]^s.mask[z])!=0
        change=np.minimum(abs(e[1:]-e[:-1]),1.)
        fcs.append(M._chain_scatter(pos,soft*different*change,s.n,good))
    fast=np.zeros(s.n)
    for bd in range(M.N_BOUND-1):
        w=np.sqrt(cw[M.BOUND_GROUP[bd]]*cw[M.BOUND_GROUP[bd+1]])
        fast+=p["x_fc_w"]*w*np.sqrt(np.maximum(fcs[bd]*fcs[bd+1],0.))
    a=M._wmean(s,total,p["w_x"])
    c=M._wmean(s,fast,p["w_x"])
    return a,c


@njit(cache=True)
def envelope(d,half):
    if len(d)<3:return d.copy()
    out=np.empty(len(d));ix=np.empty(len(d),np.int64)
    a=0;b=0;j=0
    for i in range(len(d)):
        lim=min(i+half,len(d)-1)
        while j<=lim:
            while b>a and d[ix[b-1]]<=d[j]:b-=1
            ix[b]=j;b+=1;j+=1
        while ix[a]<i-half:a+=1
        out[i]=d[ix[a]]
    return out


def coord_atoms(nt, s, p, compact):
    """Pairwise exact note identities and CbarV2 primitives on the RC3 row grid."""
    v = CbarV2(s.batch, s, p)
    hold = np.zeros((2,s.n))
    hand_ids = M.HAND
    # s.heldmask omits heads struck exactly at this row; charge occupied
    # fingers against real taps, not the simultaneous heads of new holds.
    for k in range(7):
        tap = (nt.col == k) & ~nt.is_ln
        if not tap.any():
            continue
        row = np.searchsorted(s.t,nt.time[tap])
        held = v._bits(row)
        mat = v._lockmat()[k]
        same = (hand_ids == hand_ids[k]) & (hand_ids[k] != 1) & (np.arange(7) != k)
        # A shared cross-hand-rate baseline applies to every occupied hand;
        # only same-hand occupancy can earn the extra surcharge.
        np.add.at(hold[0],row,(held @ ((mat>0)*.20))/s.dt[row])
        np.add.at(hold[1],row,(held @ (same*mat*3.))/s.dt[row])
    w = float(p["cbv_w"])
    z = np.zeros(s.n)
    terms = (v._t_lockstack(),z,z,z) if compact else (
        v._t_lockstack(), v._t_lockanchor(), v._t_shield(), v._t_overlap())
    return [M._wmean(s,x,w) for x in hold] + [np.maximum(x,0.) for x in terms]


def movement(s, p):
    """Smooth positive IOI contrast, directed rolls and reversals."""
    r = np.zeros((3,s.n))
    if s.n >= 3:
        d = np.diff(s.t)
        u = d[1:] + d[:-1]
        r[0,1:-1] = ((d[1:]-d[:-1])/np.maximum(u,1.))**2 * 1000. / (u+80.)
        c = np.arange(7,dtype=float)
        center = ((s.mask[:,None] >> np.arange(7)) & 1) @ c / np.maximum(s.size,1)
        a,b = np.diff(center)[:-1],np.diff(center)[1:]
        physical = np.minimum(abs(a)*abs(b),4.) * 1000. / (u+80.)
        r[1,1:-1] = np.where(a*b>0.,physical,0.)
        r[2,1:-1] = np.where(a*b<0.,physical,0.)
    return [M._wmean(s,z,p["w_x"]) for z in r]


@dataclass(frozen=True)
class Structural:
    a: np.ndarray
    b: np.ndarray
    t: np.ndarray
    ks: np.ndarray
    head_pos: np.ndarray
    key: int
    od: float
    gp: np.ndarray
    amount: np.ndarray
    held: np.ndarray
    interval: np.ndarray
    section: np.ndarray
    win_ms: float
    env_half: int


def extract_rows(nt, s, p, d_role, compact):
    b = s.batch
    x = s.x[s.chart]
    jm,ja = JACK(b,s,p)
    jm_trip = JACK_TRIP(b,s,{**p,**P_TRIP})
    trip_delta = np.maximum(jm_trip - jm, 0.)
    inc,ul,ur,tau = PRESSURE(b,s,p,x,s.n)
    def price_chord(g,q):
        g = M._chord_g(ul,ur,{"ga_alpha":1.,"ga_gam":g,"ga_q":q,"ga_pw":1.})
        return M._wmean(s,np.maximum(inc*(g-1.),0.),p["w_p"])
    q1,q2 = price_chord(*SHAPE1), price_chord(*SHAPE2)
    w = lambda y: M._wmean(s,y,p["w_p"])
    burst = w(M._wcount(s,s.note_impulse,100.)*10.)
    bx,fx = boundary(b,s,p,x,tau)  # unmodulated Xbase and C1 cross-only Xfast
    context = p["x_adj_w"] * geometry.window_curve(b,s,p) if not compact else np.zeros(s.n)
    A = abar(s,p,s.n)
    z = np.zeros(s.n)
    if nt.n_ln:
        R = np.maximum(R_POSITIVE(b,s,p,x),0.)
        churn,age = hb.compute_churn(b,s,p),hb.compute_holdage(b,s,p)
        ln = coord_atoms(nt,s,p,compact)
    else:
        R=churn=age=z
        ln=[z]*6
    hand_state = hand_cost.state(hand.field(hand.extract(nt,30.),s.t,p["w_jm"]))
    unit = hand_cost.price(hand_state,s.x[0],p)
    def hc(e):
        return M._wmean(s,np.maximum(ul-1,0.)**e+np.maximum(ur-1,0.)**e,p["shd_w"])
    return np.column_stack((jm,unit,ja,w(np.maximum(inc,0.)),q1,q2,burst,bx,fx,context,A,s.Ks,R,s.C,
                            hc(1.),hc(2.2),d_role,churn,age,*ln,*movement(s,p),trip_delta,z))


def state(notes, compact=False):
    n = role.canonical(notes)
    other = mirror(n)
    if (other.col.tobytes(),other.ln_end.tobytes()) < (n.col.tobytes(),n.ln_end.tobytes()):
        n,other = other,n
    s = engine.get().build(n,P)
    d_role = role.field(n,P,s.t)
    a = extract_rows(n,s,P,d_role,compact)
    b = a[:0] if compact else extract_rows(other,engine.get().build(other,P),P,d_role,False)
    t = s.t.copy()
    gp = .5*(np.r_[0.,s.dt_diff]+np.r_[s.dt_diff,0.])
    amount,held = s.C.copy(),s.held.copy()
    ks = s.Ks.astype(np.float64)
    if n.n_ln and len(t)==1:
        span = float(np.max(n.ln_end)-t[0])
        if span>0:
            center = t[0]+.5*span
            h = np.sum(n.is_ln & (n.ln_end>center))
            def support(x):
                f = np.zeros(len(FIELDS));f[10:12]=1.;f[13]=1.;f[-1]=h
                x=np.vstack((.34*x[0],f));x[0,10:12]=a[0,10:12];x[0,13]=1.2
                return x
            a,b = support(a), b if compact else support(b)
            t=np.r_[t,center];gp=np.full(2,min(max(span*.5,5.),350.))
            amount=np.array([1.2,1.]);held=np.r_[held,h];ks=np.r_[ks,1.]
    w,g,w4,g4,_ = M._aggc_geom(P)
    if n.key_count==4:
        w,g=w4,g4
    cut = np.r_[True,np.diff(t)>=g]; sec=np.cumsum(cut)-1
    local_step=float(np.quantile(np.diff(t),.5)) if len(t)>1 else 1.
    half=max(1,int(round(.5*P["env_ms"]/max(local_step,1.))))
    return Structural(a,b,t,ks,np.searchsorted(t,n.time),n.key_count,n.od,gp,amount,held,t-t[cut][sec],sec,w,half)


@njit(cache=True)
def forward(x,t,key,v,motion,compact,diag=False):
    out=np.empty(len(x))
    z=np.empty((len(x),11)) if diag else np.empty((0,0))
    sp=v[I_S_P]; invsp=1./sp
    cp=v[I_A_P_4K] if key==4 else v[I_A_P]
    cr=v[I_A_R_4K] if key==4 else v[I_A_R]
    cc=v[I_A_CB_4K] if key==4 else v[I_A_CB]
    ch=v[I_CHURN]*(1.17274 if key==4 else 1.)
    ag=v[I_HOLDAGE]*(.54386 if key==4 else 1.)
    hmix=(v[I_HC_E]-1.)/1.2
    E=0.;prev=0.
    for j in range(len(x)):
        row=x[j]
        A=max(row[10],1e-4); logA=np.log(A); invK=1./row[11]
        aj=np.exp(v[I_AJ]*invK*logA)
        jm_raw=max(row[0]+(v[I_J_TRIPLE_4K]*row[28] if key==4 else 0.),0.)
        # Separate context from the actual hand/role-adjusted jack work.
        jm=max(jm_raw+row[1]+v[I_ROLE_PRICE]*row[16],0.)
        jb=aj*min(jm,v[I_CAP_A]+v[I_CAP_B]*jm)
        jb_ctx=aj*min(jm_raw,v[I_CAP_A]+v[I_CAP_B]*jm_raw)
        ja=aj*min(row[2],v[I_CAP_A]+v[I_CAP_B]*row[2])
        pp=row[3]+v[I_GA_ALPHA]*((1.-v[I_GA_SHAPE])*row[4]+v[I_GA_SHAPE]*row[5])+v[I_P_BURST3]*row[6]
        C=max(row[13],0.)
        R=row[12]*(C/(C+v[I_LN_GATE]) if C>0. else 0.)
        hc=(1.-hmix)*row[14]+hmix*row[15]
        cb=ch*row[17]+ag*row[18]+v[I_XH_HOLDTAP]*row[19]+v[I_SH_HOLDTAP_EXTRA]*row[20]+v[I_LOCKSTACK]*row[21]
        if not compact:
            cb+=v[I_LOCKANCHOR]*row[22]+v[I_SHIELD]*row[23]+v[I_OVERLAP]*row[24]
        stream=np.exp(v[I_AP]*logA)*(cp*pp+cr*R/(C+v[I_A_C])+cc*cb/(C+v[I_A_CBC])+v[I_D_HC]*hc)
        S=np.exp(invsp*np.log(max(v[I_C_JM]*jb**sp+v[I_C_JA]*ja**sp+v[I_C_S]*stream**sp, 1e-30)))
        X=max(v[I_X_BASE]*row[7]+v[I_X_CONTEXT]*row[9],0.)+v[I_X_FAST]*row[8]
        if motion:
            X+=v[I_RHYTHM]*row[25]+v[I_ROLL]*row[26]+v[I_REVERSAL]*row[27]
        kj=v[I_K_JT]*jb_ctx
        x2=X*X
        T=np.exp(v[I_AT]*invK*logA)*(X/(X+S+v[I_T_S_OFF]))*(x2/(x2+kj*kj+1e-12))
        st=np.exp(v[I_D_DS]*np.log(S)+v[I_D_DT]*np.log(T)) if (S>0. and T>0.) else 0.
        base=v[I_D_B1]*st+v[I_D_B2]*S+v[I_HOLD_FLOOR]*row[-1]
        if j:
            dt=t[j]-t[j-1]; tau=v[I_TAU_STAM]
            if dt<=440.:
                E=E*np.exp(-dt/tau)+max(0.,prev-v[I_STAM_THETA])*(-np.expm1(-dt/tau))
            else:
                E=E*np.exp(-dt/tau)+max(0.,prev-v[I_STAM_THETA])*(-np.expm1(-440./tau))*np.exp(-(dt-440.)/tau)
        D=base*(1.+v[I_C_STAM]*E/(E+v[I_STAM_HALF]))
        out[j]=D;prev=base
        if diag:
            z[j,0]=jb;z[j,1]=ja;z[j,2]=pp;z[j,3]=X;z[j,4]=A;z[j,5]=R
            z[j,6]=cb;z[j,7]=row[16]*v[I_ROLE_PRICE];z[j,8]=stream;z[j,9]=E;z[j,10]=D
    return out,z


@njit(cache=True)
def field(a,b,t,key,v,motion,compact,diag=False):
    x,z=forward(a,t,key,v,motion,compact,diag)
    if not compact:
        y,w=forward(b,t,key,v,motion,compact,diag)
        x=.5*(x+y)
        if diag:
            z=.5*(z+w)
    return x,z


@njit(cache=True)
def layout(elapsed,section,gp,width,offset=0.):
    n=len(elapsed); points=np.empty(n); slots=np.empty(n,np.int64)
    last=0; base=0; prev=-1
    for j in range(n):
        if section[j]!=prev:
            base=last+2 if j else 0
        shift=elapsed[j]+offset*width
        k=int(shift//width)
        last=base+k;prev=section[j];slots[j]=last
        points[j]=min(width,max(0.,(k+1)*width-shift))
    return points,slots,last+2


@njit(cache=True)
def aggregate(d,gp,amount,held,points,slots,bins,width,half,heads,v):
    e=envelope(d,half)
    mass=np.empty(len(d));weighted=np.empty(len(d))
    mp=v[I_MASS_POW]; gp_p=v[I_GAP_POW]; hm=v[I_HOLD_MASS]; envw=v[I_ENV_W]; ap1=v[I_AGG_P1]
    for j in range(len(d)):
        q=min(max(gp[j],0.),387.12377)
        if q<=0.:
            mass[j]=0.;weighted[j]=0.;continue
        c0=max(amount[j],1e-6); c2=c0*c0; gate=c2/(c2+4.)
        w=gate*np.exp(mp*np.log(c0)+gp_p*np.log(q)-(hm*held[j]/(held[j]+51.9327) if held[j]>0. else 0.))
        bd=(1.-envw)*d[j]+envw*e[j]
        mass[j]=w;weighted[j]=w*np.exp(ap1*np.log(bd)) if bd>0. else 0.
    good=mass>0.
    if not np.any(good):
        raw=.50*max(d)
    else:
        src=slots[good]
        raw=integrate(points[good],src,src+1,mass[good],weighted[good],bins,width,v[I_AGG_P1],v[I_AGG_P2])
    th=v[I_PP_THR]
    if raw>th:
        raw=th+(raw-th)/v[I_PP_DIV]
    return (raw*v[I_PP_SCALE]+v[I_PP_BIAS])*(1.-.25*np.exp(-float(heads)/40.))

def parameter_vector(active,fixed):
    if set(active)&set(fixed) or set(active)|set(fixed)!=set(NAMES):
        raise ValueError("v1.3.0 active and fixed coordinates must partition the rate vector.")
    v=np.array([active[k] if k in active else fixed[k] for k in NAMES],dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.any(v<0.):
        raise ValueError("v1.3.0 parameter prices must be finite and nonnegative.")
    return v


def check_cancel(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("计算已停止。")


def physical_notes(notes):
    if notes is None or notes.n_notes<2:
        raise ValueError("A chart requires at least two physical heads.")
    if notes.key_count not in (4,6,7):
        raise ValueError("This version supports 4K, 6K and 7K.")
    if not np.all(np.isfinite(notes.time)) or not np.all(np.isfinite(notes.ln_end)):
        raise ValueError("Finite note times required.")
    out=role.canonical(notes)
    if out.n_notes<2:
        raise ValueError("A chart requires at least two physical heads.")
    return out


def segment_limits(start_ms,end_ms,scope):
    if scope not in ("context","isolated"):
        raise ValueError("Choose context or isolated.")
    start=float(start_ms) if start_ms is not None else -math.inf
    end=float(end_ms) if end_ms is not None else math.inf
    if not start<end:
        raise ValueError("Use a nonempty half-open interval [start,end).")
    return start,end,start_ms is not None or end_ms is not None


def crop(notes,start=-math.inf,end=math.inf,*,mask=None):
    take=(notes.time>=start)&(notes.time<end) if mask is None else mask
    if np.count_nonzero(take)<2:
        raise ValueError("A rated segment requires at least two physical heads; headless holding has no validated scalar output.")
    if np.all(take) and not np.any(notes.is_ln&(notes.ln_end>end)):
        return notes
    times=notes.time[take].copy()
    col=notes.col[take].copy()
    e=np.minimum(notes.ln_end[take],end)
    ln=notes.is_ln[take]&(e>times)
    return replace(notes,time=times,col=col,ln_end=e,is_ln=ln,n_notes=len(times),
                   col_idx=tuple(np.flatnonzero(col==k) for k in range(7)),
                   duration=float(times[-1]-times[0]))


def query_grid(notes):
    """Distinct physical head times plus the sole static-sustain support row."""
    t=np.unique(notes.time)
    if len(t)==1 and notes.n_ln:
        span=float(np.max(notes.ln_end)-t[0])
        if span>0.:
            t=np.r_[t,t[0]+.5*span]
    return t


def predict_pair(notes,rated,model,*,scope,include_curve,cancel):
    """Aggregate a native child with its own weights, clock phases chosen by API."""
    if model.revision==2:
        from .v130_rev2 import predict_pair as score_rev2
        return score_rev2(notes,rated,model,scope=scope,include_curve=include_curve,cancel=cancel)
    check_cancel(cancel)
    v=parameter_vector(model.parameters,model.fixed_parameters)
    own=(scope=="isolated" or notes is rated)
    full=state(rated if own else notes,model.compact)
    check_cancel(cancel)
    d,_=field(full.a,full.b,full.t,full.key,v,model.motion,model.compact)
    s=full
    if not own:
        check_cancel(cancel)
        s=state(rated,model.compact)
        heads=s.t[s.head_pos]
        j=np.searchsorted(full.t,heads)
        if np.any(j>=len(full.t)) or not np.array_equal(full.t[j],heads):
            raise RuntimeError("Context query identity mismatch.")
        positions=np.searchsorted(full.t,s.t)
        if np.all(positions<len(full.t)) and np.array_equal(full.t[positions],s.t):
            d=d[positions]
        else:
            # A cropped one-row chord may create its own virtual sustain row.
            # Real selected heads still use exact matches as checked above.
            d=np.interp(s.t,full.t,d)
    phases,slots,bins=layout(s.interval,s.section,s.gp,s.win_ms)
    check_cancel(cancel)
    raw=aggregate(d,s.gp,s.amount,s.held,phases,slots,bins,s.win_ms,s.env_half,len(s.head_pos),v)
    result={"raw_score":float(raw)}
    if include_curve:
        result.update(query=s.t,field=d)
    return result


def predict_notes(notes,model,*,start_ms=None,end_ms=None,scope="context",include_curve=False,cancel=None):
    """Score one native chart or segment, without reference or clock extension."""
    check_cancel(cancel)
    notes=physical_notes(notes)
    start,end,segment=segment_limits(start_ms,end_ms,scope)
    rated=crop(notes,start,end) if segment else notes
    return predict_pair(notes,rated,model,scope=scope,include_curve=include_curve,cancel=cancel)

