"""Static v1.3.0-rev2 native row and physical Sigmoid engine."""
from dataclasses import dataclass
import math
import numpy as np
from numba import njit
from . import v130_engine as base
from . import engine, hand, hand_cost, role
from .engine import mirror
from ._core import model as M
from ._core.components import hb
from ._core.components.lncoord_v2 import CbarV2

JACK=base.JACK
JACK_TRIP=base.JACK_TRIP
P_TRIP=base.P_TRIP
PRESSURE=base.PRESSURE
SHAPE1=base.SHAPE1
SHAPE2=base.SHAPE2
R_POSITIVE=base.R_POSITIVE


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
 'reversal',
 'k_jt_4k',
 'straddle',
 'short_attenuation')
DEFAULTS = {'c_jm': 0.35642,
 'c_ja': 0.025,
 'c_s': 0.28483,
 'aj': 1.43,
 'cap_a': 22.111,
 'cap_b': 0.588,
 's_p': 1.196,
 'a_p': 0.402,
 'a_p_4k': 0.734,
 'a_r': 85.0,
 'a_r_4k': 50.0,
 'a_c': 25.84,
 'a_cb': 2.2,
 'a_cb_4k': 5.5,
 'a_cbc': 8.0,
 'ap': 3.54,
 'at': 13.0,
 'd_hc': 0.8,
 'hc_e': 1.7049,
 'ga_alpha': 0.06,
 'ga_shape': 0.42,
 'p_burst3': 0.143,
 'x_base': 1.0,
 'x_fast': 1.2,
 'x_context': 0.65,
 'k_jt': 0.45,
 't_s_off': 3.018,
 'd_b1': 1.252,
 'd_ds': 0.58512,
 'd_dt': 0.85,
 'd_b2': 0.62764,
 'role_price': 0.10220147228915628,
 'hold_floor': 0.095,
 'ln_gate': 0.15,
 'churn': 2.2,
 'holdage': 10.0,
 'agg_p1': 2.51614,
 'agg_p2': 4.42661,
 'mass_pow': 2.55072,
 'gap_pow': 1.29092,
 'hold_mass': 4.83,
 'env_w': 0.54013,
 'j_triple_4k': 0.95,
 'pp_scale': 0.9744998701313351,
 'pp_thr': 8.416,
 'pp_div': 1.56,
 'pp_bias': 0.6824,
 'c_stam': 0.045,
 'tau_stam': 17000.0,
 'stam_theta': 1.0,
 'stam_half': 2.8,
 'xh_holdtap': 0.022,
 'sh_holdtap_extra': 0.25,
 'lockstack': 0.15,
 'lockanchor': 0.15,
 'shield': 0.045,
 'overlap': 0.0,
 'rhythm': 0.16,
 'roll': 0.22,
 'reversal': 0.14,
 'k_jt_4k': 0.45,
 'straddle': 0.16,
 'short_attenuation': 0.25}
BOUNDS = {'c_jm': (0.22, 0.53),
 'c_ja': (0.00015, 0.15),
 'c_s': (0.16, 0.46),
 'aj': (0.0, 3.0),
 's_p': (1.02, 1.46),
 'a_p': (0.16, 0.81),
 'a_p_4k': (0.24, 1.3),
 'a_r': (15.0, 180.0),
 'a_r_4k': (10.0, 120.0),
 'a_c': (13.0, 48.0),
 'a_cb': (0.15, 12.0),
 'a_cb_4k': (0.25, 24.0),
 'ap': (0.0, 4.5),
 'at': (0.0, 34.0),
 'd_hc': (0.0002, 1.8),
 'ga_alpha': (0.0008, 0.36),
 'ga_shape': (0.1, 0.9),
 'p_burst3': (0.0, 0.35),
 'x_base': (0.35, 1.4),
 'x_fast': (0.3, 2.5),
 'k_jt': (0.05, 2.5),
 't_s_off': (1.2, 6.6),
 'd_b1': (0.65, 1.8),
 'd_ds': (0.4, 0.9),
 'd_dt': (0.45, 1.45),
 'd_b2': (0.41, 0.94),
 'role_price': (0.045, 0.45),
 'ln_gate': (0.0, 1.2),
 'churn': (0.1, 9.7),
 'holdage': (0.5, 32.0),
 'mass_pow': (0.95, 4.1),
 'gap_pow': (0.7, 1.55),
 'hold_mass': (0.0, 9.0),
 'env_w': (0.1, 0.84),
 'j_triple_4k': (0.0, 3.6),
 'pp_thr': (6.3, 12.0),
 'pp_div': (1.18, 1.98),
 'c_stam': (0.005, 0.16),
 'tau_stam': (7000.0, 45000.0),
 'stam_theta': (0.25, 4.5),
 'stam_half': (0.7, 4.6),
 'xh_holdtap': (0.001, 0.5),
 'sh_holdtap_extra': (0.0008, 1.5),
 'lockstack': (0.005, 2.5),
 'lockanchor': (0.005, 2.5),
 'shield': (0.001, 1.5),
 'k_jt_4k': (0.035, 2.5),
 'straddle': (0.002, 1.95),
 'overlap': (0.0002, 0.15),
 'sig_beta': (3.0, 15.0),
 'sig_gamma': (0.71, 0.995),
 'ln_gate_4k': (0.065, 26.0),
 'r_straddle': (0.025, 0.95),
 'r_order_pen': (0.008, 0.6),
 'r_I_steep': (3.0, 15.0),
 'r_sim_tau': (2.0, 45.0),
 'w_r': (1065.1598082764933, 3649.158602428727),
 'cbv_w': (190.0, 970.0),
 'cbv_ls_sh_adj': (0.6856319469407989, 2.3489242626675515),
 'cbv_ls_sh_split': (0.9730987496814035, 3.333764235019623),
 'cbv_ls_1h_split': (0.4805767412238061, 1.646420317155632),
 'cbv_ls_2h_split': (0.20592030294807218, 0.7054677045443214),
 'cbv_ls_cross': (0.15349603112274351, 0.5258660325501398),
 'cbv_str_sh_split': (0.54, 1.85),
 'cbv_str_1h_split': (0.3186, 1.0915),
 'cbv_str_2h_split': (0.1998, 0.6845),
 'cbv_str_cross': (0.10800000000000001, 0.37000000000000005),
 'cbv_str_h1': (0.54, 1.85),
 'cbv_str_h2': (0.5940000000000001, 2.035),
 'cbv_str_h3': (0.5670000000000001, 1.9425000000000001),
 'cbv_str_bal': (0.8, 4.0),
 'cbv_sh_lock': (0.05, 1.5),
 'w_jm': (504.49220566365454, 1728.3529268106681),
 'w_p': (487.38678786953295, 1669.7510325159924),
 'w_x': (501.08836439187166, 1716.6916187499307),
 'shd_w': (543.9134671530713, 1863.4072485799668),
 'abar_scale': (0.89, 1.13),
 'env_ms': (110.0, 500.0)}
ACTIVE = {'detail': ('c_jm',
            'c_ja',
            'c_s',
            'aj',
            's_p',
            'a_p',
            'a_p_4k',
            'a_r',
            'a_r_4k',
            'a_c',
            'a_cb',
            'a_cb_4k',
            'ap',
            'at',
            'd_hc',
            'ga_alpha',
            'ga_shape',
            'p_burst3',
            'x_base',
            'x_fast',
            'k_jt',
            't_s_off',
            'd_b1',
            'd_ds',
            'd_dt',
            'd_b2',
            'role_price',
            'ln_gate',
            'churn',
            'holdage',
            'mass_pow',
            'gap_pow',
            'hold_mass',
            'env_w',
            'j_triple_4k',
            'pp_thr',
            'pp_div',
            'c_stam',
            'tau_stam',
            'stam_theta',
            'stam_half',
            'xh_holdtap',
            'sh_holdtap_extra',
            'lockstack',
            'lockanchor',
            'shield',
            'k_jt_4k',
            'straddle',
            'overlap',
            'sig_beta',
            'sig_gamma',
            'ln_gate_4k',
            'r_straddle',
            'r_order_pen',
            'r_I_steep',
            'r_sim_tau',
            'w_r',
            'cbv_w',
            'cbv_ls_sh_adj',
            'cbv_ls_sh_split',
            'cbv_ls_1h_split',
            'cbv_ls_2h_split',
            'cbv_ls_cross',
            'cbv_str_sh_split',
            'cbv_str_1h_split',
            'cbv_str_2h_split',
            'cbv_str_cross',
            'cbv_str_h1',
            'cbv_str_h2',
            'cbv_str_h3',
            'cbv_str_bal',
            'cbv_sh_lock',
            'w_jm',
            'w_p',
            'w_x',
            'shd_w',
            'abar_scale',
            'env_ms'),
 'compact': ('c_jm',
             'c_ja',
             'c_s',
             'aj',
             's_p',
             'a_p',
             'a_p_4k',
             'a_r',
             'a_r_4k',
             'a_c',
             'a_cb',
             'a_cb_4k',
             'ap',
             'at',
             'd_hc',
             'ga_alpha',
             'p_burst3',
             'x_base',
             'x_fast',
             'k_jt',
             't_s_off',
             'd_b1',
             'd_ds',
             'd_dt',
             'd_b2',
             'ln_gate',
             'churn',
             'holdage',
             'mass_pow',
             'hold_mass',
             'j_triple_4k',
             'c_stam',
             'tau_stam',
             'xh_holdtap',
             'sh_holdtap_extra',
             'lockstack',
             'k_jt_4k',
             'straddle',
             'overlap',
             'gap_pow',
             'env_w',
             'sig_beta',
             'sig_gamma',
             'ln_gate_4k',
             'r_straddle',
             'r_order_pen',
             'r_I_steep',
             'r_sim_tau',
             'w_r',
             'cbv_w',
             'cbv_ls_sh_adj',
             'cbv_ls_sh_split',
             'cbv_ls_1h_split',
             'cbv_ls_2h_split',
             'cbv_ls_cross',
             'cbv_str_sh_split',
             'cbv_str_1h_split',
             'cbv_str_2h_split',
             'cbv_str_cross',
             'cbv_str_h1',
             'cbv_str_h2',
             'cbv_str_h3',
             'cbv_str_bal',
             'cbv_sh_lock',
             'w_jm',
             'w_p',
             'w_x',
             'shd_w',
             'abar_scale',
             'env_ms')}
FLAGS = {'detail': {'compact': False,
            'hill': 3,
            'release': 'positive',
            'coord': 'full',
            'pure': True,
            'xsoft': False,
            'ln_gate': True,
            'ln_mass': False,
            'prune_context': True},
 'compact': {'compact': True,
             'hill': 3,
             'release': 'positive',
             'coord': 'full',
             'pure': True,
             'xsoft': False,
             'ln_gate': True,
             'ln_mass': False,
             'prune_context': True}}
PHYS = {'od_mult_jm': 1.2040048799564311,
 'od_mult_jc': -0.293176786084059,
 'od_mult_ja': -1.0,
 'od_mult_p': 1.1286927800588342,
 'od_mult_x': 1.1974891491877375,
 'od_mult_r': 1.058858405816053,
 'w_jm': 934.2448253030639,
 'w_jc': -0.4552854599596368,
 'w_ja': 0.0,
 'w_p': 902.5681256843202,
 'w_x': 927.941415540503,
 'w_r': 1972.5181634749874,
 'w_cb': 1716.856156378797,
 'w_a': 75.76213228110679,
 'j_c1': 0.16126641911276896,
 'j_nerf_a': 0.0,
 'j_nerf_off': 0.09265897011180577,
 'j_nerf_c': 0.0924591760779166,
 'j_agg_pow': 2.8052526777466564,
 'j_triple_tau': 140.9631484638465,
 'j_triple_pow': 2.0072942534242566,
 'j_triple_gain': 0.0,
 'j_triple_gain_4k': 0.0,
 'c_fj': 1.291464498859916,
 'c_fj_4k': 1.1891597740767534,
 'fj_bounded': 0.0,
 'd_sh': 0.0,
 'shd_w': 1007.2471613945766,
 'd_hc': 1.0188595536994098,
 'hc_e': 1.7049026540688783,
 'mj_dilute': -0.022067037291399555,
 'mj_dilute_exp': 1.7147578892981583,
 'cj_size_exp': 2.3090892612901857,
 'cj_norm': 3.9223792605276984,
 'ja_gap_thr': 250.97771855895374,
 'ja_len_exp': 0.873949786346038,
 'ja_coord_exp': 0.012636794260452278,
 'ja_len_floor': 1.0,
 'ja_norm': 4.912805983390194,
 'p_lam3': 60.0,
 'p_scale': 0.0573287291354225,
 'p_lam2': 0.005077940680001312,
 'p_v_norm': -0.03652823337937215,
 'p_boost': -1e-05,
 'p_boost_lo': 117.67575960429505,
 'p_boost_hi': 296.9197535881636,
 'p_bv_max': 0.7121791011599433,
 'p_sat_a': 12.967278061527875,
 'p_sat_b': 2.3046546300449706,
 'p_chw2': 0.6997350782823039,
 'p_chw3': 1.4710731528472405,
 'p_chw4': 0.82149995851254,
 'p_chw5': 0.8676070127257947,
 'p_chw6': 0.7474127532548505,
 'p_chw7': 1.03284160083206,
 'chord_tau_c': 24.345887557552615,
 'note_chord': 1.0,
 'chord_soft_att': 1.0,
 'chord_no_dilute': 1.0,
 'chord2_hard': 1.0,
 'ga_alpha': 0.0,
 'ga_gam': 1.0,
 'ga_q': 1.0,
 'ga_pw': 1.0,
 'p_burst3': 0.0,
 'p_burst4': 0.0,
 'an_a0': 0.20919909312275142,
 'an_a1': 0.37286666751258996,
 'an_cubic': 8.862618779101767,
 'an_on_p': -0.32562582197542633,
 'abar_scale': 0.9908465677870455,
 'abar_active': 0.02959105293159399,
 'abar_thr_lo': 0.01389142404428909,
 'abar_thr_hi': 0.11599034279776298,
 'abar_c0': 0.6575435661594596,
 'abar_c1': 1.3302402303785177,
 'abar_k': 5.052287546779305,
 'abar_mx': 0.451610888785001,
 'abar_mx_thr': 0.05208686478341369,
 'abar_mx_w': 0.4754900770279769,
 'abar_weld': 1.0,
 'x_amp': 0.09636644976733964,
 'x_cw0': 0.03724653823590479,
 'x_cw1': 0.8550132389171754,
 'x_cw2': 0.4722195790253056,
 'x_cw3': 0.023511327398546056,
 'x_din': 0.21329641269001137,
 'x_dout': 0.3884059305040078,
 'x_fc_a': 0.5315844156004722,
 'x_fc_off': 62.51708153410152,
 'x_fc_floor': 0.07402097926631954,
 'x_fc_w': 1.68396911611816,
 'x_jump_w': 0.0,
 'x_jh_p': 0.761399733967284,
 'x_jh_e': 0.7460102426771034,
 'x_jt': 0.5379957175084137,
 'x_jd_p': 0.038157319982258006,
 'x_jd_e': 2.070662416719641,
 'x_dir_in': 0.7935947499263032,
 'x_dir_out': -0.46052690940589464,
 'r_tail': 0.030479803326367228,
 'r_seq': 0.06635864238365655,
 'r_I_steep': 7.138399583260547,
 'r_I_off': 1.6936500452160337,
 'r_I_w': 1.3201531009318856,
 'r_dt_max': 6749.562099456267,
 'r_dtr_min': 0.004050313466366775,
 'r_sim_tau': 11.405453069087805,
 'r_soft_edges': 1.0,
 'r_same_col': 0.19578505600824977,
 'r_coord_e': 2.2703135423553786,
 'r_tt': 0.0,
 'r_short_thr': 232.63366268425537,
 'r_short_red': 0.0,
 'r_lock': 2.064385891165511,
 'r_simul': 0.48507152260049274,
 'r_cw_same': 1.5375480228960465,
 'r_cw_hand': 0.46562083856474745,
 'r_cw_thumb': 0.011960920198257674,
 'r_cw_cross': 0.7315226193212329,
 'r_order_pen': 0.23401924932585802,
 'r_order_tau': 702.5373743325986,
 'r_straddle': 0.4485333102831889,
 'r_ain0': 0.0,
 'r_ain1': 2.466978961221889,
 'r_ain2': 2.5020134256224296,
 'r_ain3': 1.2702424995871497,
 'cb_shield': 0.0,
 'cb_shield_tau': 56.24,
 'cb_shield_lock': 0.8062,
 'cb_lockjack': 0.0,
 'cb_lock_pow': 1.0,
 'cb_lockanchor': 0.0,
 'cb_par': 0.0,
 'cb_holdtap': 0.0,
 'cb_holdtap_pow': 1.0,
 'use_cbar_v2': 1.0,
 'churn_event_rate': 0.0,
 'cbv_w': 421.0782656735729,
 'use_cbv_shield': 1.0,
 'cbv_shield': 0.02,
 'use_cbv_straddle': 1.0,
 'cbv_straddle': 9.213308770964225,
 'use_cbv_lockstack': 1.0,
 'cbv_lockstack': 1.4797072901013206,
 'use_cbv_lockanchor': 1.0,
 'cbv_lockanchor': 1.4544619305858701,
 'use_cbv_overlap': 1.0,
 'cbv_overlap': 0.02,
 'use_cbv_holdtap': 1.0,
 'cbv_holdtap': 0.02,
 'cbv_sh_tau': 498.1744137671231,
 'cbv_sh_dt_max': 330.6326350747474,
 'cbv_sh_ln_w': 3.0,
 'cbv_sh_ln_tau': 387.1884382046885,
 'cbv_sh_lock': 0.24,
 'cbv_str_sh_adj': 0.02,
 'cbv_str_sh_split': 1.0,
 'cbv_str_1h_split': 0.59,
 'cbv_str_2h_split': 0.37,
 'cbv_str_cross': 0.2,
 'cbv_str_h0': 0.3224021808746196,
 'cbv_str_h1': 1.0,
 'cbv_str_h2': 1.1,
 'cbv_str_h3': 1.05,
 'cbv_str_bal': 2.0,
 'cbv_str_maxd': 6.0,
 'cbv_lock_pow': 0.9791217299061639,
 'cbv_ls_sh_adj': 1.269688790631109,
 'cbv_ls_sh_split': 1.8020347216322286,
 'cbv_ls_1h_split': 0.8899569281922335,
 'cbv_ls_2h_split': 0.3813338943482818,
 'cbv_ls_cross': 0.28425190948656204,
 'cbv_ov_sh_adj': 0.6949297766284019,
 'cbv_ov_sh_split': 0.02,
 'cbv_ov_1h_split': 0.02,
 'cbv_ov_2h_split': 1.5020006436255016,
 'cbv_ov_cross': 4.935679036129201,
 'cbv_ov_dt_max': 2276.1965593534787,
 'cbv_ov_cap': 3446.987025249341,
 'use_hb': 1.0,
 'cbv_churn': 3.6654074897742612,
 'cbv_churn_4k': 4.298621698858253,
 'churn_sat': 20.70743322086763,
 'churn_w': 1167.6875064558421,
 'cbv_holdage': 18.945855743518063,
 'cbv_holdage_4k': 10.303537336083915,
 'holdage_th': 0.15716538407588404,
 'holdage_cap': 0.42468477879654104,
 'c_chord2': 0.0,
 'chord2_w': 998.3068456460403,
 'd_rec': 0.0,
 'rec_half': 0.3671841225596574,
 'recov_w': 968.8229953754064,
 'recov_wl': 38307.95803840174,
 'aj': 1.430584921574431,
 'ap': 3.5424464326854332,
 'at': 12.654197587970756,
 'cap_a': 22.11098017221394,
 'cap_b': 0.5877082264222335,
 'c_jm': 0.356418909837854,
 'c_jc': 0.0,
 'c_ja': 0.038468075195409575,
 'c_s': 0.2848251338028861,
 's_p': 1.1959916923340246,
 'a_p': 0.4016531544080434,
 'a_r': 108.50536487992183,
 'a_c': 25.83867288316744,
 'a_cb': 4.234752287195439,
 'a_cbc': 2.506126891341002,
 'a_p_4k': 0.7336548088255943,
 'a_r_4k': 58.25324080278074,
 'a_cb_4k': 11.60113975415644,
 't_jack_mix': -0.3805743609330355,
 't_s_off': 3.0179360184812842,
 'd_b1': 1.2519368453250523,
 'd_ds': 0.5851191625546722,
 'd_dt': 0.7540471413982772,
 'd_b2': 0.6276354594810581,
 'c_stam': 0.0,
 'stam_tau': 8000.0,
 'stam_grp': 30.0,
 'agg_mode': 4.0,
 'agg_k': 3.923290274033005,
 'agg_C': 5.7765188650680015,
 'agg_gamma': 0.3142734186203886,
 'agg_nseg': 30.0,
 'agg_wc': 1.145022164188944,
 'agg_gap_w': 1.0873927743990874,
 'aggw_src': 0.0,
 'aggw_gap_cap': 0.0,
 'aggC_ms': 4397.439342626505,
 'aggC_gap': 2270.295764558157,
 'aggC_ms_4k': 3078.207539838553,
 'aggC_gap_4k': 1222.4669501466997,
 'aggC_pw1': 2.516135362334202,
 'aggC_pw2': 4.4266132719189,
 'aggC_wc': 2.5507155298795627,
 'aggC_gw': 1.2909164079307989,
 'aggC_gc': 387.1237698484442,
 'aggC_hw': 1.0,
 'aggC_tau': 4027.709861084976,
 'aggC_kap': 0.8250936661252044,
 'env_w': 0.5401340304888701,
 'env_ms': 296.6324480607108,
 'p93_c': 2.505172683584436,
 'w93': 0.3605157897072774,
 'p83_c': 2.667686959362696,
 'w83': 0.4815139944602175,
 'wmean': 2.1613377064577026,
 'mean_pow': 6.465267031930503,
 'pp_n0': 7.6,
 'pp_thr': 8.416405509249257,
 'pp_div': 1.559992046815954,
 'pp_scale': 0.9866220368284715,
 'calib_a': 0.9877134644832144,
 'calib_b': 0.6823998306913243,
 'hw_a': -4.833333333333334,
 'hw_half': 51.932695284873475,
 'ln_eff_tail_ms': 11.348134111824663,
 'ln_note_level': 0.0,
 'use_read': 0.0,
 'd_read': 0.0,
 'd_read_4k': 0.0,
 'read_tol': 0.18,
 'read_n_win': 16.0,
 'read_clip_lo': 0.5,
 'read_clip_hi': 1.5,
 'use_eye': 0.0,
 'd_eye': 0.0,
 'd_eye_4k': 0.0,
 'eye_w_ref': 4000.0,
 'eye_w_s': 500.0,
 'eye_tau': 0.2,
 'out_a': 0.8686939253627315,
 'out_b': -0.9591165705858606,
 'hand_beta': 0.0,
 'hand_power': 1.0,
 'hand_sync': 30.0,
 'hand_lookback': 1200.0,
 'hand_predecessor': 6.0,
 'hand_tau': 400.0,
 'hand_floor': 0.0,
 'hand_null': 0.1,
 'hand_projection': 500.0,
 'hand_average': 1.0,
 'group_jump': 0.0,
 'hand_entry': 1.0,
 'x_adj_w': 0.045,
 'jc_scale': 1.4332747701403683,
 'ja_scale': 0.9792209480655096}
UP = {'r_straddle': (0.4485333102831889, 0.025, 0.95),
 'r_order_pen': (0.23401924932585802, 0.008, 0.6),
 'r_I_steep': (7.138399583260547, 3.0, 15.0),
 'r_sim_tau': (11.405453069087805, 2.0, 45.0),
 'w_r': (1972.5181634749874, 1065.1598082764933, 3649.158602428727),
 'cbv_w': (421.0782656735729, 190.0, 970.0),
 'cbv_ls_sh_adj': (1.269688790631109, 0.6856319469407989, 2.3489242626675515),
 'cbv_ls_sh_split': (1.8020347216322286, 0.9730987496814035, 3.333764235019623),
 'cbv_ls_1h_split': (0.8899569281922335, 0.4805767412238061, 1.646420317155632),
 'cbv_ls_2h_split': (0.3813338943482818,
                     0.20592030294807218,
                     0.7054677045443214),
 'cbv_ls_cross': (0.28425190948656204, 0.15349603112274351, 0.5258660325501398),
 'cbv_str_sh_split': (1.0, 0.54, 1.85),
 'cbv_str_1h_split': (0.59, 0.3186, 1.0915),
 'cbv_str_2h_split': (0.37, 0.1998, 0.6845),
 'cbv_str_cross': (0.2, 0.10800000000000001, 0.37000000000000005),
 'cbv_str_h1': (1.0, 0.54, 1.85),
 'cbv_str_h2': (1.1, 0.5940000000000001, 2.035),
 'cbv_str_h3': (1.05, 0.5670000000000001, 1.9425000000000001),
 'cbv_str_bal': (2.0, 0.8, 4.0),
 'cbv_sh_lock': (0.24, 0.05, 1.5),
 'w_jm': (934.2448253030639, 504.49220566365454, 1728.3529268106681),
 'w_p': (902.5681256843202, 487.38678786953295, 1669.7510325159924),
 'w_x': (927.941415540503, 501.08836439187166, 1716.6916187499307),
 'x_adj_w': (0.045, 0.001, 0.52),
 'shd_w': (1007.2471613945766, 543.9134671530713, 1863.4072485799668),
 'abar_scale': (0.9908465677870455, 0.89, 1.13),
 'aggC_ms': (4397.439342626505, 1600.0, 7000.0),
 'aggC_gap': (2270.295764558157, 650.0, 3300.0),
 'env_ms': (296.6324480607108, 110.0, 500.0),
 'pp_n0': (7.6, 0.02, 14.0)}
U_COLS = {'U1': (12,),
 'U2': (19, 20, 21, 22, 23, 24, 25),
 'U3': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 28)}
FIXED_PP_N0 = 7.6


for _i,_name in enumerate(NAMES):
    globals()["I_"+_name.upper()]=_i

def variant(compact):
    return Variant(**FLAGS["compact" if compact else "detail"])

@dataclass(frozen=True)
class Variant:
    compact: bool
    hill: int
    release: str
    coord: str
    pure: bool
    xsoft: bool
    ln_gate: bool
    ln_mass: bool
    prune_context: bool

    def native(self,p):
        if set(p)!=set(ACTIVE["compact" if self.compact else "detail"]):
            raise ValueError("Unexpected rev2 parameter roster")
        if any(not BOUNDS[k][0]<=x<=BOUNDS[k][1] for k,x in p.items()):
            raise ValueError("Rev2 prices must be within their validated bounds")
        v=np.array([p.get(k,DEFAULTS[k]) for k in NAMES],dtype=np.float64)
        v[I_X_CONTEXT]=0.
        v4=v.copy()
        v4[I_HOLD_MASS]=p.get("hold_mass_4k",v[I_HOLD_MASS])
        return v,v4,float(p.get("ln_gate_4k",0.))

    def shape(self,p):
        return p["sig_beta"],p["sig_gamma"],p.get("sig_mix",1.),FIXED_PP_N0


def validate(p,compact,flags):
    if dict(flags)!=FLAGS["compact" if compact else "detail"]:
        raise ValueError("Unsupported v1.3.0-rev2 arm flags")
    variant(compact).native(p)


def predict_pair(notes,rated,model,*,scope,include_curve,cancel):
    return pair_scores(notes,rated,variant(model.compact),model.parameters,
                       scope=scope,include_curve=include_curve,cancel=cancel)


@njit(cache=True)
def forward(x,t,key,v,compact,hill,xsoft,diag=False,ln_support=0.):
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
        cb+=v[I_STRADDLE]*row[24]+v[I_OVERLAP]*row[25]
        if not compact:
            cb+=v[I_LOCKANCHOR]*row[22]+v[I_SHIELD]*row[23]
        if ln_support>0.:
            # Positive pressure-dependent gate on LN excess, not on
            # contact demand, short-chart correction, or pure rice.
            f=C/(C+ln_support)
            R*=f
            cb*=f
        stream=np.exp(v[I_AP]*logA)*(cp*pp+cr*R/(C+v[I_A_C])+cc*cb/(C+v[I_A_CBC])+v[I_D_HC]*hc)
        S=np.exp(invsp*np.log(max(v[I_C_JM]*jb**sp+v[I_C_JA]*ja**sp+v[I_C_S]*stream**sp, 1e-30)))
        xb=max(v[I_X_BASE]*row[7]+v[I_X_CONTEXT]*row[9],0.)
        xf=v[I_X_FAST]*row[8]
        if xsoft:
            xf*=np.sqrt(xb/(xb+0.38))
        X=xb+xf
        kj=(v[I_K_JT_4K] if key==4 else v[I_K_JT])*jb_ctx
        x2=X*X; j2=kj*kj
        if hill==3:
            x2*=X; j2*=kj
        elif hill==4:
            x2*=x2; j2*=j2
        if hill==0:
            T=np.exp(v[I_AT]*invK*logA)*X/(X+S+v[I_T_S_OFF]+kj)
        else:
            T=np.exp(v[I_AT]*invK*logA)*(X/(X+S+v[I_T_S_OFF]))*(x2/(x2+j2+1e-12))
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
def field(a,b,t,key,v,compact,hill,xsoft,diag=False,ln_support=0.):
    x,z=forward(a,t,key,v,compact,hill,xsoft,diag,ln_support)
    if not compact:
        y,w=forward(b,t,key,v,compact,hill,xsoft,diag,ln_support)
        x=.5*(x+y)
        if diag:
            z=.5*(z+w)
    return x,z


@njit(cache=True)
def row_mass(d,gp,amount,held,half,v):
    e=base.envelope(d,half)
    m=np.empty(len(d)); z=np.empty(len(d))
    p=v[I_MASS_POW];g0=v[I_GAP_POW];h=v[I_HOLD_MASS];w=v[I_ENV_W]
    for j in range(len(d)):
        z[j]=(1.-w)*d[j]+w*e[j]
        q=min(max(gp[j],0.),387.12377)
        if q<=0.:
            m[j]=0.
        else:
            c=max(amount[j],1e-6);c2=c*c
            m[j]=(c2/(c2+4.))*np.exp(p*np.log(c)+g0*np.log(q)-
                (h*held[j]/(held[j]+51.9327) if held[j]>0. else 0.))
    return m,z


@njit(cache=True)
def _logistic(x):
    if x>=0.:
        return 1./(1.+math.exp(-x))
    e=math.exp(x)
    return e/(1.+e)


@njit(cache=True)
def relative(d,m,beta,gamma):
    """Log-space safeguarded Newton: F(u)=sum(M*acc(d,u))/sum(M)-gamma."""
    high=0.;low=np.inf;total=0.;center=0.;positive=0.
    for i in range(len(d)):
        if m[i]<=0.:
            continue
        total+=m[i]
        if d[i]>0.:
            high=max(high,d[i]);low=min(low,d[i])
            center+=m[i]*math.log(d[i]);positive+=m[i]
    if positive==0. or total<=0. or total-positive>=gamma*total:
        return 0.
    if high==low and positive==total:
        return high
    center/=positive
    lhigh=math.log(high)
    top=lhigh
    # In the presence of zeros (perfectly recoverable mass), the root can
    # lie below the smallest positive difficulty.
    bottom=math.log(low) if positive==total else math.log(low)-40./beta
    n=len(d)
    a=np.empty(n)
    for i in range(n):
        if m[i]>0.:
            a[i]=(d[i]/high)**beta if d[i]>0. else 0.
    c=(1.-gamma)/gamma
    target=gamma*total
    zero=total-positive
    t=min(top,max(bottom,center))
    for _ in range(80):
        s=math.exp(beta*(t-lhigh))
        f=zero-target; slope=0.
        for j in range(n):
            if m[j]>0. and d[j]>0.:
                den=s+c*a[j]
                acc=s/den if den>0. else _logistic(beta*(t-math.log(d[j]))-math.log(c))
                f+=m[j]*acc
                slope+=m[j]*acc*(1.-acc)
        if abs(f)<=1e-13*total or top-bottom<=1e-11:
            return math.exp(t)
        if f>0.:
            top=t
        else:
            bottom=t
        q=t-f/(beta*slope) if slope>1e-25*total else np.nan
        t=q if bottom<q<top else .5*(bottom+top)
    return math.exp(.5*(bottom+top))


@njit(cache=True)
def phase_order(ph,include):
    return include[np.argsort(ph[include])]


@njit(cache=True)
def phase_term(a,b,p):
    return a*(max(b,0.)/max(a,1e-9))**p if a>0. else 0.


def phase_structure(s):
    ph,slot,bins=base.layout(s.interval,s.section,s.gp,s.win_ms)
    include=np.flatnonzero(s.gp>0)
    # Numba quicksort tie ordering is the reference used by integrate().
    return ph,slot,bins,phase_order(ph,include)


@njit(cache=True)
def pooled_shared(z,m,points,slot,events,bins,width,v):
    # Same piecewise-constant exact M4 integral as stage3_batch.pooled, now
    # starting from its already-computed physical mass/envelope.
    a=np.zeros(bins); b=np.zeros(bins); count=np.zeros(bins,np.int64)
    moment=np.zeros(len(m))
    p1=v[I_AGG_P1]
    for i in range(len(m)):
        w=m[i]
        if w>0.:
            y=w*np.exp(p1*np.log(z[i])) if z[i]>0. else 0.
            moment[i]=y
            j=slot[i]; a[j]+=w; b[j]+=y; count[j]+=1
    total=a.sum()
    exponent=v[I_AGG_P2]/p1; inv=1./v[I_AGG_P2]
    val=0.
    for j in range(bins):
        val+=phase_term(a[j],b[j],exponent)
    area=0.;left=0.
    for i in events:
        right=points[i]
        if right>left:
            area+=(right-left)*(max(val,0.)/total)**inv
            left=right
        j=slot[i]; k=j+1
        val-=phase_term(a[j],b[j],exponent)+phase_term(a[k],b[k],exponent)
        count[j]-=1;count[k]+=1
        a[j]-=m[i];b[j]-=moment[i]
        if count[j]==0:
            a[j]=b[j]=0.
        a[k]+=m[i];b[k]+=moment[i]
        val+=phase_term(a[j],b[j],exponent)+phase_term(a[k],b[k],exponent)
    if width>left:
        area+=(width-left)*(max(val,0.)/total)**inv
    return area/width


@njit(cache=True)
def aggregate(d,gp,amount,held,points,slot,events,bins,width,half,heads,v,beta,gamma,mix,n0):
    m,z=row_mass(d,gp,amount,held,half,v)
    if not np.any(m>0.):
        u=.50*max(d)
    else:
        u=relative(z,m,beta,gamma)
        if mix<1.:
            a=pooled_shared(z,m,points,slot,events,bins,width,v)
            u=mix*u+(1.-mix)*a
    # Before thresholding and calibration, as in RC3. Smooth and monotone
    # in both effective chart length and pooled difficulty.
    u*=heads/(heads+n0)
    if u>v[I_PP_THR]:
        u=v[I_PP_THR]+(u-v[I_PP_THR])/v[I_PP_DIV]
    return u*v[I_PP_SCALE]+v[I_PP_BIAS]


@njit(cache=True)
def geometry_of(t,key,ms,gap,e):
    if key==4:
        ms*=3078.207539838553/4397.439342626505
        gap*=1222.4669501466997/2270.295764558157
    cut=0; sec=np.zeros(len(t),np.int64); elapsed=np.empty(len(t));elapsed[0]=0.
    for j in range(1,len(t)):
        if t[j]-t[j-1]>=gap:
            cut=j
            sec[j]=sec[j-1]+1
        else:
            sec[j]=sec[j-1]
        elapsed[j]=t[j]-t[cut]
    step=float(np.median(np.diff(t))) if len(t)>1 else 1.
    return elapsed,sec,ms,max(1,int(round(.5*e/max(step,1.))))


def upstream(p):
    return {**PHYS,**{k:p.get(k,UP[k][0]) for k in UP}}


def coord(nt,s,p,which,compact):
    cv=CbarV2(s.batch,s,p)
    h0,h1,stack,anchor,shield,_=base.coord_atoms(nt,s,p,compact)
    straddle=CbarV2(s.batch,s,{**p,"cbv_str_maxd":3.})._t_straddle()
    full=cv.curve() if which=="full" else np.zeros(s.n)
    return np.column_stack((h0,h1,stack,anchor,shield,straddle,full))


def extract_block(nt,s,p,model,block):
    b=s.batch
    x=s.x[s.chart]
    z=np.zeros(s.n)
    if block=="U1":
        kernel=R_POSITIVE if model.release=="positive" else M._rbar
        return kernel(b,s,p,x)[:,None] if nt.n_ln else z[:,None]
    if block=="U2":
        return coord(nt,s,p,model.coord,model.compact) if nt.n_ln else np.zeros((s.n,7))
    jm,ja=JACK(b,s,p)
    trip=JACK_TRIP(b,s,{**p,**{k:P_TRIP[k] for k in
                      ("j_triple_gain","j_triple_gain_4k","c_ja","c_jc","c_fj","c_fj_4k")}})
    inc,l,r,tau=PRESSURE(b,s,p,x,s.n)
    w=lambda y:M._wmean(s,y,p["w_p"])
    price=lambda g,q:w(np.maximum(inc*(M._chord_g(l,r,{"ga_alpha":1.,"ga_gam":g,"ga_q":q,"ga_pw":1.})-1.),0.))
    bx,fx=base.boundary(b,s,p,x,tau)
    ctx=z
    hs=hand_cost.state(hand.field(hand.extract(nt,30.),s.t,p["w_jm"]))
    a=base.abar(s,p,s.n)
    hc=lambda e:M._wmean(s,np.maximum(l-1.,0.)**e+np.maximum(r-1.,0.)**e,p["shd_w"])
    return np.column_stack((jm,hand_cost.price(hs,s.x[0],p),ja,w(np.maximum(inc,0.)),
                            price(*SHAPE1),price(*SHAPE2),w(M._wcount(s,s.note_impulse,100.)*10.),
                            bx,fx,ctx,a,hc(1.),hc(2.2),np.maximum(trip-jm,0.)))


def exact_block(nt,s,p,model,block):
    a=extract_block(nt,s,p,model,block)
    if nt.n_ln and len(s.t)==1 and float(np.max(nt.ln_end)-s.t[0])>0.:
        a=np.vstack((.34*a[0],np.zeros(a.shape[1])))
        if block=="U3":
            a[0,10]/=.34
            a[1,10]=1.
    return a


def source_structs(notes,p,model):
    n=role.canonical(notes)
    other=mirror(n)
    if (other.col.tobytes(),other.ln_end.tobytes()) < (n.col.tobytes(),n.ln_end.tobytes()):
        n,other=other,n
    return [(nt,engine.get().build(nt,p)) for nt in
            ((n,) if model.compact else (n,other))]


def state_from_notes(notes,model,p):
    """Native state without the Dev chart or feature registries."""
    phys=upstream(p)
    src=source_structs(notes,phys,model)
    n,s=src[0]
    t=s.t.copy()
    ks=s.Ks.astype(np.float64)
    amount,held=s.C.copy(),s.held.copy()
    gp=.5*(np.r_[0.,s.dt_diff]+np.r_[s.dt_diff,0.])
    d_role=role.field(n,base.P,t)
    static=n.n_ln and len(t)==1 and float(np.max(n.ln_end)-t[0])>0.
    pair=[]
    for nn,ss in src:
        a=np.zeros((len(t)+int(static),len(base.FIELDS)))
        a[:len(t),11]=ks
        a[:len(t),13]=ss.C
        a[:len(t),16]=d_role
        if nn.n_ln:
            a[:len(t),17]=hb.compute_churn(ss.batch,ss,base.P)
            a[:len(t),18]=hb.compute_holdage(ss.batch,ss,base.P)
        if static:
            a[0]*=.34
            a[0,11]=ks[0];a[0,13]=1.2
            a[1,11]=a[1,13]=1.
            a[1,-1]=np.sum(nn.is_ln&(nn.ln_end>t[0]+.5*(np.max(n.ln_end)-t[0])))
        for k,cols in U_COLS.items():
            a[:,cols]=exact_block(nn,ss,phys,model,k)
        pair.append(a)
    if static:
        span=float(np.max(n.ln_end)-t[0]);t=np.r_[t,t[0]+.5*span]
        gp=np.full(2,min(max(span*.5,5.),350.));amount=np.array([1.2,1.])
        held=np.r_[held,pair[0][1,-1]];ks=np.r_[ks,1.]
    el,sec,ms,half=geometry_of(t,n.key_count,p.get("aggC_ms",UP["aggC_ms"][0]),
                                p.get("aggC_gap",UP["aggC_gap"][0]),p["env_ms"])
    return base.Structural(pair[0],pair[0][:0] if model.compact else pair[1],t,ks,
                             np.searchsorted(t,n.time),n.key_count,n.od,gp,amount,held,el,sec,ms,half)


def pair_scores(full_notes,rated_notes,model,p,*,scope="context",include_curve=False,cancel=None):
    """One native whole chart, or a context/isolated segment."""
    def check():
        if cancel is not None and cancel():
            raise InterruptedError("计算已停止。")
    check()
    own=scope=="isolated" or full_notes is rated_notes
    full=state_from_notes(rated_notes if own else full_notes,model,p)
    check()
    v,v4,g=model.native(p)
    vv=v4 if full.key==4 else v
    d,_=field(full.a,full.b,full.t,full.key,vv,model.compact,model.hill,model.xsoft,False,
                   g if full.key==4 else 0.)
    s=full
    if not own:
        s=state_from_notes(rated_notes,model,p)
        j=np.searchsorted(full.t,s.t)
        heads=j[s.head_pos]
        if np.any(heads>=len(full.t)) or not np.array_equal(full.t[heads],s.t[s.head_pos]):
            raise RuntimeError("Context query identity mismatch.")
        if np.all(j<len(full.t)) and np.array_equal(full.t[j],s.t):
            d=d[j]
        else:
            d=np.interp(s.t,full.t,d)
    if model.pure:
        ph=np.empty(0);sl=ev=np.empty(0,np.int64);nb=0
    else:
        ph,sl,nb,ev=phase_structure(s)
    check()
    raw=aggregate(d,s.gp,s.amount,s.held,ph,sl,ev,nb,s.win_ms,s.env_half,len(s.head_pos),vv,*model.shape(p))
    out={"raw_score":float(raw)}
    if include_curve:out.update(query=s.t,field=d)
    return out
