"""SPM Rating v1.0.0 —— 自包含评分引擎（单文件）。

本文件由 core/ 自动组装，与仓库 core/ 在数值上完全一致。直接 import 即可使用：

  from spm_engine import rate_file, rate_content, rate_notes

除 numpy、scipy 外零第三方依赖。
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_PARAMS_JSON = {"note": "SPM Rating v1.0.0 release parameters. Benchmark scores (Spearman rank correlation): 7K 0.98449 / 4K 0.98535 / 6K zero-shot 0.95796.", "params": {"a_c": 10.717498918905136, "a_cb": 4.0, "a_cb_4k": 11.357365798773392, "a_cbc": 10.310670820602846, "a_p": 0.457050886903945, "a_p_4k": 0.9671708233722217, "a_r": 59.95490931764664, "a_r_4k": 0.0, "abar_active": 0.02531712913365081, "abar_c0": 0.729854838334696, "abar_c1": 0.6730069593477146, "abar_k": 5.869324658227848, "abar_mx": 0.48830826958952744, "abar_mx_thr": 0.10636359112398937, "abar_mx_w": 0.45817570312814426, "abar_scale": 1.009862473386292, "abar_thr_hi": 0.09447563280275387, "abar_thr_lo": 0.01682395583948343, "abar_weld": 1.0, "agg_C": 5.7765188650680015, "agg_gamma": 0.3142734186203886, "agg_gap_w": 1.0873927743990874, "agg_k": 3.923290274033005, "agg_mode": 3.0, "agg_nseg": 30.0, "agg_wc": 1.145022164188944, "aj": 4.107175656089324, "an_a0": 0.17043768459177114, "an_a1": 0.34850214310770977, "an_cubic": 6.806495193662656, "an_on_p": -0.3268168035336469, "ap": 0.8414094716039964, "at": 5.704296617075204, "c_chord2": 0.5, "c_fj": 1.1223698076832407, "c_fj_4k": 1.6384853006440867, "c_ja": -0.001250731346377573, "c_jc": 0.13300526379892, "c_jm": 0.5136914244166452, "c_s": 0.4182231311873207, "calib_a": 1.0269058337789674, "calib_b": 0.506803040656835, "cap_a": 10.84672199638674, "cap_b": 0.7669642744433789, "cb_holdtap": 0.0, "cb_holdtap_pow": 1.0, "cb_lock_pow": 1.0, "cb_lockanchor": 0.0, "cb_lockjack": 0.0, "cb_par": 0.0, "cb_shield": 0.0, "cb_shield_lock": 0.8062, "cb_shield_tau": 56.24, "cbv_churn": 2.7804415879691047, "cbv_churn_4k": 0.0018149727082640878, "cbv_holdage": 3.4070743387277163, "cbv_holdage_4k": 0.0017655285135421454, "cbv_holdtap": -1.0227403453455164, "cbv_lock_pow": 0.9662307730974806, "cbv_lockanchor": 1.3023578337914123, "cbv_lockstack": 0.5382064568725327, "cbv_ls_1h_split": 0.6669564049783417, "cbv_ls_2h_split": 0.628481439284875, "cbv_ls_cross": 0.6384201937709314, "cbv_ls_sh_adj": 0.9897529278156161, "cbv_ls_sh_split": 0.694320421215397, "cbv_ov_1h_split": 0.8912801595852077, "cbv_ov_2h_split": 0.628660224600841, "cbv_ov_cap": 2451.2427577736403, "cbv_ov_cross": 0.43216200109373926, "cbv_ov_dt_max": 1511.5378347371243, "cbv_ov_sh_adj": 1.2272220519138874, "cbv_ov_sh_split": 1.2153000673016874, "cbv_overlap": 0.9863986916717904, "cbv_sh_dt_max": 256.0406131304308, "cbv_sh_ln_tau": 145.443782024068, "cbv_sh_ln_w": 1.0790072249120857, "cbv_sh_lock": 1.0870780764260457, "cbv_sh_tau": 75.33463318028997, "cbv_shield": 1.1200999548581216, "cbv_str_1h_split": 0.5323308514011785, "cbv_str_2h_split": 0.17479688357773077, "cbv_str_bal": 1.141728630652899, "cbv_str_cross": 0.09922357503493356, "cbv_str_h0": 0.8793037998512705, "cbv_str_h1": 0.9113686383081036, "cbv_str_h2": 0.8557205717895041, "cbv_str_h3": 0.5028456653288493, "cbv_str_maxd": 2.048176307602251, "cbv_str_sh_adj": 0.19036255960279166, "cbv_str_sh_split": 0.9209759560022021, "cbv_straddle": 0.45854924271224995, "cbv_w": 1140.2267181326677, "chord2_w": 998.3068456460403, "chord_tau_c": 28.00006103515625, "churn_event_rate": 0.0, "churn_sat": 14.712127862753169, "churn_w": 1022.5551356347157, "cj_norm": 1.9757525590733565, "cj_size_exp": 1.0152415724691892, "d_b1": 1.4529112152056358, "d_b2": 0.4343400978218659, "d_ds": 0.5436886594030598, "d_dt": 1.3838263378115971, "d_eye": 0.0818717181241305, "d_eye_4k": 0.014013206098578592, "d_read": 0.0, "d_read_4k": 0.0, "d_rec": 0.78125, "d_sh": -0.18851239465270642, "env_ms": 395.28032205483703, "env_w": 0.0033333333333333335, "eye_tau": 0.2, "eye_w_ref": 4000.0, "eye_w_s": 500.0, "holdage_cap": 0.9712508526439506, "holdage_th": 0.25073103183213685, "hw_a": -4.833333333333334, "hw_half": 51.932695284873475, "j_agg_pow": 4.220517945172583, "j_c1": 0.11994305455701491, "j_nerf_a": -2.266651862989442e-05, "j_nerf_c": 0.07443739821757173, "j_nerf_off": 0.15369105717534054, "j_triple_gain": 0.01930565188193239, "j_triple_gain_4k": 0.5356436393950172, "j_triple_pow": 1.005976267076746, "j_triple_tau": 135.97491132645644, "ja_coord_exp": -9.268498548764114e-05, "ja_gap_thr": 253.00356478769385, "ja_len_exp": 1.0047596122947928, "ja_len_floor": 4.074244535477681, "ja_norm": 5.981391018054386, "ln_eff_tail_ms": 0.0, "ln_note_level": 0.0, "mean_pow": 6.465267031930503, "mj_dilute": 0.23679879956914796, "mj_dilute_exp": 0.9615171298692908, "od_mult_ja": -1.0, "od_mult_jc": -0.37109374999999944, "od_mult_jm": 0.7504257038347726, "od_mult_p": 1.1264849156945338, "od_mult_r": 1.064323992348447, "od_mult_x": 1.1651306736110387, "out_a": 0.699552129082087, "out_b": -0.7830329139461331, "p83_c": 2.667686959362696, "p93_c": 2.505172683584436, "p_boost": -1e-05, "p_boost_hi": 401.3793072521453, "p_boost_lo": 186.3466709871835, "p_burst3": 0.05749725160072005, "p_burst4": 0.041571581046874806, "p_bv_max": 1.023169107328028, "p_chw2": 0.6997350782823039, "p_chw3": 1.4710731528472405, "p_chw4": 0.82149995851254, "p_chw5": 0.8676070127257947, "p_chw6": 0.7474127532548505, "p_chw7": 1.03284160083206, "p_lam2": 0.006084380903675146, "p_lam3": 27.838293996532915, "p_sat_a": 10.690152688495372, "p_sat_b": 2.0349325289971496, "p_scale": 0.05862260059258437, "p_v_norm": -0.005007331887365899, "pp_div": 1.617868498725467, "pp_n0": 59.287573788026904, "pp_scale": 1.0018513493936536, "pp_thr": 7.703696089438818, "r_I_off": 1.679847001293981, "r_I_steep": 3.8155719025104933, "r_I_w": 1.0611113361746765, "r_ain0": 1.402277599248007, "r_ain1": 0.8487725298502042, "r_ain2": 0.9889530992577216, "r_ain3": 1.4808396303580251, "r_coord_e": 0.6807568539174652, "r_cw_cross": 0.6899948287076584, "r_cw_hand": 0.42489506847315023, "r_cw_same": 0.9824539263359018, "r_cw_thumb": 0.3660514384223894, "r_dt_max": 5985.197267885601, "r_dtr_min": 0.015, "r_lock": 0.1775454477373913, "r_order_pen": 0.04982726415201301, "r_order_tau": 404.3308885287425, "r_same_col": 0.39414011153048933, "r_seq": 0.06596340070980779, "r_short_red": 0.10936100877002214, "r_short_thr": 440.09681230961536, "r_sim_tau": 10.0, "r_simul": 0.08534094291675064, "r_soft_edges": 1.0, "r_straddle": -0.11540999412449468, "r_tail": 0.02898542543998874, "r_tt": 3.5417544104085192, "read_clip_hi": 1.5, "read_clip_lo": 0.5, "read_n_win": 16.0, "read_tol": 0.18, "rec_half": 0.4886690846205094, "recov_w": 989.9492044110127, "recov_wl": 21003.231873446508, "s_p": 1.2386750812693854, "shd_w": 1071.1117107781824, "t_jack_mix": -0.35552500051452174, "t_s_off": 0.4819735114526039, "use_cbar_v2": 1.0, "use_cbv_holdtap": 1.0, "use_cbv_lockanchor": 1.0, "use_cbv_lockstack": 1.0, "use_cbv_overlap": 1.0, "use_cbv_shield": 1.0, "use_cbv_straddle": 1.0, "use_eye": 1.0, "use_hb": 1.0, "use_read": 0.0, "w83": 0.4815139944602175, "w93": 0.3605157897072774, "w_a": 60.794584215917084, "w_cb": 1716.856156378797, "w_ja": 0.0, "w_jc": -0.4781249999999999, "w_jm": 842.9060364136417, "w_p": 806.2241563795392, "w_r": 1349.4219390385326, "w_x": 939.2935347185421, "wmean": 2.1613377064577026, "x_amp": 0.14776064699735425, "x_cw0": 0.1460113252705887, "x_cw1": 0.3535388218642357, "x_cw2": 0.3389326726836199, "x_cw3": 0.04918990446028124, "x_din": -0.595854461464495, "x_dir_in": 0.2845860534263308, "x_dir_out": -0.016749922744166222, "x_dout": 0.9858637139438629, "x_fc_a": 0.541953977488367, "x_fc_floor": 0.04577642898224647, "x_fc_off": 60.317605380705, "x_fc_w": 1.6631718126735746, "x_jd_e": 1.0765985987293114, "x_jd_p": 0.0456784859552089, "x_jh_e": 1.6812589290379765, "x_jh_p": 1.0261744953668672, "x_jt": 0.5592172136082718, "x_jump_w": -0.852278272911138}}

_PARAM_DEFAULTS = {
    # ---------------------------------------------------------------- OD
    # x_f = x ** od_mult_<f>  (x = hit leniency).  1.0 表示沿用基准行为。
    # od_mult_jc/od_mult_ja 为 -1 表示沿用 od_mult_jm 的值。
    "od_mult_jm": 0.85,
    "od_mult_jc": -1.0,
    "od_mult_ja": -1.0,
    "od_mult_p": 1.15,
    "od_mult_x": 1.15,
    "od_mult_r": 1.0,

    # ------------------------------------------------------- smoothing (ms)
    # w_jc/w_ja 为 0 表示沿用 w_jm 的值（与上面 OD 系数的继承规则一致）。
    "w_jm": 1000.0,
    "w_jc": 0.0,
    "w_ja": 0.0,
    "w_p": 1000.0,
    "w_x": 1000.0,
    "w_r": 1000.0,
    "w_cb": 1000.0,
    "w_a": 150.0,

    # ------------------------------------------------------------ jack core
    # kernel:  d^-1 * (d + c1*x^0.25)^-1 * nerfer
    "j_c1": 0.11,
    "j_nerf_a": 7e-5,          # 0 disables the nerfer
    "j_nerf_off": 0.15,
    "j_nerf_c": 0.08,
    "j_agg_pow": 4.27,          # cross-column power mean for every jack channel
    "j_triple_tau": 120.0,     # ms; same-column triple bonus window
    "j_triple_pow": 1.0,
    "j_triple_gain": 0.0,      # 0 = off  (h_jack3 mechanism, time-series form)
    # 同列三连增益，按谱面键数分别取值（0 表示关闭）。
    # < 0 表示沿用 j_triple_gain 的值。
    "j_triple_gain_4k": -1.0,
    # 同列快对通道，并入迷你叠。按键数分别取值（0 表示该键数关闭；
    # c_fj_4k 不沿用）。
    "c_fj": 0.0,
    "c_fj_4k": 0.0,
    # 同手和弦密度，进入流分支的折扣项。
    "d_sh": 0.0,
    "shd_w": 1000.0,           # ms smoothing window

    # minijack: dilution by what else is going on around the jack
    "mj_dilute": 0.0,          # 0 = plain Sunny Jbar
    "mj_dilute_exp": 1.0,

    # chordjack: chord load around the jack
    "cj_size_exp": 1.0,        # g(load) = (load/cj_norm)**size_exp
    "cj_norm": 2.0,            # keeps the chordjack channel on Jbar's scale

    # anchor: run length x coordination
    "ja_gap_thr": 250.0,       # ms; gaps above this break a run
    "ja_len_exp": 1.0,
    "ja_coord_exp": 0.0,       # weight of other-column activity
    "ja_len_floor": 4.0,       # runs shorter than this contribute ~0
    "ja_norm": 6.0,            # half-saturation run length for the anchor term

    # --------------------------------------------------------------- Pbar
    "p_lam3": 24.0,            # synchronous-note penalty
    "p_scale": 0.08,
    "p_lam2": 0.006,           # LN body weight (per ms of body)
    "p_v_norm": 0.0,           # 0 表示按毫秒计，1 表示按间隔归一化
    "p_boost": 1.75e-7,
    "p_boost_lo": 160.0,
    "p_boost_hi": 360.0,
    "p_bv_max": 1.0,           # 1 表示取两者较大值，0 表示相乘
    "p_sat_a": 10.0,           # soft clamp: min(inc*anchor, max(inc, inc*p_sat_b - p_sat_a))
    "p_sat_b": 2.0,
    # per chord size multiplier (index 0 == size 1, fixed at 1.0)
    "p_chw2": 1.0, "p_chw3": 1.0, "p_chw4": 1.0,
    "p_chw5": 1.0, "p_chw6": 1.0, "p_chw7": 1.0,
    # 和弦耦合（毫秒）：三角窗把和弦尺寸变成连续的有效尺寸（和弦
    # 定价表、同手折扣项、双押密度率共用），后一行相对前一行的间隔折算
    # 成新攻击的零点几分之一，连打的行数按攻击组计数。被拆成 1 毫秒多行
    # 的和弦不再被完整重复计价。0 表示关闭（精确时间戳行为）。
    "chord_tau_c": 0.0,
    # 短窗爆发通道（三音/四音爆发项）
    "p_burst3": 0.0,
    "p_burst4": 0.0,

    # ------------------------------------------------------- anchor / Abar
    "an_a0": 0.18,
    "an_a1": 0.22,
    "an_cubic": 5.0,
    "an_on_p": 1.0,            # strength of the column-imbalance multiplier
    "abar_scale": 1.0163,
    # 0 表示相邻序号的列配对；1 表示相邻活跃列配对（4 键嵌入时也会跨过
    # 空的拇指列）。保留作对照开关，不预设取值。
    "abar_active": 0.0,
    "abar_thr_lo": 0.02,
    "abar_thr_hi": 0.07,
    "abar_c0": 0.75,
    "abar_c1": 0.65,
    "abar_k": 5.0,
    "abar_mx": 0.5,
    "abar_mx_thr": 0.11,
    "abar_mx_w": 0.6,
    # 1 表示把中间段焊到连续形式，列对乘子在阈值处连续。
    # 0 表示分段形式，在接缝处有跳变。
    "abar_weld": 0.0,

    # --------------------------------------------------------------- Xbar
    "x_amp": 0.16,
    "x_cw0": 0.225,            # boundary groups {0,7} {1,6} {2,5} {3,4}
    "x_cw1": 0.35,
    "x_cw2": 0.25,
    "x_cw3": 0.225,
    # 成对项的内切/外切拆分：内切指切换落点比起点更靠近拇指列，外切指
    # 更远离。和弦参与的切换按有效列取半权。0/0 表示对称（无方向拆分）。
    "x_din": 0.0,
    "x_dout": 0.0,
    "x_fc_a": 0.4,
    "x_fc_off": 80.0,
    "x_fc_floor": 0.06,
    "x_fc_w": 1.0,
    # geometric cross matrix (jump component)
    "x_jump_w": 0.0,           # 0 = off
    "x_jh_p": 0.45,           # same-hand penalty
    "x_jh_e": 1.5,           # same-hand distance exponent
    "x_jt": 0.5536,            # 拇指桥接折扣
    "x_jd_p": 0.0,             # 通用距离惩罚
    "x_jd_e": 1.0,
    "x_dir_in": 0.0,           # 内切方向系数（朝向拇指列）
    "x_dir_out": 0.0,

    # --------------------------------------------------------------- Rbar
    "r_tail": 0.14488,
    "r_seq": 0.108498,
    "r_I_steep": 5.0,
    "r_I_off": 0.75,
    "r_I_w": 0.8,
    "r_dt_max": 5000.0,
    # 释放间隔定价的物理下限（秒）：用感知尺度的下限代替纯数值钳制。
    # 0.0001 表示关闭。
    "r_dtr_min": 0.0001,
    # 近同时事件（尾部与头部）的混合带（毫秒）：协调指数线性插值，
    # 不做硬判决。0 表示关闭。
    "r_sim_tau": 0.0,
    # 1 表示截断处加余弦坡，零宽释放区间分得一行的份额。0 表示关闭。
    "r_soft_edges": 0.0,
    "r_same_col": 0.26775,
    "r_coord_e": 0.71931,
    "r_tt": 4.2242,
    "r_short_thr": 331.62,
    "r_short_red": 0.10009,
    "r_lock": 0.11649,
    "r_simul": 0.05,
    "r_cw_same": 1.0, "r_cw_hand": 0.8, "r_cw_thumb": 0.4, "r_cw_cross": 0.2,
    # new: release-order triple + lock-straddle matrix
    "r_order_pen": 0.0,        # 102/120 harder than 012/210
    "r_order_tau": 400.0,      # ms; max span of the triple
    "r_straddle": 0.0,         # lock-straddle weight
    "r_ain0": 1.0, "r_ain1": 1.0, "r_ain2": 1.0, "r_ain3": 1.0,

    # --------------------------------------------------------------- Cbar
    "cb_shield": 0.0,
    "cb_shield_tau": 56.24,
    "cb_shield_lock": 0.8062,
    "cb_lockjack": 0.0,
    "cb_lock_pow": 1.0,
    "cb_lockanchor": 0.0,
    "cb_par": 0.0,             # true LN overlap (non-nested) density
    "cb_holdtap": 0.0,
    "cb_holdtap_pow": 1.0,

    # ------------------------------------------------- Cbar v2 (lncoord_v2)
    # Row-grid LN coordination.  `use_cbar_v2` is the master gate: while it
    # is 0 the channel returns exact zeros and Cbar == Cbar v1 bit-for-bit.
    # Each sub-term additionally has its own `use_cbv_<term>` switch and its
    # own `cbv_<term>` weight, so terms ablate independently.
    "use_cbar_v2": 0.0,
    # 1 表示按尾部时刻计数释放事件（不依赖网格的速率）。改变释放拥挤度
    # 的尺度，开启后需要重新拟合权重。0 表示关闭。
    "churn_event_rate": 0.0,
    "cbv_w": 1000.0,           # smoothing window (ms) shared by all v2 terms
    # --- term gates + weights
    "use_cbv_shield": 0.0,     "cbv_shield": 0.0,
    "use_cbv_straddle": 0.0,   "cbv_straddle": 0.0,
    "use_cbv_lockstack": 0.0,  "cbv_lockstack": 0.0,
    "use_cbv_lockanchor": 0.0, "cbv_lockanchor": 0.0,
    "use_cbv_overlap": 0.0,    "cbv_overlap": 0.0,
    "use_cbv_holdtap": 0.0,    "cbv_holdtap": 0.0,
    # --- shield shape: tap/short-LN -> LN on the same column
    "cbv_sh_tau": 60.0,        # ms; decay of the predecessor gap
    "cbv_sh_dt_max": 250.0,    # ms; max gap that still counts as a shield
    "cbv_sh_ln_w": 1.0,        # weight of a zero-length LN predecessor vs tap
    "cbv_sh_ln_tau": 120.0,    # ms; long LN predecessors stop being a shield
    "cbv_sh_lock": 1.0,        # amplification by held other columns
    # --- 衩叠对矩阵 + 被钉手指权重
    "cbv_str_sh_adj": 0.0,     # unused at d==2; kept for matrix completeness
    "cbv_str_sh_split": 1.0,   # (0,2) (4,6) -- the true straddle
    "cbv_str_1h_split": 0.45,  # (1,3) (3,5) -- thumb-bridged
    "cbv_str_2h_split": 0.20,  # (2,4)
    "cbv_str_cross": 0.10,     # d >= 3
    "cbv_str_h0": 0.70,        # pinned-column weight indexed by AIN_GROUP
    "cbv_str_h1": 1.00,        #   h1 = middle finger -- the worst pin
    "cbv_str_h2": 0.70,
    "cbv_str_h3": 0.40,        #   h3 = thumb
    "cbv_str_bal": 1.0,        # exponent on the alternation-balance factor
    "cbv_str_maxd": 2.0,       # max pair distance considered (2 = 衩叠)
    # --- lock matrix: acting on column a while column j is held
    "cbv_lock_pow": 1.0,     # lock-mass exponent for lockstack/lockanchor
    "cbv_ls_sh_adj": 0.60,
    "cbv_ls_sh_split": 1.00,
    "cbv_ls_1h_split": 0.80,
    "cbv_ls_2h_split": 0.50,
    "cbv_ls_cross": 0.30,
    # --- overlap (partial, non-nested LN pairs) pair matrix
    "cbv_ov_sh_adj": 1.00,
    "cbv_ov_sh_split": 1.00,
    "cbv_ov_1h_split": 0.70,
    "cbv_ov_2h_split": 0.50,
    "cbv_ov_cross": 0.30,
    "cbv_ov_dt_max": 1500.0,   # ms; max head-to-head distance for a pair
    "cbv_ov_cap": 2000.0,      # ms; clamp on the priced overlap length

    # ------------------------------------------------- 混合长键补充项
    # 四条曲线级机制，针对 7 键混合与长键谱面。
    # use_hb 为总开关；每项有自己的权重（0 表示关闭）。
    # 按键数覆盖（0 表示沿用 7 键的值），与 a_p_4k 等规则相同。
    "use_hb": 0.0,
    # release churn (concave), added into the Cbar curve
    "cbv_churn": 0.0,
    "cbv_churn_4k": 0.0,
    "churn_sat": 15.0,         # releases/s soft saturation
    "churn_w": 1000.0,         # ms window
    # LN hold-age (occupation duration), added into the Cbar curve
    "cbv_holdage": 0.0,
    "cbv_holdage_4k": 0.0,
    "holdage_th": 0.25,        # seconds before a hold starts costing
    "holdage_cap": 1.0,        # seconds cap per hold
    # chord2 density rate, additive into the stream branch
    "c_chord2": 0.0,
    "chord2_w": 1000.0,        # ms window
    # recovery-aware burst discount on the stream branch
    "d_rec": 0.0,
    "rec_half": 0.5,           # half-saturation of the burst ratio
    "recov_w": 1000.0,         # short window (ms)
    "recov_wl": 20000.0,       # baseline window (ms)

    # ------------------------------------------------------------- combine
    "aj": 3.0,                 # Abar exponent /Ks on the jack branches
    "ap": 0.6667,              # Abar exponent on the stream branch
    "at": 3.0,                 # Abar exponent /Ks on T
    "cap_a": 8.0,
    "cap_b": 0.85,
    "c_jm": 0.484,             # jack-branch weights inside S
    "c_jc": 0.0,
    "c_ja": 0.0,
    "c_s": 0.516,
    "s_p": 1.0,                # power-mean exponent of S branches
    "a_p": 0.72454,            # Pbar weight inside the stream branch
    "a_r": 32.2385,            # Rbar weight
    "a_c": 11.0205,            # C offset for Rbar
    "a_cb": 0.0,               # Cbar weight
    "a_cbc": 11.0205,
    # 4 键谱面的覆盖系数（门控为 s.is_4k）。两种键数在流分支与长键定价
    # 上取值不同：每个 *_4k 覆盖为 0 表示沿用共用值。
    "a_p_4k": 0.0,
    "a_r_4k": 0.0,
    "a_cb_4k": 0.0,
    "t_jack_mix": 0.0,         # jack contribution to the twist numerator
    "t_s_off": 1.0,
    "d_b1": 1.1879,
    "d_ds": 0.5,
    "d_dt": 1.5,
    "d_b2": 0.3845,

    # ----------------------------------------------------------- aggregate
    "agg_mode": 3.0,           # 0 percentile | 1 soft-max | 2 sigmoid | 3 power mean
    "agg_k": 4.0,              # mode1: lambda | mode2: sigmoid k | mode3: power
    "agg_C": 3.96885,
    "agg_gamma": 0.19562,
    "agg_nseg": 30.0,
    "agg_wc": 1.0,             # weight exponent on the local note count C
    "agg_gap_w": 1.0,          # weight exponent on the row gap
    # 聚合前 D 的峰包络：D <- (1-env_w)*D + env_w * rolling_max(D, env_ms)。
    # 0 表示关闭。
    "env_w": 0.0,
    "env_ms": 800.0,
    # percentile mode (agg_mode < 0.5)
    "p93_c": 0.88, "w93": 0.25,
    "p83_c": 0.94, "w83": 0.20,
    "wmean": 0.55, "mean_pow": 5.0,

    # ---------------------------------------------------------- postprocess
    "pp_n0": 1.0286,
    "pp_thr": 9.1064,
    "pp_div": 1.9709,
    "pp_scale": 1.0944,
    "calib_a": 1.0,
    "calib_b": 0.0,
    # 按住感知聚合权重：按住长键的行在聚合时降权。
    "hw_a": 0.0,
    "hw_half": 30.0,
    # ln_eff_tail_ms：长键有效尾部缩减（毫秒）。结构级参数：它改变行结构
    # 的按住数组，所以放在 STRUCT_KEYS 里，单独处理，不放在普通参数组里。
    "ln_eff_tail_ms": 0.0,
    # ln_note_level 为 1 时，按住状态按每条长键的头尾数组构建，而不是按行
    # 构建。行级构建会把与长键头同行的点键误标成长键头。结构级参数，
    # 与 ln_eff_tail_ms 同级。0 表示行级构建。
    "ln_note_level": 0.0,
    # 读谱分支：间隔规整度曲线（默认关闭，保留作对照）
    "use_read": 0.0,         # 0 = off, 1 = on
    "d_read": 0.0,           # 读谱系数（7 键）
    "d_read_4k": 0.0,        # 读谱系数（4 键覆盖；0 表示与 7 键相同）
    "read_tol": 0.18,        # 规整度容差
    "read_n_win": 16.0,      # 局部中位窗（事件数）
    "read_clip_lo": 0.5,     # 读谱修正项的裁剪范围
    "read_clip_hi": 1.5,
    # 视觉规整度：拍点对齐度，流分支内的密度门控交互
    "use_eye": 0.0,          # 0 = off, 1 = on
    "d_eye": 0.0,            # 7 键系数
    "d_eye_4k": 0.0,         # 4 键覆盖；0 表示与 7 键相同
    "eye_w_ref": 4000.0,     # local rate reference window (ms)
    "eye_w_s": 500.0,        # smoothing window (ms)
    "eye_tau": 0.2,          # alignment decay scale
    # output-domain affine (per dataset; only 4K needs a non-trivial one)
    "out_a": 1.0,
    "out_b": 0.0,
}

DEFAULTS = _PARAM_DEFAULTS

_PARAM_DEFAULTS_REF = _PARAM_DEFAULTS

# 参数分组。各通道的键按前缀组织，供评分代码与验证脚本按组取用。
BLOCKS: Dict[str, List[str]] = {
    # od_mult_ja / w_ja / c_ja 不参与调整：长连通道被证明冗余，
    # 它的形状参数保留默认值（继承语义），以保持旧参数文件的兼容。
    "od": [k for k in _PARAM_DEFAULTS if k.startswith("od_mult_")
           and k != "od_mult_ja"],
    "smooth": [k for k in _PARAM_DEFAULTS if k.startswith("w_") and k != "w_ja"],
    "jack": [k for k in _PARAM_DEFAULTS if k.startswith(("j_", "mj_", "cj_", "ja_"))],
    "jackshape": ["cj_size_exp", "cj_norm", "ja_len_exp", "ja_len_floor",
                  "ja_norm", "ja_coord_exp", "mj_dilute_exp"],
    "jackgain": ["mj_dilute", "j_triple_gain", "j_triple_gain_4k", "j_triple_tau",
                 "c_fj", "c_fj_4k", "shd_w"],
    "pbar": [k for k in _PARAM_DEFAULTS if k.startswith("p_")],
    "abar": [k for k in _PARAM_DEFAULTS if k.startswith(("an_", "abar_"))
             and k != "abar_weld"],
    "xbar": [k for k in _PARAM_DEFAULTS if k.startswith("x_")],
    "rbar": [k for k in _PARAM_DEFAULTS if k.startswith("r_")
             and k != "r_soft_edges"],
    "cbar": [k for k in _PARAM_DEFAULTS if k.startswith("cb_")],
    # 长键协调通道 v2。`sat_cbar_v2` 是权重与形状参数组；`use_*` 开关
    # 是 0/1 手工设置，用于对照实验，不参与数值优化。
    "sat_cbar_v2": [k for k in _PARAM_DEFAULTS if k.startswith("cbv_")],
    "gates": ["use_cbar_v2", "r_soft_edges", "abar_weld", "ln_note_level",
              "churn_event_rate"] +
             [k for k in _PARAM_DEFAULTS if k.startswith("use_cbv_")],
    # 和弦耦合参数（密度通道的和弦尺寸、同手折扣项、混合补充项的双押
    # 密度率共用），单独成组，不放在上述通道组里。
    "chordsoft": ["chord_tau_c"],
    "eye": ["d_eye", "d_eye_4k"],
    # cbv_churn / cbv_holdage carry the cbv_ prefix, so they live in the
    # sat_cbar_v2 block automatically; the hb block holds the rest.
    "hb": ["c_chord2", "d_rec",
           "churn_sat", "churn_w", "holdage_th", "holdage_cap",
           "chord2_w", "rec_half", "recov_w", "recov_wl"],
    "combine": ["aj", "ap", "at", "cap_a", "cap_b", "c_jm", "c_jc",
                "c_s", "s_p", "a_p", "a_r", "a_c", "a_cb", "a_cbc",
                "a_p_4k", "a_r_4k", "a_cb_4k", "d_sh",
                "t_jack_mix", "t_s_off", "d_b1", "d_ds", "d_dt", "d_b2"],
    "agg": ["agg_k", "agg_C", "agg_gamma", "agg_wc", "agg_gap_w",
            "p93_c", "w93", "p83_c", "w83", "wmean", "mean_pow",
            "hw_a", "hw_half", "env_w", "env_ms"],
    "post": ["pp_n0", "pp_thr", "pp_div", "pp_scale", "calib_a", "calib_b",
             "out_a", "out_b"],
    "affine": ["out_a", "out_b", "calib_a", "calib_b"],
}

# 改变结构输出的参数。其他参数可以复用已缓存的结构数组（注释保持
# 显式并经过审计：结构键如果漏进这个表，会让每次参数步进都变成全量重算）。
STRUCT_KEYS: List[str] = [
    "ja_gap_thr", "j_triple_tau", "r_dt_max", "r_short_thr",
    "cb_shield_tau", "r_order_tau", "agg_nseg",
    "cbv_sh_dt_max", "cbv_ov_dt_max", "cbv_str_maxd", "cbv_w",
    "ln_eff_tail_ms", "ln_note_level",
]


def get(params: Dict[str, float], key: str) -> float:
    v = params.get(key)
    return _PARAM_DEFAULTS[key] if v is None else v


def make(overrides: Dict[str, float] | None = None,
         strict: bool = True) -> Dict[str, float]:
    p = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(_PARAM_DEFAULTS)
        if unknown:
            if strict:
                raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
            overrides = {k: v for k, v in overrides.items() if k in _PARAM_DEFAULTS}
        p.update(overrides)
    return p


def load(path: str, strict: bool = False) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "params" in raw:
        raw = raw["params"]
    return make(raw, strict=strict)


def save(params: Dict[str, float], path: str, note: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"note": note,
               "params": {k: float(v) for k, v in params.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)


def block_keys(names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for n in names:
        out.extend(BLOCKS.get(n, [n]))
    return sorted(set(out))


def nm_block_keys(names: Iterable[str]) -> List[str]:
    """分组键减去结构级键。

    结构级键改变行结构的输出；在分片计算里，这类键的每一步都会触发
    全进程的结构重建。结构级键用网格加内部搜索单独处理，不放在普通
    参数组里。
    """
    sk = set(STRUCT_KEYS)
    return [k for k in block_keys(names) if k not in sk]


def struct_fingerprint(params: Dict[str, float]) -> tuple:
    return tuple(round(float(params.get(k, _PARAM_DEFAULTS[k])), 9) for k in STRUCT_KEYS)


# 不改变难度曲线输出的参数：合并层系数、聚合、标定，以及合并层的
# 视觉/读谱系数。其他参数（连打/密度/不均匀度/跨列/释放/协调/击打
# 宽容度/平滑窗 + 结构级键）都会影响曲线。curve_fingerprint 让调用方在
# 只有非曲线参数变化时复用已缓存的曲线，这是最大的加速来源。
NON_CURVE_KEYS: set = set(
    BLOCKS["combine"] + BLOCKS["agg"] + BLOCKS["post"] + BLOCKS["eye"]
) | {"use_read", "d_read", "d_read_4k", "read_clip_lo",
     "read_clip_hi", "read_tol", "read_n_win",
     # c_chord2 / d_rec are applied in combine() (they scale/add to the
     # stream branch), not in compute_curves -- they must NOT trigger a
     # curve recompute.  (use_hb and the *_w / *_sat / *_th / *_cap shape
     # params DO change the curves and stay curve-affecting.)
     "c_chord2", "d_rec"}
# NB: use_eye is deliberately NOT here -- it gates EyeR computation inside
# compute_curves, so it is curve-affecting.


def curve_fingerprint(params: Dict[str, float]) -> tuple:
    return tuple(round(float(params.get(k, _PARAM_DEFAULTS[k])), 9)
                 for k in sorted(DEFAULTS) if k not in NON_CURVE_KEYS)


if __name__ == "__main__":
    print(f"{len(DEFAULTS)} parameters")
    seen = set()
    for b, keys in BLOCKS.items():
        dup = seen & set(keys)
        if dup:
            print(f"!! block {b} overlaps: {sorted(dup)}")
        seen |= set(keys)
    missing = set(_PARAM_DEFAULTS) - seen
    print("unblocked:", sorted(missing))
    for k in STRUCT_KEYS:
        if k not in _PARAM_DEFAULTS:
            print("!! struct key not in _PARAM_DEFAULTS:", k)



# ===================== core/grid.py =====================


import numpy as np

K = 7                      # fixed 7-column frame
N_BOUND = K + 1            # 8 column boundaries for Xbar

POPCOUNT = np.array([bin(i).count("1") for i in range(128)], dtype=np.int64)


class Grid:
    """Row grid derived from a parsed chart."""

    __slots__ = ("t", "mask", "size", "n", "dt", "col_rows", "col_dt",
                 "bound_rows", "bound_dt", "note_times", "is_ln", "ln_end",
                 "key_count", "od", "x", "n_notes", "duration", "rate",
                 "col_notes", "row_of_note")

    def __init__(self, notes, od_leniency: float):
        time = notes.time
        # ---- rows: distinct timestamps
        first = np.empty(len(time), dtype=bool)
        first[0] = True
        np.not_equal(time[1:], time[:-1], out=first[1:])
        row_of_note = np.cumsum(first) - 1
        self.row_of_note = row_of_note
        R = int(row_of_note[-1]) + 1
        t = np.zeros(R, dtype=np.float64)
        t[row_of_note] = time
        mask = np.zeros(R, dtype=np.int32)
        np.bitwise_or.at(mask, row_of_note, (1 << notes.col).astype(np.int32))
        # LN: row is "ln" if any note there is an LN
        is_ln_row = np.zeros(R, dtype=bool)
        np.logical_or.at(is_ln_row, row_of_note, notes.is_ln)
        ln_end = np.full(R, -1.0, dtype=np.float64)
        np.maximum.at(ln_end, row_of_note,
                      np.where(notes.is_ln, notes.ln_end, -1.0))

        self.t = t
        self.mask = mask
        self.size = POPCOUNT[mask]
        self.n = R
        self.is_ln = is_ln_row
        self.ln_end = ln_end
        self.note_times = time
        self.col = notes.col
        self.n_notes = notes.n_notes
        self.key_count = notes.key_count
        self.od = notes.od
        self.x = float(od_leniency)
        self.duration = float(t[-1] - t[0]) if R > 1 else 0.0
        self.rate = notes.rate

        # ---- row gaps (step function on [t[r], t[r+1]) )
        dt = np.diff(t) / 1000.0                      # seconds
        self.dt = dt                                  # (R-1,)

        # ---- per-column note rows + gaps
        self.col_rows = []
        self.col_dt = []
        self.col_notes = []
        for k in range(K):
            idx = np.flatnonzero((mask >> k) & 1)
            self.col_rows.append(idx)
            self.col_notes.append(idx)
            if idx.size >= 2:
                self.col_dt.append(np.diff(t[idx]) / 1000.0)
            else:
                self.col_dt.append(np.zeros(0, dtype=np.float64))

        # ---- per-boundary (two-column union) rows + gaps
        self.bound_rows = []
        self.bound_dt = []
        for b in range(N_BOUND):
            cols = []
            if b - 1 >= 0:
                cols.append(b - 1)
            if b < K:
                cols.append(b)
            if not cols:
                self.bound_rows.append(np.zeros(0, dtype=np.int64))
                self.bound_dt.append(np.zeros(0, dtype=np.float64))
                continue
            merged = np.unique(np.concatenate([self.col_rows[c] for c in cols]))
            self.bound_rows.append(merged)
            self.bound_dt.append(np.diff(t[merged]) / 1000.0
                                 if merged.size >= 2 else
                                 np.zeros(0, dtype=np.float64))

    # ------------------------------------------------------------------
    @property
    def n_ln(self) -> int:
        return int(self.is_ln.sum())

    def col_step(self, k: int, values: np.ndarray) -> np.ndarray:
        """Scatter per-column-gap `values` (len = len(col_rows[k])-1) onto
        the row grid: value[j] holds on rows [idx[j], idx[j+1])."""
        out = np.zeros(self.n, dtype=np.float64)
        idx = self.col_rows[k]
        if idx.size < 2:
            return out
        lengths = np.diff(idx)
        # a chord containing column k at row idx[j] -> interval length >= 0
        seg = np.repeat(values[:len(lengths)], np.maximum(lengths, 0))
        start = idx[0]
        end = start + len(seg)
        out[start:min(end, self.n)] = seg[:min(end, self.n) - start]
        return out

    def bound_step(self, b: int, values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        idx = self.bound_rows[b]
        if idx.size < 2:
            return out
        lengths = np.diff(idx)
        seg = np.repeat(values[:len(lengths)], np.maximum(lengths, 0))
        start = idx[0]
        end = start + len(seg)
        out[start:min(end, self.n)] = seg[:min(end, self.n) - start]
        return out


# ----------------------------------------------------------------------
def step_to_row(t: np.ndarray, src_idx: np.ndarray, values: np.ndarray,
                n: int) -> np.ndarray:
    """Scatter values defined on consecutive src_idx intervals onto rows."""
    out = np.zeros(n, dtype=np.float64)
    if src_idx.size < 2:
        return out
    lengths = np.diff(src_idx)
    seg = np.repeat(values[:len(lengths)], np.maximum(lengths, 0))
    start = int(src_idx[0])
    end = min(start + len(seg), n)
    if end > start:
        out[start:end] = seg[:end - start]
    return out


def window_mean(t: np.ndarray, val: np.ndarray, W: float) -> np.ndarray:
    """Centred moving average of the step function `val` with window W (ms).

    val[r] is constant on [t[r], t[r+1]); t must be strictly increasing.
    Returns (1/W) * int_{s-W/2}^{s+W/2} val, sampled at s = t[r].
    """
    n = t.size
    if n < 2:
        return val.copy()
    dt = np.diff(t)
    inc = val[:-1] * dt
    I = np.concatenate(([0.0], np.cumsum(inc)))        # I[j] = int up to t[j]

    def integ(z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, t[0], t[-1])
        j = np.clip(np.searchsorted(t, zc, side="right") - 1, 0, n - 2)
        return I[j] + val[j] * (zc - t[j])

    lo = t - 0.5 * W
    hi = t + 0.5 * W
    out = (integ(hi) - integ(lo)) / W
    return np.maximum(out, 0.0)


def window_sum(t: np.ndarray, val: np.ndarray, W: float) -> np.ndarray:
    """Centred moving *sum* (not average) over window W, in ms*value."""
    return window_mean(t, val, W) * W


def local_count(t: np.ndarray, weights: np.ndarray, W: float) -> np.ndarray:
    """Number of `weights` mass inside [s-W/2, s+W/2) at s = t[r].

    Used for C(s) (local note count) and for density measures.
    """
    n = t.size
    csum = np.concatenate(([0.0], np.cumsum(weights)))

    def cnt(z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, t[0], t[-1] + 1.0)
        return csum[np.searchsorted(t, zc, side="left")]

    return cnt(t + 0.5 * W) - cnt(t - 0.5 * W)


def held_count(notes_ln_head: np.ndarray, notes_ln_tail: np.ndarray,
               t: np.ndarray) -> np.ndarray:
    """Number of LNs whose body strictly covers time t (row grid).

    An LN with head h and tail e is *held* on (h, e).
    """
    n = t.size
    if notes_ln_head.size == 0:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(notes_ln_head, kind="stable")
    h = notes_ln_head[order]
    e = notes_ln_tail[order]
    # events: +1 at h (exclusive) / -1 at e (inclusive of t==e? use strict)
    starts = np.searchsorted(t, h, side="right")
    ends = np.searchsorted(t, e, side="left")
    d = np.zeros(n + 2, dtype=np.float64)
    np.add.at(d, starts, 1.0)
    np.add.at(d, ends, -1.0)
    return np.maximum(np.cumsum(d)[:n], 0.0)


def run_lengths(keep: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values each element belongs to.

    `keep` is boolean over a sequence; runs are maximal True streaks.
    Elements with keep == False get 0.
    """
    n = keep.size
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out
    brk = np.concatenate(([True], ~keep[:-1])) & keep
    ids = np.cumsum(brk) - 1
    if not np.any(brk):
        return out
    lens = np.bincount(ids[keep])
    out[keep] = lens[ids[keep]]
    return out


def active_col_count(grid: Grid, W: float = 300.0) -> np.ndarray:
    """Number of distinct columns with a note in [s-W/2, s+W/2)."""
    n = grid.n
    cnt = np.zeros((K, n), dtype=np.float64)
    for k in range(K):
        idx = grid.col_rows[k]
        if idx.size == 0:
            continue
        v = np.zeros(n, dtype=np.float64)
        np.add.at(v, idx, 1.0)
        cnt[k] = local_count(grid.t, v, W)
    return (cnt > 0).sum(axis=0)


# ===================== core/parser.py =====================


import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

MANIA_MODE = 3

# 4K original column -> 7K frame column
EMBED_4K_TO_7K = (1, 2, 4, 5)

# Default OD -> hit window mapping is not used directly; we only need a
# normalised "timing pressure" scalar.  Standard osu!mania windows (ms):
OD_WINDOWS = {0: (22.0, 64.0, 97.0, 151.0, 188.0),
              5: (19.5, 49.0, 82.0, 136.0, 172.0),
              10: (16.0, 34.0, 67.0, 121.0, 151.0)}


def od_windows(od: float) -> Tuple[float, float, float, float, float]:
    """Interpolate the osu!mania hit windows (300g, 300, 200, 100, 50) for OD."""
    od = float(min(max(od, 0.0), 10.0))
    lo_i = int(od // 5) * 5
    hi_i = min(lo_i + 5, 10)
    t = (od - lo_i) / 5.0
    a, b = OD_WINDOWS[lo_i], OD_WINDOWS[hi_i]
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(5))  # type: ignore[return-value]


def rate_od(od: float, rate: float) -> float:
    """Convert OD to the given speed rate (DT raises effective OD)."""
    # standard: OD_ms = rate-scaled; new_od such that window/rate matches
    if rate <= 0:
        return od
    w = od_windows(od)[0]
    target = w / rate
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if od_windows(mid)[0] > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Notes:
    """Note arrays in the 7K frame.  All arrays share the leading axis."""
    time: np.ndarray        # float64, ms, non-decreasing
    col: np.ndarray         # int32, 0..6
    ln_end: np.ndarray      # float64, ms; == time for non-LN
    is_ln: np.ndarray       # bool
    key_count: int          # original key count (4 or 7)
    od: float               # effective OD after rate conversion
    rate: float
    n_notes: int
    # per-column index lists (list of 7 int64 arrays)
    col_idx: Tuple[np.ndarray, ...] = ()
    # derived
    duration: float = 0.0
    title: str = ""

    @property
    def n_ln(self) -> int:
        return int(self.is_ln.sum())

    @property
    def ln_ratio(self) -> float:
        if self.n_notes == 0:
            return 0.0
        return float(self.is_ln.sum()) / float(self.n_notes)


def col_from_x(x: float, key_count: int) -> int:
    """osu!mania encodes the column as x = round(512*(c+0.5)/K)."""
    c = int(float(x) * key_count / 512.0)
    if c < 0:
        c = 0
    if c > key_count - 1:
        c = key_count - 1
    return c


def _col_from_x(x: float, key_count: int) -> int:
    c = col_from_x(x, key_count)
    if key_count == 7:
        return c
    if key_count == 4:
        return EMBED_4K_TO_7K[c]
    # generic: spread evenly across the 7 tracks
    if key_count <= 1:
        return 3
    return int(round(float(c) * 6.0 / (key_count - 1)))


def parse_content(text: str, rate: float = 1.0,
                  title: str = "") -> Optional[Notes]:
    lines = text.splitlines()
    key_count = 0
    od = 5.0
    section = ""
    hit_objects: List[Tuple[int, float, int, int]] = []   # (type, t, x, end)
    in_objects = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section = line.strip("[]").strip()
            in_objects = section == "HitObjects"
            continue
        if ":" in line and not in_objects:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k == "Mode" and v != str(MANIA_MODE):
                return None
            if k == "CircleSize":
                try:
                    key_count = int(float(v))
                except ValueError:
                    pass
            elif k == "OverallDifficulty":
                try:
                    od = float(v)
                except ValueError:
                    pass
            continue
        if in_objects:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                x = int(float(parts[0]))
                t = float(parts[2])
                typ = int(parts[3])
            except ValueError:
                continue
            end = t
            if typ & 128:
                end = float(parts[5].split(":")[0]) if len(parts) > 5 else t
            hit_objects.append((typ, t, x, end))
    if not hit_objects or key_count <= 0:
        return None

    eff_od = rate_od(od, rate)
    hit_objects.sort(key=lambda o: (o[1], o[2]))
    n = len(hit_objects)
    time = np.empty(n, dtype=np.float64)
    col = np.empty(n, dtype=np.int32)
    ln_end = np.empty(n, dtype=np.float64)
    is_ln = np.zeros(n, dtype=bool)
    for i, (typ, t, x, end) in enumerate(hit_objects):
        time[i] = t / rate
        col[i] = _col_from_x(x, key_count)
        if typ & 128 and end > t:
            ln_end[i] = end / rate
            is_ln[i] = True
        else:
            ln_end[i] = time[i]
    order = np.argsort(time, kind="stable")
    time = time[order]
    col = col[order]
    ln_end = ln_end[order]
    is_ln = is_ln[order]

    col_idx = tuple(np.flatnonzero(col == c) for c in range(7))
    return Notes(time=time, col=col, ln_end=ln_end, is_ln=is_ln,
                 key_count=key_count, od=eff_od, rate=rate, n_notes=n,
                 col_idx=col_idx,
                 duration=float(time[-1] - time[0]) if n else 0.0,
                 title=title)


def parse_file(path: str, rate: float = 1.0) -> Optional[Notes]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    return parse_content(text, rate=rate, title=os.path.basename(path))


# --------------------------------------------------------------- cache
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cache", "notes")


def cached_parse(path: str, rate: float = 1.0) -> Optional[Notes]:
    """Parse with an npz disk cache keyed by (file, size, mtime, rate)."""
    try:
        st = os.stat(path)
        key = f"{os.path.basename(path)[:-4]}_{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return parse_file(path, rate)
    if abs(rate - 1.0) > 1e-9:
        key += f"_r{rate:.4f}"
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fp = os.path.join(_CACHE_DIR, key + ".npz")
    if os.path.exists(fp):
        try:
            with np.load(fp, allow_pickle=False) as z:
                col_idx = tuple(z[f"ci{c}"] for c in range(7))
                return Notes(time=z["time"], col=z["col"], ln_end=z["ln_end"],
                             is_ln=z["is_ln"], key_count=int(z["key_count"]),
                             od=float(z["od"]), rate=float(z["rate"]),
                             n_notes=int(z["n_notes"]), col_idx=col_idx,
                             duration=float(z["duration"]),
                             title=str(z["title"]))
        except Exception:                                    # noqa: BLE001
            pass
    notes = parse_file(path, rate)
    if notes is None:
        return None
    try:
        data = {"time": notes.time, "col": notes.col, "ln_end": notes.ln_end,
                "is_ln": notes.is_ln, "key_count": np.int64(notes.key_count),
                "od": np.float64(notes.od), "rate": np.float64(notes.rate),
                "n_notes": np.int64(notes.n_notes),
                "duration": np.float64(notes.duration),
                "title": np.array(notes.title)}
        for c in range(7):
            data[f"ci{c}"] = notes.col_idx[c]
        np.savez_compressed(fp, **data)
    except Exception:                                        # noqa: BLE001
        pass
    return notes


if __name__ == "__main__":
    import sys
    import time as _t
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "tests", "cases", "bm_0001.osu")
    t0 = _t.perf_counter()
    nt = parse_file(p)
    print(f"{nt.n_notes} notes, k={nt.key_count}, od={nt.od:.2f}, "
          f"ln={nt.n_ln} ({nt.ln_ratio:.3f}), dur={nt.duration/1000:.1f}s, "
          f"parse {(_t.perf_counter()-t0)*1000:.1f}ms")
    print("cols used:", sorted(set(nt.col.tolist())))


# ===================== core/batch.py =====================


import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


CHART_STRIDE = 1 << 30          # ms offset between charts (>> any window)


# --------------------------------------------------------------- primitives
def seg_cumsum(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Cumulative sum restarted at every segment boundary."""
    c = np.cumsum(x)
    n_seg = starts.size - 1
    base = np.zeros(n_seg, dtype=c.dtype)
    if n_seg > 1:
        prev_end = starts[1:-1] - 1
        base[1:] = c[prev_end]
    return c - np.repeat(base, np.diff(starts))


def seg_cumsum_from(c: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Same as seg_cumsum but takes an already-computed global cumsum."""
    n_seg = starts.size - 1
    base = np.zeros(n_seg, dtype=c.dtype)
    if n_seg > 1:
        base[1:] = c[starts[1:-1] - 1]
    return c - np.repeat(base, np.diff(starts))


@dataclass
class Batch:
    """Flat arrays covering every chart in the evaluation set."""
    # --- identity
    ids: List[str] = field(default_factory=list)
    starts: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    # --- row grid (flat)
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))       # local ms
    tg: np.ndarray = field(default_factory=lambda: np.zeros(0))      # global ms
    chart: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    size: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    is_ln: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    ln_end: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dt: np.ndarray = field(default_factory=lambda: np.zeros(0))      # sec, len n-1
    n: int = 0
    # --- per chart scalars
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))       # hit leniency
    od: np.ndarray = field(default_factory=lambda: np.zeros(0))
    key_count: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    n_notes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    duration: np.ndarray = field(default_factory=lambda: np.zeros(0))
    is_4k: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # --- per-column index chains (flat, per column)
    col_pos: List[np.ndarray] = field(default_factory=list)   # flat row positions
    col_seg: List[np.ndarray] = field(default_factory=list)   # starts per chart
    col_dt: List[np.ndarray] = field(default_factory=list)    # gaps (sec)
    # --- per-boundary index chains
    bnd_pos: List[np.ndarray] = field(default_factory=list)
    bnd_seg: List[np.ndarray] = field(default_factory=list)
    bnd_dt: List[np.ndarray] = field(default_factory=list)
    # --- LN tails (flat, per chart sorted by tail time)
    tail_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    tail_chart: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    tail_seg: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    tail_next_head: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_next_head_col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    # --- note-level (not row-level) flat arrays, for C(s) and chord stats
    note_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    note_chart: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    note_seg: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    note_row: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    note_col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    @property
    def n_charts(self) -> int:
        return len(self.ids)

    def row_of_chart(self, c: int) -> slice:
        return slice(int(self.starts[c]), int(self.starts[c + 1]))


# ------------------------------------------------------------------ building
def _od_leniency(od: float) -> float:
    q = (64.5 - float(np.ceil(3.0 * od))) / 500.0
    x = 0.3 * np.sqrt(max(q, 1e-6))
    return float(min(x, 0.6 * (x - 0.09) + 0.09))


def build_batch(charts: List, rate: float = 1.0,
                parse_cache: bool = True) -> Batch:
    """Parse a list of dataset Chart objects into a flat Batch."""
    b = Batch()
    ids, t_list, mask_list, isln_list, lne_list = [], [], [], [], []
    note_t_list, note_chart_list, note_row_list = [], [], []
    x_list, od_list, kc_list, nnotes_list, dur_list, rows_list = [], [], [], [], [], []
    tail_t, tail_h, tail_col, tail_chart, tail_rows = [], [], [], [], []

    parsed: List[Optional[Notes]] = []
    for ci, ch in enumerate(charts):
        nt = cached_parse(ch.path, rate) if parse_cache else parse_file(ch.path, rate)
        parsed.append(nt)
        if nt is None or nt.n_notes < 2:
            continue
        ids.append(ch.id)
        # rows
        first = np.empty(nt.n_notes, dtype=bool)
        first[0] = True
        np.not_equal(nt.time[1:], nt.time[:-1], out=first[1:])
        row_of_note = np.cumsum(first) - 1
        R = int(row_of_note[-1]) + 1
        t = np.zeros(R, dtype=np.float64)
        t[row_of_note] = nt.time
        mask = np.zeros(R, dtype=np.int32)
        np.bitwise_or.at(mask, row_of_note, (1 << nt.col).astype(np.int32))
        isln = np.zeros(R, dtype=bool)
        np.logical_or.at(isln, row_of_note, nt.is_ln)
        lne = np.full(R, -1.0)
        np.maximum.at(lne, row_of_note, np.where(nt.is_ln, nt.ln_end, -1.0))

        t_list.append(t)
        mask_list.append(mask)
        isln_list.append(isln)
        lne_list.append(lne)
        rows_list.append(R)
        x_list.append(_od_leniency(nt.od))
        od_list.append(nt.od)
        kc_list.append(nt.key_count)
        nnotes_list.append(nt.n_notes)
        dur_list.append(float(t[-1] - t[0]))
        note_t_list.append(nt.time)
        note_chart_list.append(np.full(nt.n_notes, len(ids) - 1, np.int32))
        note_row_list.append(row_of_note)

        # LN tails
        li = np.flatnonzero(nt.is_ln)
        if li.size:
            order = np.argsort(nt.ln_end[li], kind="stable")
            li = li[order]
            tail_t.append(nt.ln_end[li])
            tail_h.append(nt.time[li])
            tail_col.append(nt.col[li].astype(np.int32))
            tail_chart.append(np.full(li.size, len(ids) - 1, np.int32))
            tail_rows.append(row_of_note[li])
        else:
            tail_t.append(np.zeros(0))
            tail_h.append(np.zeros(0))
            tail_col.append(np.zeros(0, np.int32))
            tail_chart.append(np.zeros(0, np.int32))
            tail_rows.append(np.zeros(0, np.int64))

    # ---- next head in column after each tail (vectorised per column)
    next_head = np.full(sum(len(a) for a in tail_t), np.inf)
    next_head_col = np.full(sum(len(a) for a in tail_t), -1, np.int32)
    if tail_t:
        tt = np.concatenate(tail_t)
        tc = np.concatenate(tail_chart)
        tl = np.concatenate(tail_col)
        for k in range(K):
            m = tl == k
            if not m.any():
                continue
            # for each chart, heads in column k
            for ci in np.unique(tc[m]):
                cm = m & (tc == ci)
                heads = None
                # gather heads of column k in chart ci
                nt = parsed[ci]
                if nt is None:
                    continue
                hk = nt.time[(nt.col == k)]
                j = np.searchsorted(hk, tt[cm], side="right")
                ok = j < hk.size
                idx = np.flatnonzero(cm)
                next_head[idx[ok]] = hk[j[ok]]
                next_head_col[idx[ok]] = k
        # fall back: next head in ANY column
        miss = ~np.isfinite(next_head)
        if miss.any():
            for ci in np.unique(tc[miss]):
                nt = parsed[ci]
                if nt is None:
                    continue
                mm = miss & (tc == ci)
                j = np.searchsorted(nt.time, tt[mm], side="right")
                ok = j < nt.n_notes
                idx = np.flatnonzero(mm)
                next_head[idx[ok]] = nt.time[j[ok]]
                next_head_col[idx[ok]] = nt.col[j[ok]]

    # ---- assemble flat arrays
    b.ids = ids
    n_per = np.array(rows_list, dtype=np.int64)
    b.starts = np.concatenate(([0], np.cumsum(n_per))).astype(np.int64)
    b.n = int(n_per.sum())
    b.t = np.concatenate(t_list) if t_list else np.zeros(0)
    b.chart = np.repeat(np.arange(len(ids), dtype=np.int32), n_per)
    off = (b.chart.astype(np.int64) * CHART_STRIDE)
    b.tg = b.t + off
    b.mask = np.concatenate(mask_list).astype(np.int32) if mask_list else np.zeros(0, np.int32)
    b.size = POPCOUNT[b.mask].astype(np.int32)
    b.is_ln = np.concatenate(isln_list) if isln_list else np.zeros(0, bool)
    b.ln_end = np.concatenate(lne_list) if lne_list else np.zeros(0)
    b.x = np.array(x_list)
    b.od = np.array(od_list)
    b.key_count = np.array(kc_list, np.int32)
    b.n_notes = np.array(nnotes_list, np.int64)
    b.duration = np.array(dur_list)
    b.is_4k = (b.key_count == 4)

    # dt (per row, step function length n-1 within chart)
    dt = np.diff(b.tg) / 1000.0
    dt = np.where(np.diff(b.chart) == 0, dt, 1.0)
    b.dt = dt

    # note-level flat
    b.note_t = np.concatenate(note_t_list) if note_t_list else np.zeros(0)
    b.note_chart = np.concatenate(note_chart_list) if note_chart_list else np.zeros(0, np.int32)
    b.note_row = np.concatenate(note_row_list) if note_row_list else np.zeros(0, np.int64)
    b.note_col = np.concatenate([nt.col for nt in parsed if nt is not None and nt.n_notes >= 2]).astype(np.int32) if note_t_list else np.zeros(0, np.int32)
    b.note_seg = np.concatenate(([0], np.cumsum(
        np.array([a.size for a in note_t_list], np.int64)))).astype(np.int64)

    # tails
    if tail_t:
        b.tail_t = np.concatenate(tail_t)
        b.tail_h = np.concatenate(tail_h)
        b.tail_col = np.concatenate(tail_col).astype(np.int32)
        b.tail_chart = np.concatenate(tail_chart).astype(np.int32)
        b.tail_seg = np.concatenate(([0], np.cumsum(
            np.array([a.size for a in tail_t], np.int64)))).astype(np.int64)
        b.tail_next_head = next_head
        b.tail_next_head_col = next_head_col
    else:
        b.tail_seg = np.zeros(1, np.int64)

    # ---- per-column chains
    kc = b.key_count
    for k in range(K):
        rows_k = np.flatnonzero((b.mask >> k) & 1).astype(np.int64)
        keep = np.ones(rows_k.size, dtype=bool)
        # drop duplicates within a chord (can't happen: one note per col per row)
        pos = rows_k[keep]
        b.col_pos.append(pos)
        cids = b.chart[pos]
        seg = np.searchsorted(cids, np.arange(len(ids) + 1), side="left").astype(np.int64)
        b.col_seg.append(seg)
        g = np.diff(b.tg[pos]) / 1000.0
        g = np.where(np.diff(cids) == 0, g, np.inf)
        b.col_dt.append(g)
    # ---- per-boundary chains
    for bd in range(N_BOUND):
        cols = [c for c in (bd - 1, bd) if 0 <= c < K]
        if not cols:
            b.bnd_pos.append(np.zeros(0, np.int64))
            b.bnd_seg.append(np.zeros(len(ids) + 1, np.int64))
            b.bnd_dt.append(np.zeros(0))
            continue
        flat = np.unique(np.concatenate([b.col_pos[c] for c in cols]))
        # b.col_pos are already globally sorted per column; merge sorted
        parts = [b.col_pos[c] for c in cols]
        if len(parts) == 1:
            merged = parts[0]
        else:
            merged = np.union1d(parts[0], parts[1])
        # remove rows where the boundary has <2 distinct columns used
        #  (single-column boundaries are legitimate: they measure that column)
        b.bnd_pos.append(merged.astype(np.int64))
        cids = b.chart[merged]
        seg = np.searchsorted(cids, np.arange(len(ids) + 1), side="left").astype(np.int64)
        b.bnd_seg.append(seg)
        g = np.diff(b.tg[merged]) / 1000.0
        g = np.where(np.diff(cids) == 0, g, np.inf)
        b.bnd_dt.append(g)
    return b


# --------------------------------------------------------------- disk cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cache")


def batch_cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"batch_{name}.npz")


def save_batch(b: Batch, name: str) -> None:
    data = {
        "starts": b.starts, "t": b.t, "mask": b.mask, "is_ln": b.is_ln,
        "ln_end": b.ln_end, "x": b.x, "od": b.od, "key_count": b.key_count,
        "n_notes": b.n_notes, "duration": b.duration,
        "note_t": b.note_t, "note_chart": b.note_chart, "note_row": b.note_row,
        "note_col": b.note_col,
        "note_seg": b.note_seg,
        "tail_t": b.tail_t, "tail_h": b.tail_h, "tail_col": b.tail_col,
        "tail_chart": b.tail_chart, "tail_seg": b.tail_seg,
        "tail_next_head": b.tail_next_head,
        "tail_next_head_col": b.tail_next_head_col,
        "ids": np.array(b.ids),
    }
    for k in range(K):
        data[f"cp{k}"] = b.col_pos[k]
        data[f"cs{k}"] = b.col_seg[k]
        data[f"cd{k}"] = b.col_dt[k]
    for bd in range(N_BOUND):
        data[f"bp{bd}"] = b.bnd_pos[bd]
        data[f"bs{bd}"] = b.bnd_seg[bd]
        data[f"bd{bd}"] = b.bnd_dt[bd]
    np.savez_compressed(batch_cache_path(name), **data)


def load_batch(name: str) -> Optional[Batch]:
    p = batch_cache_path(name)
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            b = Batch()
            b.ids = [str(s) for s in z["ids"]]
            b.starts = z["starts"]
            b.t = z["t"]
            b.mask = z["mask"]
            b.is_ln = z["is_ln"]
            b.ln_end = z["ln_end"]
            b.x = z["x"]
            b.od = z["od"]
            b.key_count = z["key_count"]
            b.n_notes = z["n_notes"]
            b.duration = z["duration"]
            b.note_t = z["note_t"]
            b.note_chart = z["note_chart"]
            b.note_row = z["note_row"]
            b.note_col = z["note_col"]
            b.note_seg = z["note_seg"]
            b.tail_t = z["tail_t"]
            b.tail_h = z["tail_h"]
            b.tail_col = z["tail_col"]
            b.tail_chart = z["tail_chart"]
            b.tail_seg = z["tail_seg"]
            b.tail_next_head = z["tail_next_head"]
            b.tail_next_head_col = z["tail_next_head_col"]
            b.n = int(b.t.size)
            b.chart = np.repeat(np.arange(len(b.ids), dtype=np.int32),
                                np.diff(b.starts))
            b.tg = b.t + b.chart.astype(np.int64) * CHART_STRIDE
            b.size = POPCOUNT[b.mask].astype(np.int32)
            b.is_4k = (b.key_count == 4)
            dt = np.diff(b.tg) / 1000.0
            b.dt = np.where(np.diff(b.chart) == 0, dt, 1.0)
            b.col_pos = [z[f"cp{k}"] for k in range(K)]
            b.col_seg = [z[f"cs{k}"] for k in range(K)]
            b.col_dt = [z[f"cd{k}"] for k in range(K)]
            b.bnd_pos = [z[f"bp{bd}"] for bd in range(N_BOUND)]
            b.bnd_seg = [z[f"bs{bd}"] for bd in range(N_BOUND)]
            b.bnd_dt = [z[f"bd{bd}"] for bd in range(N_BOUND)]
            return b
    except Exception:                                        # noqa: BLE001
        return None


# ===================== core/model.py =====================


import numpy as np


K = 7
N_BOUND = 8
EPS = 1e-9

# popcount lookup for 7-bit row masks
_POPCOUNT7 = np.array([bin(i).count("1") for i in range(128)], dtype=np.int32)

# 7K: 0,1,2 left hand | 3 thumb | 4,5,6 right hand
HAND = np.array([0, 0, 0, 1, 2, 2, 2], dtype=np.int32)
_IDX7 = np.arange(K, dtype=np.int64)
AIN_GROUP = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int32)   # ring/mid/idx/thumb
BOUND_GROUP = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.int32)
SHIELD_LOOKBACK = 16


# ------------------------------------------------------------------ helpers
def _step_scatter(idx0: np.ndarray, idx1: np.ndarray, val: np.ndarray,
                  n: int) -> np.ndarray:
    """Step function equal to val[j] on rows [idx0[j], idx1[j]).

    Ends at or beyond the last row are absorbed into a sentinel slot so the
    prefix sum always returns to zero (never leaks to the end of the array).
    """
    m = n + 1
    a = np.bincount(np.minimum(idx0, m), weights=val, minlength=m + 1)
    b = np.bincount(np.minimum(idx1, m), weights=val, minlength=m + 1)
    return np.cumsum(a[:m] - b[:m])[:n]


def _chain_scatter(pos: np.ndarray, val: np.ndarray, n: int,
                   same: np.ndarray) -> np.ndarray:
    v = np.where(same, val, 0.0)
    return _step_scatter(pos[:-1], pos[1:], v, n)


def _run_lengths(keep: np.ndarray) -> np.ndarray:
    out = np.zeros(keep.size, dtype=np.float64)
    if keep.size == 0 or not keep.any():
        return out
    brk = np.concatenate(([True], ~keep[:-1])) & keep
    ids = np.cumsum(brk) - 1
    lens = np.bincount(ids[keep])
    out[keep] = lens[ids[keep]]
    return out


class Struct:
    """Parameter-free per-row arrays, built once per batch."""
    __slots__ = ("n", "n_charts", "starts", "counts", "chart", "t", "tg",
                 "tlo", "thi", "dt", "dt_ms", "mask", "size", "x", "is_ln",
                 "ln_end", "dcol", "act", "usage", "anchor_raw", "held",
                 "heldmask", "ln_body_ms", "C", "Ks", "cumsize", "note_row",
                 "note_impulse", "col_chain", "bnd_chain", "tail", "is_4k",
                 "_tail_lo", "_tail_hi", "jlo", "jhi", "_wmean_ops",
                 "_chord_ops",
                 "n_notes", "duration", "dt_diff", "chart_same", "batch")


# _wmean interpolation operators are cached ON the struct object itself.
# (Keying a module-level dict by id(s) is unsound: a rebuilt struct can
# reuse a freed id and silently inherit the previous dataset's index arrays.)


def _wmean_index(s: Struct, W: float):
    """Cached (j_hi, z_hi, j_lo, z_lo) interpolation operator for window W."""
    ops = getattr(s, "_wmean_ops", None)
    if ops is None:
        ops = {}
        s._wmean_ops = ops
    op = ops.get(W)
    if op is not None:
        return op
    tg = s.tg
    zhi = np.minimum(tg + 0.5 * W, s.thi)
    zlo = np.maximum(tg - 0.5 * W, s.tlo)
    jhi = np.clip(np.searchsorted(tg, zhi, side="right") - 1, s.jlo, s.jhi)
    jlo = np.clip(np.searchsorted(tg, zlo, side="right") - 1, s.jlo, s.jhi)
    op = (jhi, zhi - tg[jhi], jlo, zlo - tg[jlo])
    if len(ops) > 64:
        ops.clear()
    ops[W] = op
    return op


def _wmean(s: Struct, val: np.ndarray, W: float) -> np.ndarray:
    """Centred moving average of the step function `val` over a W-ms window.

    The interpolation index is clipped to the *owning chart's* row range --
    clipping to [0, n-2] would silently reach across the chart-id time gap.
    """
    inc = np.zeros(s.n)
    inc[:-1] = val[:-1] * s.dt_diff
    # I[i] = integral from the chart's first row up to t[i]  (exclusive of i)
    I = seg_cumsum(inc, s.starts) - inc
    jhi, dhi, jlo, dlo = _wmean_index(s, W)
    out = ((I[jhi] + val[jhi] * dhi) - (I[jlo] + val[jlo] * dlo)) / W
    return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _wcount(s: Struct, w: np.ndarray, W: float) -> np.ndarray:
    tg = s.tg
    c = np.concatenate(([0.0], np.cumsum(w)))

    def cnt(z):
        zc = np.clip(z, s.tlo, s.thi + 1.0)
        return c[np.searchsorted(tg, zc, side="left")]

    return cnt(tg + 0.5 * W) - cnt(tg - 0.5 * W)


# ------------------------------------------------------ 和弦耦合
# 精确时间戳的和弦在 1 毫秒扰动下会分裂。
# chord_tau_c 大于 0 时，相邻行的音符按耦合权重分摊和弦尺寸，
# 有效和弦尺寸连续变化。耦合算子挂在行结构上缓存。
def _chord_coupling(s: Struct, tau_c: float):
    """(wf, wb): forward/backward adjacent-row coupling weights.

    wf[r] = w(t[r+1] - t[r]) for same-chart successor, else 0;
    wb[r] = wf[r-1].  Cached on the struct per tau_c (same pattern as
    _wmean_ops).
    """
    ops = getattr(s, "_chord_ops", None)
    if ops is None:
        ops = {}
        s._chord_ops = ops
    got = ops.get(tau_c)
    if got is None:
        w = np.clip(1.0 - s.dt_ms / tau_c, 0.0, 1.0)
        wf = np.zeros(s.n)
        wf[:-1] = np.where(s.chart_same, w[:-1], 0.0)
        wb = np.zeros(s.n)
        wb[1:] = np.where(s.chart_same, w[:-1], 0.0)
        got = (wf, wb)
        if len(ops) > 16:
            ops.clear()
        ops[tau_c] = got
    return got


def _couple(x: np.ndarray, wf: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """x plus adjacent-row neighbours discounted by the coupling weights."""
    fwd = np.zeros_like(x)
    fwd[:-1] = x[1:]
    bwd = np.zeros_like(x)
    bwd[1:] = x[:-1]
    return x + wf * fwd + wb * bwd


def _tri_window_count(s: Struct, val: np.ndarray, tau_c: float) -> np.ndarray:
    """Triangular-window count: sum_r' val[r'] * max(0, 1 - |t[r']-t[r]|/tau_c).

    Chart-safe O(n) via cumsums + searchsorted (window clipped to the
    owning chart).  With val = row size this is the graded effective chord
    size: it reaches the full group size for a chord split into rows a few
    ms apart, and equals size exactly for isolated rows.
    """
    tg = s.tg
    t = s.t.astype(np.float64)
    n = s.n
    cv = np.concatenate(([0.0], np.cumsum(val)))
    cvt = np.concatenate(([0.0], np.cumsum(val * t)))
    lo = np.clip(tg - tau_c, s.tlo, s.thi + 1.0)
    hi = np.clip(tg + tau_c, s.tlo, s.thi + 1.0)
    jl = np.searchsorted(tg, lo, side="left")
    jh = np.searchsorted(tg, hi, side="left")
    r = np.arange(n)
    # left part: rows [jl, r) with |dt| = t[r] - t[r']
    sl = cv[r] - cv[jl]
    slt = cvt[r] - cvt[jl]
    left = sl - (t * sl - slt) / tau_c
    # right part: rows [r, jh) with |dt| = t[r'] - t[r]
    sr = cv[jh] - cv[r]
    srt = cvt[jh] - cvt[r]
    right = sr - (srt - t * sr) / tau_c
    return np.maximum(
        np.nan_to_num(left + right, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _edge_ramp(x, cutoff) -> np.ndarray:
    """Raised-cosine fade: 1 up to 80% of `cutoff`, 0 at/above `cutoff`.

    Used by r_soft_edges to replace hard time cutoffs (r_dt_max,
    r_order_tau) with a continuous outer-20% fade.
    """
    x0 = 0.8 * cutoff
    u = np.clip((x - x0) / max(cutoff - x0, 1e-9), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * u))


def _abar_val(dd, mx, p) -> np.ndarray:
    """Abar column-pair multiplier.

    abar_weld 为 1 时中间段取连续形式，在低阈值处连续；为 0 时取分段
    形式，在接缝处有跳变。高阈值处两种形式都由 min(., 1) 饱和保证连续。
    """
    lo = np.minimum(p["abar_c0"] + p["abar_mx"] * mx, 1.0)
    if p.get("abar_weld", 0.0):
        mid = np.minimum(p["abar_c0"] + p["abar_k"] * (dd - p["abar_thr_lo"])
                         + p["abar_mx"] * mx, 1.0)
    else:
        mid = np.minimum(p["abar_c1"] + p["abar_k"] * dd +
                         p["abar_mx"] * mx, 1.0)
    return np.where(dd < p["abar_thr_lo"], lo,
                    np.where(dd < p["abar_thr_hi"], mid, 1.0))


# ------------------------------------------------------------------- struct
def build_struct(b, p: dict | None = None) -> Struct:
    s = Struct()
    s.batch = b  # keep a reference for components that need raw batch data
    n = b.n
    starts = b.starts
    counts = np.diff(starts)
    s.n = n
    s.n_charts = b.n_charts
    s.starts = starts
    s.counts = counts
    s.chart = b.chart
    s.t = b.t
    s.tg = b.tg
    s.tlo = np.repeat(b.tg[starts[:-1]], counts)
    s.thi = np.repeat(b.tg[starts[1:] - 1], counts)
    dtd = np.diff(b.tg)
    s.chart_same = (np.diff(b.chart) == 0)
    s.dt_diff = np.where(s.chart_same, dtd, 0.0)
    dt = np.zeros(n)
    dt[:-1] = s.dt_diff
    s.dt_ms = np.maximum(dt, 0.0)
    s.dt = np.where(dt > 0, dt / 1000.0, 1.0)
    s.mask = b.mask
    s.size = b.size
    s.x = b.x
    s.is_ln = b.is_ln
    s.ln_end = b.ln_end
    s.is_4k = b.is_4k
    s.n_notes = b.n_notes
    s.duration = b.duration
    s.cumsize = np.concatenate(([0.0], np.cumsum(b.size.astype(np.float64))))
    s.jlo = np.repeat(starts[:-1], counts)
    s.jhi = np.repeat(np.maximum(starts[1:] - 1, 0), counts)

    # ---- column chains
    chain = []
    for k in range(K):
        pos = b.col_pos[k]
        if pos.size:
            cid = b.chart[pos]
            same = np.zeros(pos.size, dtype=bool)
            same[:-1] = (cid[:-1] == cid[1:])
            g = np.zeros(max(pos.size - 1, 0))
            if pos.size >= 2:
                g = np.where(same[:-1], np.diff(b.tg[pos]) / 1000.0, np.inf)
        else:
            cid = np.zeros(0, np.int32)
            same = np.zeros(0, bool)
            g = np.zeros(0)
        chain.append(dict(pos=pos, g=g, same=same, cid=cid,
                          tc=b.t[pos] if pos.size else np.zeros(0),
                          tgc=b.tg[pos] if pos.size else np.zeros(0),
                          seg=b.col_seg[k]))
    s.col_chain = chain

    bchain = []
    for bd in range(N_BOUND):
        pos = b.bnd_pos[bd]
        if pos.size:
            cid = b.chart[pos]
            same = np.zeros(pos.size, dtype=bool)
            same[:-1] = (cid[:-1] == cid[1:])
            g = np.where(same[:-1], np.diff(b.tg[pos]) / 1000.0, np.inf) \
                if pos.size >= 2 else np.zeros(0)
        else:
            g = np.zeros(0)
        # effective column of each chain element within the boundary's
        # column pair {bd-1, bd}: left-only -> bd-1, right-only -> bd,
        # chord on both -> midpoint bd-0.5.  Drives the inward/outward
        # (内外切) direction split of the Xbar pairwise term.
        if pos.size and 0 < bd < K:
            m = b.mask[pos].astype(np.int64)
            has_l = ((m >> (bd - 1)) & 1).astype(np.float64)
            has_r = ((m >> bd) & 1).astype(np.float64)
            e = (has_l * (bd - 1) + has_r * bd) / np.maximum(has_l + has_r, 1.0)
        else:
            e = np.full(pos.size, float(min(max(bd, 0), K - 1)))
        bchain.append(dict(pos=pos, g=g, e=e,
                           same=(np.zeros(pos.size, bool) if pos.size == 0
                                 else bchain_same(b, pos))))
    s.bnd_chain = bchain

    # ---- per-row column gap
    dcol = np.full((K, n), 1e9, dtype=np.float32)
    for k in range(K):
        c = chain[k]
        if c["pos"].size >= 2:
            dcol[k] = _chain_scatter(c["pos"], np.where(np.isfinite(c["g"]),
                                                        c["g"], 1e9),
                                     n, c["same"][:-1]).astype(np.float32)
    s.dcol = dcol

    # ---- active columns + column usage
    act = np.zeros((K, n), dtype=bool)
    usage = np.zeros((K, n), dtype=np.float32)
    for k in range(K):
        imp = np.zeros(n)
        if chain[k]["pos"].size:
            imp[chain[k]["pos"]] = 1.0
        act[k] = _wcount(s, imp, 300.0) > 0
        usage[k] = _wcount(s, imp, 800.0).astype(np.float32)
    s.act = act
    s.usage = usage

    u = np.sort(usage, axis=0)[::-1, :]
    nz = u > 1e-6
    ratios = np.where(nz[1:], u[1:] / np.maximum(u[:-1], EPS), 0.0)
    walk = np.sum(np.where(nz[:-1], u[:-1] *
                           np.maximum(1.0 - 4.0 * (0.5 - ratios) ** 2, 0.0), 0.0),
                  axis=0)
    mx = np.sum(np.where(nz[:-1], u[:-1], 0.0), axis=0)
    s.anchor_raw = np.where(mx > EPS, walk / np.maximum(mx, EPS), 0.0)

    # ---- LN held count / held column mask
    # ln_eff_tail_ms: LN effective tail reduction.  For density/lock
    # calculations the player is already preparing to release near the tail,
    # so the LN occupies less time than its paper length.  We shorten the
    # effective end by this amount (ms) for held/density purposes only.
    # Release calculations (Rbar) still use the true ln_end.
    ln_eff = float((p or {}).get("ln_eff_tail_ms", 0.0))
    # ln_note_level 为 1 时，按住状态按每条长键的头尾数组构建；行级构建
    # 会把与长键头同行的点键误标成长键头。0 表示行级构建。
    note_level = float((p or {}).get("ln_note_level", 0.0)) != 0.0
    held = np.zeros(n)
    heldmask = np.zeros(n, dtype=np.int32)
    if note_level and b.tail_t.size:
        for k in range(K):
            tk = np.flatnonzero(b.tail_col == k)
            if tk.size == 0:
                continue
            h = b.tail_h[tk]
            e = b.tail_t[tk]
            ch = b.tail_chart[tk].astype(np.int64)
            e_eff = np.maximum(e - ln_eff, h + 1.0)  # never shorter than 1ms
            j0 = np.searchsorted(b.tg, h + ch * (1 << 30) + 1.0, side="left")
            j1 = np.searchsorted(b.tg, e_eff + ch * (1 << 30), side="left")
            d = np.zeros(n + 1)
            np.add.at(d, np.minimum(j0, n), 1.0)
            np.add.at(d, np.minimum(j1, n), -1.0)
            hk = np.maximum(np.cumsum(d)[:n], 0.0)
            held += hk
            heldmask |= np.where(hk > 0, (1 << k), 0).astype(np.int32)
    else:
        for k in range(K):
            c = chain[k]
            if c["pos"].size == 0:
                continue
            li = c["pos"][b.is_ln[c["pos"]]]
            if li.size == 0:
                continue
            h = b.t[li]
            e = b.ln_end[li]
            e_eff = np.maximum(e - ln_eff, h + 1.0)  # never shorter than 1ms
            j0 = np.searchsorted(b.tg, b.tg[li] + 1.0, side="left")
            j1 = np.searchsorted(b.tg, e_eff + b.chart[li].astype(np.int64) * (1 << 30),
                                 side="left")
            d = np.zeros(n + 1)
            np.add.at(d, np.minimum(j0, n), 1.0)
            np.add.at(d, np.minimum(j1, n), -1.0)
            hk = np.maximum(np.cumsum(d)[:n], 0.0)
            held += hk
            heldmask |= np.where(hk > 0, (1 << k), 0).astype(np.int32)
    s.held = held
    s.heldmask = heldmask
    s.ln_body_ms = held * s.dt_ms

    # ---- C (local note count, +-500 ms) and Ks
    note_row = b.note_row + np.repeat(starts[:-1], b.n_notes)
    s.note_row = note_row
    imp = np.zeros(n)
    np.add.at(imp, note_row, 1.0)
    s.note_impulse = imp
    s.C = _wcount(s, imp, 1000.0)
    s.Ks = np.maximum(act.sum(axis=0), 1).astype(np.float64)

    # ---- tails
    s.tail = None
    if b.tail_t.size:
        cid = b.tail_chart.astype(np.int64)
        ch32 = b.tail_chart
        tg_tail = b.tail_t + cid * (1 << 30)
        # clip into the owning chart's row range (a tail can land past the
        # last note head, which would otherwise spill into the next chart)
        lo_r = b.starts[:-1][ch32]
        hi_r = np.maximum(b.starts[1:][ch32] - 1, lo_r)
        j0 = np.clip(np.searchsorted(b.tg, tg_tail, side="left"), lo_r, hi_r)
        # per-chart row bounds reused by _rbar
        s._tail_lo = lo_r
        s._tail_hi = hi_r
        # next event after the tail: next tail in the chart, or next head
        m = b.tail_t.size
        nxt_tail = np.full(m, np.inf)
        nxt_tail_col = np.full(m, -1, np.int64)
        idx = np.arange(m - 1)
        ok = b.tail_chart[idx] == b.tail_chart[idx + 1]
        nxt_tail[idx[ok]] = b.tail_t[idx[ok] + 1]
        nxt_tail_col[idx[ok]] = b.tail_col[idx[ok] + 1]
        is_tail = nxt_tail <= b.tail_next_head - 1e-9
        nxt = np.where(is_tail, nxt_tail, b.tail_next_head)
        nxt_col = np.where(is_tail, nxt_tail_col,
                           b.tail_next_head_col.astype(np.int64))
        nxt_col = np.where(np.isfinite(nxt), nxt_col, b.tail_col.astype(np.int64))
        s.tail = dict(t=b.tail_t, h=b.tail_h, col=b.tail_col,
                      chart=b.tail_chart, seg=b.tail_seg, j0=j0,
                      nh=b.tail_next_head, nhc=b.tail_next_head_col,
                      nt=nxt_tail,
                      nxt=nxt, nxt_col=nxt_col, is_tail=is_tail, tg=tg_tail)
    return s


def bchain_same(b, pos):
    cid = b.chart[pos]
    same = np.zeros(pos.size, dtype=bool)
    same[:-1] = (cid[:-1] == cid[1:])
    return same


# ------------------------------------------------------------------- curves
def _jack_kernel(g, xrow, p, od_mult):
    xk = np.power(xrow, od_mult)
    g = np.maximum(g, 1e-4)
    base = (1.0 / g) * (1.0 / (g + p["j_c1"] * np.power(np.maximum(xk, 1e-6), 0.25)))
    if p.get("j_nerf_a", 0.0):
        base = base * (1.0 - p["j_nerf_a"] *
                       np.power(p["j_nerf_off"] + np.abs(g - p["j_nerf_c"]), -4.0))
    return base


def build_jack_stats(b, s: Struct, p: dict) -> list:
    """Per-column jack-gap arrays.

    Extracted verbatim from `compute_curves` so the Cbar v2 component
    (`core.components.lncoord_v2`) can build per-column lock-modulated
    curves without duplicating this loop.  Pure move -- no arithmetic change.
    """
    xrow = s.x[s.chart]
    gap_thr = float(p["ja_gap_thr"]) / 1000.0
    h3_tau = max(float(p["j_triple_tau"]), 1.0) / 1000.0
    # chord_tau_c 大于 0 时，按攻击组计数行数，而不是原始行数。
    # 攻击组之间没有耦合：被拆成 1 毫秒多行的和弦不再让同列击打
    # 之间的行数翻倍，双押叠通道的负载也不再减半。
    tau_c = float(p.get("chord_tau_c", 0.0))
    att_cum = None
    if tau_c > 0.0:
        wf, wb = _chord_coupling(s, tau_c)
        att_cum = np.concatenate(
            ([0.0], np.cumsum((wb == 0.0).astype(np.float64))))
    # 同列三连增益按谱面键数分别取值：7 键与 4 键各一个系数，
    # 4 键系数小于 0 时表示沿用 7 键的值。
    g7 = float(p["j_triple_gain"])
    g4k_v = float(p.get("j_triple_gain_4k", -1.0))
    g4k = g7 if g4k_v < 0.0 else g4k_v
    is4k_chart = b.is_4k
    stats = []
    for k in range(K):
        c = s.col_chain[k]
        pos = c["pos"]
        if pos.size < 2:
            stats.append(None)
            continue
        g = c["g"]
        valid = np.isfinite(g) & (np.diff(c["cid"]) == 0)
        gs = np.maximum(np.where(valid, g, 1.0), 1e-4)
        if att_cum is not None:
            nrows = np.maximum(att_cum[pos[1:]] - att_cum[pos[:-1]], 1.0)
        else:
            nrows = np.maximum(pos[1:] - pos[:-1], 1).astype(np.float64)
        total = s.cumsize[pos[1:]] - s.cumsize[pos[:-1]]
        other = np.maximum(total - 1.0, 0.0)
        same_step = np.diff(c["cid"]) == 0
        runlen = _run_lengths((g <= gap_thr) & same_step)
        # same-column triple (h_jack3).  The three notes must be in ONE
        # chart -- using local times without that guard lets a chart
        # boundary produce an enormous negative span and blow the channel up.
        trip = np.zeros(max(pos.size - 1, 0))
        tg3 = s.t[pos]
        if tg3.size >= 3 and same_step.size >= 2:
            same3 = same_step[:-1] & same_step[1:]
            span_s = (tg3[2:] - tg3[:-2]) / 1000.0     # ms -> s
            trip[:-1] = np.where(
                same3, np.maximum(0.0, 1.0 - span_s / (2.0 * h3_tau)), 0.0)
        # fast same-column (fj): pairs with gap < 150 ms priced (1/g)(1-g/0.15)
        cid_gap = c["cid"][:-1]
        fj = np.where(same_step & (gs < 0.150),
                      (1.0 / gs) * np.maximum(1.0 - gs / 0.150, 0.0), 0.0)
        # per-gap triple gain by frame
        tgain = np.where(is4k_chart[cid_gap], g4k, g7)
        stats.append(dict(pos=pos, gs=gs, valid=valid, nrows=nrows,
                          other=other, load=total / nrows, runlen=runlen,
                          trip=trip, tgain=tgain, fj=fj, xr=xrow[pos[:-1]],
                          kern=_jack_kernel(gs, xrow[pos[:-1]], p,
                                            float(p["od_mult_jm"]))))
    return stats


def compute_curves(b, s: Struct, p: dict) -> dict:
    n = s.n
    xrow = s.x[s.chart]

    # ============================================================ jack family
    agg_pow = max(float(p["j_agg_pow"]), 0.25)
    triple_gain = float(p["j_triple_gain"])
    mj_dilute = float(p["mj_dilute"])
    mj_exp = float(p["mj_dilute_exp"])
    cj_exp = float(p["cj_size_exp"])
    ja_len_exp = float(p["ja_len_exp"])
    ja_coord_exp = float(p["ja_coord_exp"])
    ja_floor = float(p["ja_len_floor"])
    cj_norm = max(float(p["cj_norm"]), 1e-3)
    ja_norm = max(float(p["ja_norm"]), 1e-3)
    w_jm = float(p["w_jm"])
    # Per-channel smoothing width / OD response for the chordjack and anchor
    # channels.  w<=0 / od_mult<0 means "inherit the minijack values".
    # The anchor channel is redundant, so its knobs stay parked; the live
    # chordjack channel may carry its own values.
    od_jm = float(p["od_mult_jm"])
    w_jc = float(p.get("w_jc", 0.0))
    w_jc = w_jm if w_jc <= 0.0 else w_jc
    w_ja = float(p.get("w_ja", 0.0))
    w_ja = w_jm if w_ja <= 0.0 else w_ja
    od_jc = float(p.get("od_mult_jc", -1.0))
    od_jc = od_jm if od_jc < 0.0 else od_jc
    od_ja = float(p.get("od_mult_ja", -1.0))
    od_ja = od_jm if od_ja < 0.0 else od_ja

    stats = build_jack_stats(b, s, p)

    def jack_channel(kind: str, w_ms: float, od_mult: float) -> np.ndarray:
        num = np.zeros(n)
        den = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            if kind == "plain":
                mod = np.ones(st["pos"].size - 1)
            elif kind == "mj":
                mod = (1.0 / (1.0 + mj_dilute *
                              np.power(st["other"] / st["nrows"], mj_exp))
                       * (1.0 + st["tgain"] * st["trip"]))
            elif kind == "cj":
                mod = np.power(np.maximum(st["load"], 1e-6) / cj_norm, cj_exp)
            else:  # "ja"
                e = np.maximum(st["runlen"] + 2.0 - ja_floor, 0.0)
                runf = np.power(e / (e + ja_norm), ja_len_exp)
                mod = runf * np.power(1.0 + st["other"] / st["nrows"], ja_coord_exp)
            kern = st["kern"] if od_mult == od_jm else \
                _jack_kernel(st["gs"], st["xr"], p, od_mult)
            v = np.where(st["valid"], kern * mod, 0.0)
            cur = np.maximum(_wmean(s, _chain_scatter(st["pos"], v, n,
                                                      st["valid"]), w_ms), 0.0)
            w = _chain_scatter(st["pos"],
                               np.where(st["valid"], 1.0 / st["gs"], 0.0),
                               n, st["valid"])
            num += np.power(cur, agg_pow) * w
            den += w
        val = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0)
        return np.power(np.maximum(val, 0.0), 1.0 / agg_pow)

    need = set()
    if p["c_jm"] or p["t_jack_mix"] or p["cb_lockjack"]:
        need.add("mj")
    if p["c_jc"]:
        need.add("cj")
    if p["c_ja"] or p["cb_lockanchor"]:
        need.add("ja")
    if p["c_s"] or p["a_p"]:
        need.add("plain") if False else None
    need.add("plain")            # Jbar is also the Cbar lock-jack carrier
    Jbar = jack_channel("plain", w_jm, od_jm)
    Jm = jack_channel("mj", w_jm, od_jm) if "mj" in need else Jbar

    # fast same-column (fj) channel: rapid same-column pairs.  Per-frame:
    # c_fj for 7K rows, c_fj_4k for 4K rows (0 = off for that frame).
    # NB: FJ is merged into Jm BEFORE the Jc/Ja fallbacks below, so a
    # fallback always captures the final Jm.
    c_fj = float(p.get("c_fj", 0.0))
    c_fj_4k = float(p.get("c_fj_4k", 0.0))
    if c_fj != 0.0 or c_fj_4k != 0.0:
        FJ = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            FJ += _chain_scatter(st["pos"], np.where(st["valid"], st["fj"],
                                                     0.0), n, st["valid"])
        FJ = _wmean(s, FJ, w_jm)
        c_fj_row = np.where(s.is_4k[s.chart], c_fj_4k, c_fj)
        Jm = Jm + c_fj_row * FJ

    Jc = jack_channel("cj", w_jc, od_jc) if "cj" in need else Jm
    Ja = jack_channel("ja", w_ja, od_ja) if "ja" in need else Jm

    # ================================================================== Pbar
    xp = np.power(xrow, float(p["od_mult_p"]))
    d = s.dt
    tau_c = float(p.get("chord_tau_c", 0.0))
    d_fp = d
    if tau_c > 0.0:
        wf, wb = _chord_coupling(s, tau_c)
        # the synchronous-note term uses the gap to the next ATTACK (rows
        # with no coupling from their predecessor), skipping rows absorbed
        # into this row's chord group -- otherwise a split chord's
        # carrying row reads an artificially tiny interval and loses fp.
        att_idx = np.flatnonzero(wb == 0.0)
        j = np.searchsorted(att_idx, np.arange(n), side="right")
        has = j < att_idx.size
        na = att_idx[np.minimum(j, att_idx.size - 1)]
        ok_na = has & (s.chart[na] == s.chart)
        d_fp = np.where(ok_na, (s.t[na] - s.t) / 1000.0, d)
        d_fp = np.where(d_fp > 0, d_fp, d)
    q = np.minimum(d_fp - xp / 2.0, xp / 6.0)
    fp = np.power(np.maximum(p["p_scale"] / xp *
                             (1.0 - (p["p_lam3"] / xp) * q * q), 0.0), 0.25)
    r = 7.5 / d
    lo, hi = float(p["p_boost_lo"]), float(p["p_boost_hi"])
    bst = np.where((r > lo) & (r < hi),
                   1.0 + p["p_boost"] * (r - lo) * np.power(r - hi, 2.0), 1.0)
    vv = 1.0 + p["p_lam2"] * s.ln_body_ms / \
        np.power(np.maximum(s.dt_ms, 1.0), float(p["p_v_norm"]))
    chw = np.ones(K + 1)
    chw[2:K + 1] = [p.get(f"p_chw{i}", 1.0) for i in range(2, K + 1)]
    if tau_c > 0.0:
        # 和弦连续化：
        #  * u 为三角窗内的音符数，即有效和弦尺寸（被拆成几毫秒多行的
        #    和弦仍保持接近完整的组尺寸）；定价表在 u 处线性插值。
        #  * att 为攻击折扣：后一行相对前一行只计零点几分之一个新攻击
        #    （被前一行的和弦组吸收），否则密度的 1/d 强度会把一个拆开
        #    的和弦算成多次完整攻击。折扣只作用于速率部分：长键身体增益
        #    跟随每行自己的区间（行求和下与网格划分无关），不能打折，
        #    否则身体质量会搬到被折扣的子行上。
        u = np.clip(_tri_window_count(s, s.size.astype(np.float64), tau_c),
                    1.0, float(K))
        i0 = np.minimum(u.astype(np.int64), K - 1)
        frac = u - i0
        chw_u = chw[i0] * (1.0 - frac) + chw[i0 + 1] * frac
        att = 1.0 - wb
        if p["p_bv_max"] > 0.5:
            mix = att * np.maximum(bst, 1.0) + np.maximum(vv - 1.0, 0.0)
        else:
            mix = att * bst * vv
    else:
        chw_u = chw[np.minimum(s.size, K)]
        mix = np.maximum(bst, vv) if p["p_bv_max"] > 0.5 else bst * vv
    inc = (1.0 / d) * fp * chw_u * mix
    anchor_mult = 1.0 + p["an_on_p"] * np.minimum(
        s.anchor_raw - p["an_a0"],
        p["an_cubic"] * np.power(s.anchor_raw - p["an_a1"], 3.0))
    inc = np.minimum(inc * anchor_mult,
                     np.maximum(inc, inc * float(p["p_sat_b"]) - float(p["p_sat_a"])))
    if p["p_burst3"]:
        inc = inc + p["p_burst3"] * _wcount(s, s.note_impulse, 100.0) * 10.0
    if p["p_burst4"]:
        inc = inc + p["p_burst4"] * _wcount(s, s.note_impulse, 150.0) * 6.6667
    Pbar = _wmean(s, inc, float(p["w_p"]))

    # ================================================================== Abar
    A = np.ones(n)
    dk = s.dcol
    act = s.act
    # abar_active 为 0 时序号相邻的列配对；为 1 时活跃列相邻才配对，
    # 4 键嵌入的谱面还会跨过拇指列。
    if p["abar_active"] >= 0.5:
        prev_d = np.full(n, -1.0)
        prev_a = np.full(n, -1.0)
        for k in range(K):
            d0, a0 = prev_d, prev_a
            on = (a0 >= 0.0) & act[k]
            d1 = dk[k]
            mx = np.maximum(np.where(on, d0, 0.0), d1)
            dd = np.abs(np.where(on, d0, d1) - d1) + p["abar_mx_w"] * np.maximum(
                0.0, mx - p["abar_mx_thr"])
            val = _abar_val(dd, mx, p)
            A *= np.where(on, val, 1.0)
            prev_d = np.where(act[k], d1, d0)
            prev_a = np.where(act[k], 1.0, a0)
    else:
        for k in range(K - 1):
            on = act[k] & act[k + 1]
            d0, d1 = dk[k], dk[k + 1]
            mx = np.maximum(d0, d1)
            dd = np.abs(d0 - d1) + p["abar_mx_w"] * np.maximum(
                0.0, mx - p["abar_mx_thr"])
            val = _abar_val(dd, mx, p)
            A *= np.where(on, val, 1.0)
    Abar = _wmean(s, A, float(p["w_a"])) * p["abar_scale"]

    # ================================================================== Xbar
    x_x = np.power(xrow, float(p["od_mult_x"]))
    cw = np.array([p["x_cw0"], p["x_cw1"], p["x_cw2"], p["x_cw3"]])
    # 成对项的内切/外切方向拆分：d 大于 0 表示朝拇指列移动（内切）；
    # 和弦参与的切换按有效列取半权。0/0 表示对称（无方向拆分）。
    x_din = float(p.get("x_din", 0.0))
    x_dout = float(p.get("x_dout", 0.0))
    Xsum = np.zeros(n)
    fc_rows = []
    for bd in range(N_BOUND):
        c = s.bnd_chain[bd]
        if c["pos"].size < 2:
            fc_rows.append(np.zeros(n))
            continue
        if tau_c > 0.0:
            # 相隔不足 tau_c 的边界事件合并成一次攻击：被拆成几毫秒多行
            # 的和弦是一次边界事件，取整组的跨度，而不是多次钳制最大值。
            pos = c["pos"]
            cid = s.chart[pos]
            bg = np.full(pos.size, np.inf)
            bg[1:] = np.where(cid[1:] == cid[:-1],
                              (s.tg[pos[1:]] - s.tg[pos[:-1]]) / 1000.0,
                              np.inf)
            keep = bg >= tau_c / 1000.0
            e_arr = c["e"][keep]
            pos = pos[keep]
            cid = cid[keep]
            g = np.full(max(pos.size - 1, 0), np.inf)
            same = np.zeros(pos.size, dtype=bool)
            if pos.size >= 2:
                g = np.where(cid[1:] == cid[:-1],
                             (s.tg[pos[1:]] - s.tg[pos[:-1]]) / 1000.0,
                             np.inf)
                same[:-1] = cid[:-1] == cid[1:]
        else:
            pos = c["pos"]
            g = c["g"]
            same = c["same"]
            e_arr = c["e"]
        g = np.where(np.isfinite(g), g, 1.0)
        gx = np.maximum(g, x_x[pos[:-1]])
        Xb = p["x_amp"] * np.power(gx, -2.0)
        if x_din != 0.0 or x_dout != 0.0:
            dd = np.abs(e_arr[:-1] - 3.0) - np.abs(e_arr[1:] - 3.0)
            Xb = Xb * (1.0 + x_din * np.maximum(dd, 0.0)
                       + x_dout * np.maximum(-dd, 0.0))
        Xsum += cw[BOUND_GROUP[bd]] * _chain_scatter(
            pos, np.where(same[:-1], Xb, 0.0), n, same[:-1])
        base = np.maximum(np.maximum(g, p["x_fc_floor"]), 0.75 * x_x[pos[:-1]])
        fc = np.maximum(p["x_fc_a"] * np.power(base, -2.0) - p["x_fc_off"], 0.0)
        fc_rows.append(_chain_scatter(pos,
                                      np.where(same[:-1], fc, 0.0),
                                      n, same[:-1]))
    for bd in range(N_BOUND - 1):
        w = np.sqrt(cw[BOUND_GROUP[bd]] * cw[BOUND_GROUP[bd + 1]])
        Xsum += p["x_fc_w"] * w * np.sqrt(np.maximum(fc_rows[bd] *
                                                     fc_rows[bd + 1], 0.0))

    # ---- geometric cross matrix (jump channel)
    if p["x_jump_w"]:
        Xsum = Xsum + p["x_jump_w"] * _jump_channel(b, s, p, x_x)
    Xbar = _wmean(s, Xsum, float(p["w_x"]))

    # ================================================================== Rbar
    Rbar = _rbar(b, s, p, xrow)

    # 同手和弦密度：单手（拇指列除外）同行达到两个音符的行。
    # 无条件计算（开销小）：d_sh 是合并层系数，按它门控曲线会把曲线
    # 冻进曲线缓存里。
    d_sh = float(p.get("d_sh", 0.0))
    mask = s.mask.astype(np.int32)
    left = _POPCOUNT7[mask & 0b0000111]
    right = _POPCOUNT7[mask & 0b1110000]
    if tau_c > 0.0:
        # 同手和弦的分级计数：三角窗统计同手音符数，再做连续化。
        # （同手折扣项是逐行数值，由滑动平均做时间积分，所以不需要攻击折扣，
        # 子区间求和本来就是同一跨度。）
        le = _tri_window_count(s, left.astype(np.float64), tau_c)
        ri = _tri_window_count(s, right.astype(np.float64), tau_c)
        sh = (le * np.clip(le - 1.0, 0.0, 1.0)
              + ri * np.clip(ri - 1.0, 0.0, 1.0))
    else:
        sh = (np.where(left >= 2, left, 0.0)
              + np.where(right >= 2, right, 0.0))
    SHd = _wmean(s, sh.astype(np.float64), max(float(p.get("shd_w", 1000.0)), 50.0))

    # ================================================================== Cbar
    Cbar = _cbar(b, s, p, xrow, stats, Jbar, Ja)
    CbarV2 = _cbar_v2(b, s, p, stats)
    Cbar = Cbar + CbarV2

    # ============================================================ hybrid/LN
    # 混合长键补充项。释放拥挤度与按住时长并入长键协调曲线（它们属于长键
    # 协调现象，沿用同样的定价）；双押密度率与恢复折扣是独立曲线，
    # 由合并阶段的流分支消费。
    Chord2 = np.zeros(n)
    Recov = np.zeros(n)
    if p.get("use_hb", 0.0):
        _hb = _HB_MOD
        is4k_row = s.is_4k[s.chart]
        churn_c = float(p.get("cbv_churn", 0.0))
        churn_4k = float(p.get("cbv_churn_4k", 0.0))
        holdage_c = float(p.get("cbv_holdage", 0.0))
        holdage_4k = float(p.get("cbv_holdage_4k", 0.0))
        if churn_c or churn_4k:
            # 0 = inherit the 7K coefficient (per-frame LN physics)
            cw = np.where(is4k_row, churn_4k if churn_4k != 0.0 else churn_c,
                          churn_c)
            Cbar = Cbar + cw * _hb.compute_churn(b, s, p)
        if holdage_c or holdage_4k:
            hw = np.where(is4k_row,
                          holdage_4k if holdage_4k != 0.0 else holdage_c,
                          holdage_c)
            Cbar = Cbar + hw * _hb.compute_holdage(b, s, p)
        if p.get("c_chord2", 0.0):
            Chord2 = _hb.compute_chord2(b, s, p)
        if p.get("d_rec", 0.0):
            Recov = _hb.compute_recov(b, s, p)

    # ================================================================== eye
    # E4 beat-alignment curve (眼难度).  Param-independent shape; only the
    # combine coefficients are tuned.  Gated: use_eye == 0 -> exact zero.
    EyeR = np.zeros(n)
    if p.get("use_eye", 0.0):
        compute_eye_curve = _eye_compute_eye_curve
        EyeR = compute_eye_curve(b, s, p)

    return dict(Jbar=Jbar, Jm=Jm, Jc=Jc, Ja=Ja, Pbar=Pbar, Xbar=Xbar,
                Abar=Abar, Rbar=Rbar, Cbar=Cbar, CbarV2=CbarV2,
                anchor=anchor_mult, EyeR=EyeR, Chord2=Chord2, Recov=Recov,
                SHd=SHd)


def _cbar_v2(b, s: Struct, p: dict, stats) -> np.ndarray:
    """Cbar v2 channel (core/components/lncoord_v2.py).

    Imported lazily: that module imports helpers from this one, so a
    top-level import here would be circular.  Default `use_cbar_v2 = 0`
    makes this return zeros and leave every number untouched.
    """
    if not p.get("use_cbar_v2", 0.0):
        return np.zeros(s.n)
    compute_cbar_v2 = _lncoord_v2_compute_cbar_v2
    return compute_cbar_v2(b, s, p, stats)


# --------------------------------------------------------------- jump / X
def _jump_channel(b, s: Struct, p: dict, x_x: np.ndarray) -> np.ndarray:
    """Rate-weighted geometric transition cost between consecutive notes."""
    n = s.n
    # 7x7 weight matrix indexed by (from_col, to_col) -- small and static.
    d = np.abs(_IDX7[:, None] - _IDX7[None, :])
    same_hand = (HAND[:, None] == HAND[None, :]) & (HAND[:, None] != 1)
    thumb = (HAND[:, None] == 1) | (HAND[None, :] == 1)
    cross = (HAND[:, None] != HAND[None, :]) & ~thumb
    W = np.ones((K, K))
    W += p["x_jd_p"] * np.power(np.maximum(d, 1e-9), p["x_jd_e"])
    W = np.where(same_hand & (d > 0),
                 1.0 + p["x_jh_p"] * np.power(np.maximum(d, 1e-9),
                                              -p["x_jh_e"]), W)
    W = np.where(thumb, 1.0 - p["x_jt"] / np.maximum(d, 1.0), W)
    W = np.where(cross, 1.0 - p["x_jh_p"] * np.minimum(d / K, 1.0), W)
    # direction: inward = the *to* note is closer to the thumb column 3
    din = (np.abs(_IDX7[None, :] - 3) < np.abs(_IDX7[:, None] - 3)).astype(np.float64)
    W = W * (1.0 + p["x_dir_in"] * din + p["x_dir_out"] * (1.0 - din))

    # transitions between consecutive notes (global note order)
    a = b.note_col[:-1]
    bb = b.note_col[1:]
    ok = (b.note_chart[:-1] == b.note_chart[1:])
    w = np.where(ok, W[a, bb] - 1.0, 0.0)
    rows = b.note_row[1:] + np.repeat(s.starts[:-1], b.n_notes)[1:]
    step = np.zeros(n)
    np.add.at(step, rows, w)
    counts = np.zeros(n)
    np.add.at(counts, rows, np.where(ok, 1.0, 0.0))
    mean_w = step / np.maximum(counts, 1.0)
    return mean_w / s.dt


def _row_of_time(s: Struct, t: dict, times: np.ndarray,
                 sel: np.ndarray | None = None) -> np.ndarray:
    """Row index of `times` clipped to the owning chart's row range."""
    if sel is None:
        lo, hi = s._tail_lo, s._tail_hi
        ch = t["chart"]
    else:
        lo, hi = s._tail_lo[sel], s._tail_hi[sel]
        ch = t["chart"][sel]
    z = times + ch.astype(np.int64) * (1 << 30)
    return np.clip(np.searchsorted(s.tg, z, side="left"), lo, hi)


# --------------------------------------------------------------------- Rbar
def _coord_weight(c1, c2, p):
    """释放协调权重：同列 / 同手 / 拇指 / 跨手四种情形。"""
    cw = np.full(c1.shape, p["r_cw_cross"], dtype=np.float64)
    cw = np.where(c1 == c2, p["r_cw_same"], cw)
    same_hand = (HAND[c1] == HAND[c2]) & (HAND[c1] != 1)
    cw = np.where(same_hand, p["r_cw_hand"], cw)
    thumb = (HAND[c1] == 1) | (HAND[c2] == 1)
    return np.where(thumb, p["r_cw_thumb"], cw)


def _rbar(b, s: Struct, p: dict, xrow: np.ndarray) -> np.ndarray:
    n = s.n
    t = s.tail
    if t is None:
        return np.zeros(n)
    xr = np.power(xrow, float(p["od_mult_r"]))
    xs = xr[t["j0"]]
    dur = t["t"] - t["h"]
    I_h = 0.001 * np.abs(dur - 80.0) / xs
    nh = t["nh"]
    I_t = np.where(np.isfinite(nh), 0.001 * np.abs(nh - t["t"] - 80.0) / xs, 10.0)
    I = 2.0 / (2.0 + np.exp(-p["r_I_steep"] * (I_h - p["r_I_off"]))
               + np.exp(-p["r_I_steep"] * (I_t - p["r_I_off"])))

    R = np.zeros(n)
    # 释放通道的连续化开关（默认值均为关闭状态，即精确时间戳行为）：
    #   r_dtr_min   间隔定价的物理下限
    #   r_soft_edges 截断处的余弦坡与亚下限间隔的时间铺底
    #   r_sim_tau   尾/头协调指数选择的混合带
    dtr_min = p.get("r_dtr_min", 1e-4)
    soft = p.get("r_soft_edges", 0.0)
    # interval width floor (ms): a release priced at dtr_eff = max(dtr,
    # dtr_min) is spread over at least dtr_eff of time, so a simultaneous
    # (zero-width) release pair contributes the same mass as a sub-floor
    # one instead of being silently dropped -- and a 1 ms nudge no longer
    # creates mass out of nothing.  0 表示关闭。
    wfloor = dtr_min * 1000.0 if soft else 0.0
    sim_tau = p.get("r_sim_tau", 0.0)
    # ---------- A. per-tail release
    nxt = t["nxt"]
    nxt_is_tail = t["is_tail"]
    nxt_col = t["nxt_col"]
    ok = np.isfinite(nxt) & ((nxt - t["t"]) <= p["r_dt_max"])
    if ok.any():
        dtr = np.where(ok, (nxt - t["t"]) / 1000.0, 1.0)
        rv = p["r_tail"] * np.power(np.maximum(dtr, dtr_min), -0.5) / xs * (1.0 + p["r_I_w"] * I)
        same_col = (nxt_col == t["col"]) & ~nxt_is_tail
        rv = rv * np.where(same_col, p["r_same_col"], 1.0)
        # 协调权重用正幂形式表达：跨手释放容易，但通道整体仍是难度，
        # 不会变成负难度。
        cw = _coord_weight(t["col"], nxt_col, p)
        if sim_tau > 0.0:
            # near-simultaneous next events: blend the tail/head coord
            # exponents linearly over +-sim_tau ms instead of the 1e-9
            # hard is_tail pick.  d_ev < 0 = the next tail comes first.
            with np.errstate(invalid="ignore"):
                d_ev = t["nt"] - nh      # inf - inf when neither exists
            beta = np.clip(0.5 - d_ev / (2.0 * sim_tau), 0.0, 1.0)
            beta = np.where(np.isfinite(d_ev), beta,
                            np.where(nxt_is_tail, 1.0, 0.0))
            ex = p["r_coord_e"] * (beta + (1.0 - beta) * p["r_tt"])
        else:
            ex = p["r_coord_e"] * np.where(nxt_is_tail, 1.0, p["r_tt"])
        rv = rv * np.power(np.maximum(cw, 1e-3), ex)
        # short-LN reduction
        thr = max(p["r_short_thr"], 1.0)
        red = p["r_short_red"] + (1.0 - p["r_short_red"]) * np.minimum(dur, thr) / thr
        rv = rv * np.where(dur > 0, red, 1.0)
        # lock terms
        lock = np.zeros(t["col"].size)
        hm = s.heldmask[t["j0"]]
        col = t["col"]
        # 锁手加成
        if p["r_lock"]:
            for j in range(K):
                bit = ((hm >> j) & 1).astype(bool) & (j != col)
                if bit.any():
                    lock = lock + np.where(bit, _coord_weight(col, np.full(
                        col.size, j), p), 0.0)
        rv = rv * (1.0 + p["r_lock"] * lock)
        # new: straddle (finger-independence) lock matrix
        if p["r_straddle"]:
            ain = np.array([p["r_ain0"], p["r_ain1"], p["r_ain2"], p["r_ain3"]])
            ag = AIN_GROUP
            S0 = np.zeros(t["col"].size)
            for j in range(K):
                bit = ((hm >> j) & 1).astype(bool)
                if bit.any():
                    S0 = S0 + np.where(bit, ain[ag[j]], 0.0)
            S0 = S0 - np.where(((hm >> col) & 1).astype(bool), ain[ag[col]], 0.0)
            rv = rv * (1.0 + p["r_straddle"] * S0)
        if soft:
            # raised-cosine fade over the outer 20% before r_dt_max
            # instead of the hard cutoff (B3)
            rv = rv * _edge_ramp(nxt - t["t"], p["r_dt_max"])
        rv = np.where(ok, rv, 0.0)
        end_t = np.where(ok, np.maximum(t["nxt"], t["t"] + wfloor), t["t"])
        j1 = _row_of_time(s, t, end_t)
        if soft:
            # mass-exact step scaling (B2): the step holds over the grid
            # span [t[j0], t[j1]), which on a sparse grid far exceeds the
            # intended span (row-ceil overextension); scale the value so
            # the time-integral matches the intended span on any grid.
            span_i = np.maximum(end_t - t["t"], 0.0)
            j1c = np.minimum(np.maximum(j1, t["j0"]), n - 1)
            grid_span = s.t[j1c] - s.t[t["j0"]]
            den = np.maximum(grid_span, np.maximum(span_i, 1e-9))
            scale = np.where(span_i > 0.0, span_i / den, 1.0)
            rv = rv * scale
        R = R + _step_scatter(t["j0"], np.maximum(j1, t["j0"]), rv, n)

    # ---------- B. tail-to-tail sequence
    m = t["col"].size - 1
    if m > 0:
        idx = np.arange(m)
        idx = idx[t["chart"][idx] == t["chart"][idx + 1]]
        ch = t["chart"][idx]
        dtr = (t["t"][idx + 1] - t["t"][idx]) / 1000.0
        dtr = np.maximum(dtr, dtr_min)
        cw = _coord_weight(t["col"][idx], t["col"][idx + 1], p)
        sv = (p["r_seq"] * np.power(dtr, -0.5) / xs[idx]
              * (1.0 + p["r_I_w"] * (I[idx] + I[idx + 1]))
              * np.power(np.maximum(cw, 1e-3), p["r_coord_e"]))
        j0 = t["j0"][idx]
        end_B = np.maximum(t["t"][idx + 1], t["t"][idx] + wfloor)
        j1 = np.maximum(_row_of_time(s, t, end_B, idx + 1), j0)
        if soft:
            span_i = np.maximum(end_B - t["t"][idx], 0.0)
            j1c = np.minimum(j1, n - 1)
            grid_span = s.t[j1c] - s.t[j0]
            sv = sv * (span_i / np.maximum(grid_span, span_i))
        R = R + _step_scatter(j0, j1, sv, n)

    # ---------- C. release-order triple (102/120 harder than 012/210)
    if p["r_order_pen"]:
        m = t["col"].size - 2
        if m > 0:
            i = np.arange(m)
            ch = t["chart"][i]
            ok3 = (ch == t["chart"][i + 2])
            span = t["t"][i + 2] - t["t"][i]
            if not soft:
                ok3 &= span <= p["r_order_tau"]
            c1, c2, c3 = t["col"][i], t["col"][i + 1], t["col"][i + 2]
            hh = HAND[c1]
            ok3 &= (hh == HAND[c2]) & (hh == HAND[c3]) & (hh != 1)
            ok3 &= (c1 != c2) & (c2 != c3) & (c1 != c3)
            jump = np.abs(c1.astype(np.int64) - c2) + np.abs(c2.astype(np.int64) - c3)
            lo = np.minimum(np.minimum(c1, c2), c3).astype(np.int64)
            hi_ = np.maximum(np.maximum(c1, c2), c3).astype(np.int64)
            excess = np.maximum(jump - (hi_ - lo), 0.0).astype(np.float64)
            val = np.where(ok3, p["r_order_pen"] * excess / xs[i], 0.0)
            if soft:
                # raised-cosine fade over the outer 20% of the span
                # window instead of the hard cutoff (B5)
                val = val * _edge_ramp(span, p["r_order_tau"])
            j0 = t["j0"][i]
            end_C = np.maximum(t["t"][i + 2], t["t"][i] + wfloor)
            j1 = np.maximum(_row_of_time(s, t, end_C, i + 2), j0)
            if soft:
                span_i = np.maximum(end_C - t["t"][i], 0.0)
                j1c = np.minimum(j1, n - 1)
                grid_span = s.t[j1c] - s.t[j0]
                val = val * (span_i / np.maximum(grid_span, span_i))
            R = R + _step_scatter(j0, j1, val, n)

    return np.maximum(_wmean(s, R, float(p["w_r"])), 0.0)


# --------------------------------------------------------------------- Cbar
def _cbar(b, s: Struct, p: dict, xrow: np.ndarray, stats, Jbar, Ja) -> np.ndarray:
    n = s.n
    out = np.zeros(n)
    held = s.held
    any_ln = b.ln_end.max() > 0 if n else False
    if not any_ln:
        return out

    # ---- shield: same-column note shortly before an LN head
    if p["cb_shield"]:
        tau = max(p["cb_shield_tau"], 1.0)
        sh = np.zeros(n)
        for k in range(K):
            c = s.col_chain[k]
            pos = c["pos"]
            if pos.size < 3:
                continue
            li = np.flatnonzero(s.is_ln[pos])
            li = li[(li >= 1) & (li < pos.size)]
            if li.size == 0:
                continue
            ip = li[:, None] - np.arange(1, SHIELD_LOOKBACK + 1)[None, :]
            okm = ip >= 0
            idx = np.clip(ip, 0, pos.size - 1)
            dtp = c["tc"][li][:, None] - c["tc"][idx]
            okm &= (dtp > 0) & (dtp < 500.0)
            okm &= (c["cid"][idx] == c["cid"][li][:, None])
            contrib = np.where(okm, np.exp(-np.maximum(dtp, 0.0) / tau), 0.0)
            lock = np.zeros(li.size)
            hm = s.heldmask[pos[li]]
            col = k
            if p["cb_shield_lock"]:
                ain = np.array([p["r_ain0"], p["r_ain1"], p["r_ain2"], p["r_ain3"]])
                ag = AIN_GROUP
                S0 = np.zeros(li.size)
                for j in range(K):
                    bit = ((hm >> j) & 1).astype(bool) & (j != col)
                    if bit.any():
                        S0 = S0 + np.where(bit, ain[ag[j]], 0.0)
                lock = p["cb_shield_lock"] * S0
            val = contrib.sum(axis=1) * (1.0 + lock)
            rows = pos[li]
            j1 = np.minimum(rows + 1, n)
            sh = sh + _step_scatter(rows, j1, val, n)
        out = out + p["cb_shield"] * _wmean(s, sh, float(p["w_cb"]))

    # ---- locked jack / anchor
    hp = np.power(np.maximum(held, 0.0), float(p["cb_lock_pow"]))
    if p["cb_lockjack"]:
        out = out + p["cb_lockjack"] * Jbar * hp
    if p["cb_lockanchor"]:
        out = out + p["cb_lockanchor"] * Ja * hp
    # ---- hold mass / hold-tap density
    if p["cb_par"]:
        out = out + p["cb_par"] * hp
    if p["cb_holdtap"]:
        out = out + p["cb_holdtap"] * (s.size / s.dt) * hp
    return np.maximum(out, 0.0)


# ------------------------------------------------------------------ combine
def combine(s: Struct, cur: dict, p: dict) -> np.ndarray:
    A = np.maximum(cur["Abar"], 1e-6)
    Ks = s.Ks
    C = np.maximum(s.C, 0.0)

    def branch(v, mult):
        return np.power(A, mult / Ks) * np.minimum(
            np.maximum(v, 0.0), p["cap_a"] + p["cap_b"] * np.maximum(v, 0.0))

    # 流分支的按键数权重：7 键行用 a_p/a_r/a_cb，4 键嵌入行用 *_4k
    # 覆盖（为 0 表示沿用共用值）。
    is4k_row = s.is_4k[s.chart]
    a_p_4k = float(p.get("a_p_4k", 0.0))
    a_r_4k = float(p.get("a_r_4k", 0.0))
    a_cb_4k = float(p.get("a_cb_4k", 0.0))
    a_p_row = np.where(is4k_row, a_p_4k, p["a_p"]) if a_p_4k != 0.0 else p["a_p"]
    a_r_row = np.where(is4k_row, a_r_4k, p["a_r"]) if a_r_4k != 0.0 else p["a_r"]
    a_cb_row = (np.where(is4k_row, a_cb_4k, p["a_cb"]) if a_cb_4k != 0.0
                else p["a_cb"])

    Jm_b = branch(cur["Jm"], p["aj"])
    Jc_b = branch(cur["Jc"], p["aj"])
    Ja_b = branch(cur["Ja"], p["aj"])
    # 视觉规整度：流分支内的密度门控规整度交互。
    # 按键数分别取值（7 键与 4 键各一个系数）。
    d_eye = float(p.get("d_eye", 0.0))
    d_eye_4k = float(p.get("d_eye_4k", 0.0))
    eye_term = 0.0
    if p.get("use_eye", 0.0) and (d_eye != 0.0 or d_eye_4k != 0.0):
        d_row = np.where(is4k_row, d_eye_4k, d_eye) if d_eye_4k != 0.0 else d_eye
        eye_term = d_row * np.maximum(cur["Pbar"], 0.0) * cur["EyeR"]
    stream = np.power(A, p["ap"]) * (
        a_p_row * np.maximum(cur["Pbar"], 0.0)
        + a_r_row * np.maximum(cur["Rbar"], 0.0) / (C + p["a_c"])
        + a_cb_row * np.maximum(cur["Cbar"], 0.0) / (C + p["a_cbc"])
        + float(p.get("d_sh", 0.0)) * cur.get("SHd", 0.0)
        + eye_term)

    # hybrid/LN: chord2 density into stream; recovery burst discount.
    if p.get("use_hb", 0.0):
        c_chord2 = float(p.get("c_chord2", 0.0))
        if c_chord2 != 0.0:
            stream = stream + c_chord2 * cur["Chord2"]
        d_rec = float(p.get("d_rec", 0.0))
        if d_rec != 0.0:
            rec = np.maximum(cur["Recov"], 0.0)
            half = max(float(p.get("rec_half", 0.5)), 1e-3)
            stream = np.maximum(stream - d_rec * rec / (rec + half), 0.0)

    sp = float(p["s_p"])
    if abs(sp - 1.0) < 1e-6:
        S = (p["c_jm"] * Jm_b + p["c_jc"] * Jc_b + p["c_ja"] * Ja_b
             + p["c_s"] * stream)
    else:
        inner = (p["c_jm"] * np.power(Jm_b, sp) + p["c_jc"] * np.power(Jc_b, sp)
                 + p["c_ja"] * np.power(Ja_b, sp)
                 + p["c_s"] * np.power(np.maximum(stream, 0.0), sp))
        S = np.power(np.maximum(inner, 0.0), 1.0 / sp)
    S = np.maximum(S, 0.0)

    Xt = np.maximum(cur["Xbar"], 0.0) + p["t_jack_mix"] * Jm_b
    T = np.power(A, p["at"] / Ks) * Xt / (Xt + S + p["t_s_off"])
    D = (p["d_b1"] * np.power(S, p["d_ds"]) * np.power(np.maximum(T, 1e-9), p["d_dt"])
         + p["d_b2"] * S)

    # ---- read (眼难度) modifier
    if p.get("use_read", 0.0):
        compute_read, compute_read_gate = _read_compute_read, _read_compute_read_gate
        read_curve = compute_read(s.batch, s, p)
        gate = compute_read_gate(s.batch, s, p)
        dr = p.get("d_read", 0.0)
        dr_4k = p.get("d_read_4k", 0.0)
        # per-frame: dr for 7K, dr_4k for 4K
        dr_eff = dr + (dr_4k - dr) * gate
        if np.any(dr_eff != 0.0):
            modifier = np.clip(1.0 + dr_eff * read_curve,
                               p.get("read_clip_lo", 0.5),
                               p.get("read_clip_hi", 1.5))
            D = D * modifier

    return np.maximum(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


# ---------------------------------------------------------------- aggregate
def _percentiles(s: Struct, D: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-chart weighted percentile aggregate (Sunny style)."""
    n = s.n
    order = np.lexsort((D, s.chart))
    Ds = D[order]
    ws = w[order]
    cum = seg_cumsum(ws, s.starts)
    tot = np.repeat(cum[s.starts[1:] - 1], s.counts)
    f = cum / np.maximum(tot, EPS)
    out = np.zeros(s.n_charts)
    for base, targs in ((0.93, (0.945, 0.935, 0.925, 0.915)),
                        (0.83, (0.845, 0.835, 0.825, 0.815))):
        acc = np.zeros(s.n_charts)
        for t in targs:
            # first index in each chart where f >= t
            hit = (f >= t)
            # position of first True per chart via segmented trick
            idx = np.zeros(s.n_charts, dtype=np.int64)
            flat = np.flatnonzero(hit)
            cids = s.chart[flat]
            first_mask = np.ones(flat.size, dtype=bool)
            first_mask[1:] = cids[1:] != cids[:-1]
            idx[cids[first_mask]] = flat[first_mask] - s.starts[:-1][cids[first_mask]]
            # guard: charts with no hit -> last row
            acc += Ds[s.starts[:-1] + np.clip(idx, 0, s.counts - 1)]
        out += (base == 0.93 and 0.25 or 0.20) * (base == 0.93 and 0.88 or 0.94) * acc / 4.0
    # weighted power mean
    wsum = np.zeros(s.n_charts)
    np.add.at(wsum, s.chart, np.power(Ds, 5.0) * ws)
    wden = np.zeros(s.n_charts)
    np.add.at(wden, s.chart, ws)
    out += 0.55 * np.power(wsum / np.maximum(wden, EPS), 0.2)
    return out


def _peak_envelope(s: Struct, D: np.ndarray, ms: float) -> np.ndarray:
    """Rolling-max envelope of D over a +-ms/2 time window, per chart.

    The power-mean aggregate dilutes narrow difficulty peaks into the chart's
    bulk; high peaks still matter to players.  Widening
    peaks into ~ms-wide plateaus restores their time share in any weighted
    mean.  The window is approximated in rows via each chart's median row
    spacing, which keeps this a C-speed per-chart maximum_filter1d.
    """
    from scipy.ndimage import maximum_filter1d
    out = np.empty_like(D)
    for c in range(s.n_charts):
        a, z = int(s.starts[c]), int(s.starts[c + 1])
        v = D[a:z]
        m = v.size
        if m < 3:
            out[a:z] = v
            continue
        dur = max(float(s.duration[c]), 1.0)
        med_dt = max(dur / m, 1.0)
        half = max(1, int(round(0.5 * ms / med_dt)))
        out[a:z] = maximum_filter1d(v, size=2 * half + 1, mode="nearest")
    return out


def aggregate(s: Struct, D: np.ndarray, p: dict,
              hold_w: bool = False) -> np.ndarray:
    """Per-chart SR from the difficulty curve D(t).

    agg_mode: 0 = Sunny percentiles, 1 = soft-max (exponential mean),
              2 = sigmoid accuracy model, 3 = weighted power mean.
    """
    n = s.n
    # 峰包络：聚合会稀释窄峰，env_w = 0 表示关闭。
    env_w = float(p.get("env_w", 0.0))
    if env_w != 0.0:
        E = _peak_envelope(s, D, max(float(p.get("env_ms", 800.0)), 50.0))
        D = (1.0 - env_w) * D + env_w * E
    dnext = np.zeros(n)
    dnext[:-1] = s.dt_diff
    dprev = np.zeros(n)
    dprev[1:] = dnext[:-1]
    dprev[s.starts[:-1]] = 0.0
    gp = np.maximum(0.5 * (dnext + dprev), 0.0)
    w = np.power(gp, float(p["agg_gap_w"])) * \
        np.power(np.maximum(s.C, 0.0), float(p["agg_wc"]))
    if hold_w and p.get("hw_a", 0.0):
        h = s.held
        w = w * (1.0 + p["hw_a"] * h / (h + max(p.get("hw_half", 1.0), 1e-6)))
    w = np.maximum(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    D = np.maximum(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    mode = int(p["agg_mode"])
    if mode == 0:
        return _percentiles(s, D, w, p)
    if mode == 1:
        lam = max(float(p["agg_k"]), 1e-3)
        m = D.max() if n else 0.0
        sh = np.zeros(s.n_charts)
        np.add.at(sh, s.chart, w * np.exp(np.clip(lam * (D - m), -50, 50)))
        sw = np.zeros(s.n_charts)
        np.add.at(sw, s.chart, w)
        return m + np.log(np.maximum(sh, EPS) / np.maximum(sw, EPS)) / lam
    if mode == 3:
        pw = max(float(p["agg_k"]), 0.05)
        acc = np.zeros(s.n_charts)
        np.add.at(acc, s.chart, w * np.power(D, pw))
        sw = np.zeros(s.n_charts)
        np.add.at(sw, s.chart, w)
        return np.power(acc / np.maximum(sw, EPS), 1.0 / pw)
    return _sigmoid_agg(s, D, w, p)


def _percentiles(s: Struct, D: np.ndarray, w: np.ndarray, p: dict) -> np.ndarray:
    order = np.lexsort((D, s.chart))
    Ds = D[order]
    ws = w[order]
    cum = seg_cumsum(ws, s.starts)
    tot = np.maximum(np.repeat(cum[s.starts[1:] - 1], s.counts), EPS)
    f = cum / tot
    out = np.zeros(s.n_charts)
    for lo, hi, coef, scale in ((0.915, 0.945, p["p93_c"], p["w93"]),
                                (0.815, 0.845, p["p83_c"], p["w83"])):
        acc = np.zeros(s.n_charts)
        for t in np.linspace(lo, hi, 4):
            acc += _value_at_frac(s, Ds, f, float(t))
        out += scale * coef * acc / 4.0
    wsum = np.zeros(s.n_charts)
    np.add.at(wsum, s.chart, np.power(Ds, p["mean_pow"]) * ws)
    wden = np.zeros(s.n_charts)
    np.add.at(wden, s.chart, ws)
    out += p["wmean"] * np.power(wsum / np.maximum(wden, EPS),
                                 1.0 / max(p["mean_pow"], 1e-6))
    return out


def _value_at_frac(s: Struct, Ds: np.ndarray, f: np.ndarray,
                   t: float) -> np.ndarray:
    """First sorted-D value whose cumulative weight fraction reaches t."""
    hit = f >= t
    flat = np.flatnonzero(hit)
    out = np.zeros(s.n_charts)
    if flat.size == 0:
        return out
    cids = s.chart[flat]
    first = np.ones(flat.size, dtype=bool)
    first[1:] = cids[1:] != cids[:-1]
    sel = flat[first]
    out[cids[first]] = Ds[sel]
    return out


def _sigmoid_agg(s: Struct, D: np.ndarray, w: np.ndarray, p: dict) -> np.ndarray:
    nseg = int(max(p["agg_nseg"], 2))
    order = np.lexsort((D, s.chart))
    Ds = D[order]
    ws = w[order]
    cum = seg_cumsum(ws, s.starts)
    tot = np.maximum(np.repeat(cum[s.starts[1:] - 1], s.counts), EPS)
    f = np.minimum(cum / tot, 1.0 - 1e-12)
    binid = np.minimum((f * nseg).astype(np.int64), nseg - 1)
    key = s.chart[order] * nseg + binid
    size = s.n_charts * nseg
    sd = np.bincount(key, weights=Ds * ws, minlength=size)
    sw = np.bincount(key, weights=ws, minlength=size)
    sd = sd.reshape(s.n_charts, nseg)
    sw = sw.reshape(s.n_charts, nseg)
    dseg = sd / np.maximum(sw, EPS)

    k = max(float(p["agg_k"]), 1e-3)
    C0 = float(p["agg_C"])
    gamma = float(min(max(p["agg_gamma"], 1e-3), 0.999))
    W = sw.sum(axis=1)
    target = W * gamma
    lo = dseg.min(axis=1) - 5.0
    hi = dseg.max(axis=1) + 5.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        e = np.clip(k * (dseg - mid[:, None]), -50, 50)
        val = (sw / (C0 + np.exp(e))).sum(axis=1)
        greater = val > target
        lo = np.where(greater, mid, lo)
        hi = np.where(greater, hi, mid)
    return 0.5 * (lo + hi)


def postprocess(s: Struct, sr: np.ndarray, p: dict) -> np.ndarray:
    n_eff = s.n_notes.astype(np.float64)
    sr = sr * n_eff / (n_eff + max(p["pp_n0"], 1e-6))
    thr = p["pp_thr"]
    div = max(p["pp_div"], 1e-3)
    sr = np.where(sr > thr, thr + (sr - thr) / div, sr)
    return np.maximum(sr * p["pp_scale"] * p["calib_a"] + p["calib_b"], 0.0)




# ===================== core/components/eye.py =====================


import numpy as np



def _eye_compute_eye_curve(b, s, p: dict) -> np.ndarray:
    """Row-grid alignment curve R(t) in (0, 1]; 1 = locally on-grid."""
    w_ref = max(float(p.get("eye_w_ref", 4000.0)), 100.0)
    w_s = max(float(p.get("eye_w_s", 500.0)), 50.0)
    tau = max(float(p.get("eye_tau", 0.2)), 1e-3)

    dt_ms = np.maximum(s.dt_ms, 0.0)
    # rows with no successor (chart-final) have dt_ms == 0: exclude them by
    # substituting the local mean below (dist = 0 -> neutral).
    valid = dt_ms > 0.0
    logdt = np.log(np.maximum(dt_ms, 1e-6))
    mu = _wmean(s, np.where(valid, logdt, 0.0), w_ref)
    cnt = _wmean(s, valid.astype(np.float64), w_ref)
    # _wmean is a plain windowed mean; approximate the valid-only mean:
    mu = np.where(cnt > 1e-6, mu / np.maximum(cnt, 1e-9), 0.0)
    r = np.mod(logdt - mu, np.log(2.0))
    dist = np.minimum(r, np.log(2.0) - r)
    dist = np.where(valid, dist, 0.0)
    return np.exp(-_wmean(s, dist, w_s) / tau)


# ===================== core/components/hb.py =====================


import numpy as np



def _hb_compute_churn(b, s, p: dict) -> np.ndarray:
    """Concave LN-release churn curve (releases/s, saturated)."""
    n = s.n
    if s.tail is None:
        return np.zeros(n)
    imp = np.zeros(n)
    np.add.at(imp, s.tail["j0"], 1.0)
    r = _wmean(s, imp, max(float(p.get("churn_w", 1000.0)), 50.0))
    sat = max(float(p.get("churn_sat", 15.0)), 1.0)
    return r / (1.0 + r / sat)


def _hb_compute_holdage(b, s, p: dict) -> np.ndarray:
    """LN occupation duration curve (seconds past threshold, per column sum).

    Vectorised: within one column LNs never overlap, so the head time of the
    currently-held LN is a step function on the row grid (step_scatter of the
    head over its [head, end) span).  age(t) = clamp(min(t-head, cap) - th).
    """
    n = s.n
    th = float(p.get("holdage_th", 0.25))
    cap = float(p.get("holdage_cap", 1.0))
    out = np.zeros(n)
    tg = s.tg
    t_loc = s.t.astype(np.float64)
    for k in range(7):
        c = s.col_chain[k]
        pos = c["pos"]
        if pos.size == 0:
            continue
        ln_rows = pos[s.is_ln[pos]]
        if ln_rows.size == 0:
            continue
        h = t_loc[ln_rows]                          # local ms
        e = np.minimum(b.ln_end[ln_rows], h + cap * 1000.0)
        cid = b.chart[ln_rows].astype(np.int64)
        j0 = ln_rows
        z1 = e + cid * (1 << 30)
        j1 = np.clip(np.searchsorted(tg, z1, side="left"),
                     s.jlo[j0], s.jhi[j0] + 1)
        # step functions over each LN's span (disjoint within a column)
        H = _step_scatter(j0, j1, h, n)             # active LN's head (ms)
        M = _step_scatter(j0, j1, np.ones_like(h), n)  # held indicator
        age_s = np.clip(np.minimum(t_loc - H, cap * 1000.0) / 1000.0 - th,
                        0.0, None)
        out += np.where(M > 0, age_s, 0.0)
    return out


def _hb_compute_chord2(b, s, p: dict) -> np.ndarray:
    """Double-note row rate (per second)."""
    w = max(float(p.get("chord2_w", 1000.0)), 50.0)
    return _wmean(s, (s.size == 2).astype(np.float64), w)


def _hb_compute_recov(b, s, p: dict) -> np.ndarray:
    """Burst-over-baseline excess (recovery discount driver)."""
    w = max(float(p.get("recov_w", 1000.0)), 50.0)
    wl = max(float(p.get("recov_wl", 20000.0)), 1000.0)
    C_s = _wmean(s, s.C, w)
    C_l = np.maximum(_wmean(s, s.C, wl), 1e-6)
    burst = np.maximum(C_s / C_l - 1.0, 0.0)
    return _wmean(s, burst, w)


# ===================== core/components/lncoord_v2.py =====================


import numpy as np

_DEFAULTS_REF = _PARAM_DEFAULTS

# column-pair groups of the 7x7 cross-column matrices (mirror-symmetric by
# construction):  same hand adjacent | same hand split | one hand split
# (thumb) | two hand split | far cross
_SUFFIXES = ("sh_adj", "sh_split", "1h_split", "2h_split", "cross")

_IDX = np.arange(K, dtype=np.int64)
_STRIDE = 1 << 30          # must match core.batch.CHART_STRIDE

# default matrix shapes, used only when a key is absent from `p`
_DEF_SHAPE = {"sh_adj": 0.6, "sh_split": 1.0, "1h_split": 0.8,
              "2h_split": 0.5, "cross": 0.3}

_lncoord_v2_TERMS = ("shield", "straddle", "lockstack", "lockanchor", "overlap", "holdtap")


def _lncoord_v2_dist_matrix(p, prefix: str) -> np.ndarray:
    """7x7 mirror-symmetric cross-column cost matrix.

    Partition of unordered pairs (a, b), a != b -- disjoint and complete:
        both in {0,1,2} or both in {4,5,6}   -> same hand
        exactly one of them is 3             -> thumb ("one hand split")
        one in {0,1,2}, other in {4,5,6}     -> cross hand ("two hand split")
    within same hand, d == 1 -> sh_adj, d == 2 -> sh_split.  d >= 3 -> cross.

    Diagonal is 0: acting on a column is not locked by that same column.
    """
    d = np.abs(_IDX[:, None] - _IDX[None, :])
    h = HAND[:, None]
    same_hand = (h == HAND[None, :]) & (h != 1)
    thumb = (h == 1) | (HAND[None, :] == 1)
    cross = (h != HAND[None, :]) & ~thumb

    def g(suf):
        v = p.get(prefix + suf)
        return _DEF_SHAPE[suf] if v is None else float(v)

    M = np.full((K, K), g("cross"), dtype=np.float64)
    M = np.where(cross & (d <= 2) & (d > 0), g("2h_split"), M)
    M = np.where(thumb & (d <= 2) & (d > 0), g("1h_split"), M)
    M = np.where(same_hand & (d == 1), g("sh_adj"), M)
    M = np.where(same_hand & (d == 2), g("sh_split"), M)
    M = np.where(d == 0, 0.0, M)
    return M


class _lncoord_v2_CbarV2:
    """Interval(row)-based LN coordination channel.

    Parameters
    ----------
    b, s : Batch / Struct from `core.model`
    p    : flat parameter dict (missing keys fall back to `core.params`)
    stats: optional per-column jack stats from `build_jack_stats` (rebuilt
           if absent)
    """

    _lncoord_v2_TERMS = _lncoord_v2_TERMS

    def __init__(self, b, s, p, stats=None):
        self.b = b
        self.s = s
        self.p = p
        self.n = s.n
        self._stats = stats
        self._mats: dict = {}

    # ---------------------------------------------------------------- utils
    def _f(self, key: str) -> float:
        v = self.p.get(key)
        if v is None:
            v = DEFAULTS.get(key, 0.0)
        return float(v)

    def _gate(self, name: str) -> bool:
        return self._f("use_cbv_" + name) != 0.0

    def _mat(self, prefix: str) -> np.ndarray:
        M = self._mats.get(prefix)
        if M is None:
            M = _lncoord_v2_dist_matrix(self.p, prefix)
            self._mats[prefix] = M
        return M

    @property
    def stats(self):
        if self._stats is None:
            self._stats = build_jack_stats(self.b, self.s, self.p)
        return self._stats

    def _lockmat(self) -> np.ndarray:
        return self._mat("cbv_ls_")

    def _bits(self, rows: np.ndarray) -> np.ndarray:
        """(m, K) float matrix: 1 where column j is held at `rows`."""
        hm = self.s.heldmask[rows]
        return ((hm[:, None] >> _IDX[None, :]) & 1).astype(np.float64)

    def _lockmass(self, rows: np.ndarray, col: int) -> np.ndarray:
        """sum_j M[col, j] * held_j(row), j != col  --  per entry of `rows`.

        M[col, col] == 0 so the column itself is excluded automatically.
        This is the whole point of v2: the *identity* of the held columns
        matters, not just their number.
        """
        return self._bits(rows) @ self._lockmat()[col]

    # ------------------------------------------------------------- 1. shield
    def _t_shield(self) -> np.ndarray:
        """盾: a tap (or very short LN) struck shortly before an LN head on
        the *same* column, amplified by what else is held.

        v1 summed exp(-dt/tau) over the 16 previous same-column notes, which
        double counts long chains.  v2 uses the **immediate** predecessor --
        that is what "rapidly striking another LN on the same column" means.

        The predecessor's own duration matters: a tap leaves the finger free
        and is the classic 盾 setup; a long LN already occupies the finger,
        so re-striking is a different (and cheaper) action.  Hence
        `cbv_sh_ln_w * exp(-dur / cbv_sh_ln_tau)` for an LN predecessor.
        """
        s, b, n = self.s, self.b, self.n
        tau = max(self._f("cbv_sh_tau"), 1.0)
        dtmax = max(self._f("cbv_sh_dt_max"), 1.0)
        ln_w = self._f("cbv_sh_ln_w")
        ln_tau = max(self._f("cbv_sh_ln_tau"), 1.0)
        amp = self._f("cbv_sh_lock")
        out = np.zeros(n)
        for k in range(K):
            c = s.col_chain[k]
            pos = c["pos"]
            if pos.size < 2:
                continue
            li = np.flatnonzero(s.is_ln[pos])
            li = li[li >= 1]
            if li.size == 0:
                continue
            pv = li - 1
            dtp = c["tc"][li] - c["tc"][pv]
            ok = (c["cid"][pv] == c["cid"][li]) & (dtp > 0.0) & (dtp < dtmax)
            # predecessor type weight.
            # A mania column cannot hold two overlapping notes, so for two
            # consecutive notes in column k we always have
            #     t_prev + dur_prev <= t_now   ->   dur_prev <= dtp
            # Clamping the predecessor's duration to `dtp` therefore costs
            # nothing on clean data and bounds the damage on mixed chords,
            # where `b.ln_end[row]` is the row max and may belong to a
            # *different* column (see LIMITATIONS).  Without the clamp that
            # case reads as an arbitrarily long LN and the term collapses.
            prow = pos[pv]
            is_ln_prev = s.is_ln[prow]
            dur = np.clip(b.ln_end[prow] - s.t[prow], 0.0, dtp)
            wpred = np.where(is_ln_prev,
                             ln_w * np.exp(-np.maximum(dur, 0.0) / ln_tau), 1.0)
            val = np.where(ok, wpred * np.exp(-np.maximum(dtp, 0.0) / tau), 0.0)
            rows = pos[li]
            val = val * (1.0 + amp * self._lockmass(rows, k))
            out = out + _step_scatter(rows, np.minimum(rows + 1, n), val, n)
        return _wmean(s, out, self._f("cbv_w"))

    # ----------------------------------------------------------- 2. straddle
    def _t_straddle(self) -> np.ndarray:
        """衩叠: alternation between columns a and b while a column *between*
        them is held.  The canonical case is 0/2 alternating while 1 (the
        left middle finger) is pinned, or 4/6 while 5 is pinned.

        This is the term v1 structurally cannot express: `held^p` is a count,
        so it cannot see that the pinned finger is the one *in the middle*.

        Construction
        ------------
        For every pair (a, b) with b - a == d, 2 <= d <= cbv_str_maxd:

            act   = sqrt(rate_a * rate_b) * balance ** cbv_str_bal
                    (geometric mean = both sides busy; balance = alternation
                     rather than one side dominating)
            pin   = sum over interior columns m of held_m * hw[AIN_GROUP[m]]
            term  = M[a, b] * act * pin

        `M` is the mirror-symmetric 7x7 pair matrix (`_lncoord_v2_dist_matrix`) whose
        d == 2 entries are exactly the four groups named in the brief:
            (0,2) (4,6)  same_hand_split  <- true 衩叠, middle finger pinned
            (1,3) (3,5)  one_hand_split   <- thumb-bridged
            (2,4)        two_hand_split
        `hw` (cbv_str_h0..h3, indexed by AIN_GROUP) adds a second,
        finger-identity weighting on the pinned column on top of the pair
        group, so "middle finger pinned" can be stressed independently.

        Everything is evaluated per row against `s.usage` (a +-400 ms local
        note count) and `s.heldmask` (per-row), so the term is local in time.
        """
        s, n = self.s, self.n
        M = self._mat("cbv_str_")
        hw = np.array([self._f(f"cbv_str_h{i}") for i in range(4)])
        bal_p = self._f("cbv_str_bal")
        maxd = int(min(max(self._f("cbv_str_maxd"), 2.0), K - 1))
        r = s.usage.astype(np.float64)                     # (K, n)
        hmb = ((self.s.heldmask[:, None] >> _IDX[None, :]) & 1).astype(np.float64)
        out = np.zeros(n)
        # A pair can only be "locked straddle" if some column strictly
        # between a and b can actually be held in this frame.  In the 4K
        # embedding (frame cols {1,2,4,5}) the interior of pairs like (1,3)
        # is column 3 (thumb) or empty -- never held, pin==0, so the term
        # silently died on the whole 4K dataset.  Gate on the interior
        # columns being *ever active*.
        for d in range(2, maxd + 1):
            inter = np.arange(1, d)                        # offsets a+1..b-1
            for a in range(K - d):
                bcol = a + d
                interior = a + inter
                if not s.act[interior].any(axis=1).any():
                    continue
                ra, rb = r[a], r[bcol]
                co = np.sqrt(np.maximum(ra * rb, 0.0))
                bal = 2.0 * np.minimum(ra, rb) / np.maximum(ra + rb, 1e-9)
                act = co * np.power(bal, bal_p)
                pin = (hmb[:, interior] * hw[AIN_GROUP[interior]]).sum(axis=1)
                out = out + M[a, bcol] * act * pin
        return _wmean(s, out, self._f("cbv_w"))

    # -------------------------------------------------- 3/4. locked jack/anchor
    def _col_locked(self, kind: str) -> np.ndarray:
        """Power-mean over columns of (per-column jack curve * its lock mass).

        Unlike v1's `Jbar * held^p`, the lock mass is evaluated **per
        column** before the aggregation, so a jack in column 0 is amplified
        by what is held relative to column 0 specifically.

        kind == "plain"     -> stack speed            (lockstack)
        kind == "anchor"    -> stack speed gated by same-column run length
                               (lockanchor), using the same run-length shape
                               as the Ja channel so the two agree.
        """
        s, p, n = self.s, self.p, self.n
        agg_pow = max(self._f("j_agg_pow"), 0.25)
        ja_floor = self._f("ja_len_floor")
        ja_norm = max(self._f("ja_norm"), 1e-3)
        ja_len_exp = self._f("ja_len_exp")
        ja_coord_exp = self._f("ja_coord_exp")
        # cbv_lock_pow：逐列锁质量的指数。1.0 表示线性；小于 1 时重锁
        # 状态饱和。
        lock_pow = self._f("cbv_lock_pow")
        num = np.zeros(n)
        den = np.zeros(n)
        for k in range(K):
            st = self.stats[k]
            if st is None:
                continue
            if kind == "anchor":
                e = np.maximum(st["runlen"] + 2.0 - ja_floor, 0.0)
                runf = np.power(e / (e + ja_norm), ja_len_exp)
                mod = runf * np.power(1.0 + st["other"] / st["nrows"],
                                      ja_coord_exp)
            else:
                mod = np.ones(st["pos"].size - 1)
            v = np.where(st["valid"], st["kern"] * mod, 0.0)
            cur = _wmean(s, _chain_scatter(st["pos"], v, n, st["valid"]),
                         self._f("cbv_w"))
            # lock mass is a step function over the *gap* [pos[j], pos[j+1])
            lm = _chain_scatter(st["pos"],
                                self._lockmass(st["pos"][:-1], k),
                                n, st["valid"])
            if lock_pow != 1.0:
                lm = np.power(np.maximum(lm, 0.0), lock_pow)
            w = _chain_scatter(st["pos"],
                               np.where(st["valid"], 1.0 / st["gs"], 0.0),
                               n, st["valid"])
            num = num + np.power(np.maximum(cur * lm, 0.0), agg_pow) * w
            den = den + w
        val = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0)
        return np.power(np.maximum(val, 0.0), 1.0 / agg_pow)

    def _t_lockstack(self) -> np.ndarray:
        return self._col_locked("plain")

    def _t_lockanchor(self) -> np.ndarray:
        return self._col_locked("anchor")

    # ------------------------------------------------------------ 5. overlap
    def _t_overlap(self) -> np.ndarray:
        """Genuinely *partial* LN overlap (not containment).

        Two LNs i (earlier) and j (later) on different columns:
            containment  t_i < t_j and e_j <= e_i   -> the classic
                         "hold + something inside"; already priced by
                         holdtap / lockstack, and cheaper.
            partial      t_i < t_j < e_i < e_j      -> the two releases are
                         interleaved with a re-press: this is real 面叠.
        Contribution = min(e_i - t_j, cap)/1000 * M[c_i, c_j], held as a step
        function across the overlap window [t_j, e_i).
        """
        s, b, n = self.s, self.b, self.n
        M = self._mat("cbv_ov_")
        dtmax = max(self._f("cbv_ov_dt_max"), 1.0)
        cap = max(self._f("cbv_ov_cap"), 1.0)

        rows_l, cols_l = [], []
        for k in range(K):
            pos = s.col_chain[k]["pos"]
            if pos.size == 0:
                continue
            m = s.is_ln[pos]
            if not m.any():
                continue
            rows_l.append(pos[m])
            cols_l.append(np.full(int(m.sum()), k, np.int64))
        if not rows_l:
            return np.zeros(n)
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        o = np.argsort(s.tg[rows], kind="stable")
        rows, cols = rows[o], cols[o]
        N = rows.size
        if N < 2:
            return np.zeros(n)

        tg = s.tg[rows]
        while True:
            hi = np.searchsorted(tg, tg + dtmax, side="right")
            cnt = np.maximum(hi - np.arange(N) - 1, 0)
            if cnt.sum() <= 4_000_000 or dtmax <= 50.0:
                break
            dtmax *= 0.5
        total = int(cnt.sum())
        if total == 0:
            return np.zeros(n)
        i_rep = np.repeat(np.arange(N), cnt)
        offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 1
        j_rep = i_rep + offs

        ti, tj = s.t[rows[i_rep]], s.t[rows[j_rep]]
        ei, ej = b.ln_end[rows[i_rep]], b.ln_end[rows[j_rep]]
        ok = (tj > ti) & (ei > tj) & (ej > ei)
        if not ok.any():
            return np.zeros(n)
        ov = np.minimum(ei - tj, cap)
        val = np.where(ok, (ov / 1000.0) * M[cols[i_rep], cols[j_rep]], 0.0)
        j0 = rows[j_rep]
        z = (s.t[j0] + ov) + s.chart[j0].astype(np.int64) * _STRIDE
        j1 = np.clip(np.searchsorted(s.tg, z, side="left"), s.jlo[j0], s.jhi[j0])
        return _wmean(s, _step_scatter(j0, np.maximum(j1, j0), val, n),
                      self._f("cbv_w"))

    # ------------------------------------------------------------ 6. holdtap
    def _t_holdtap(self) -> np.ndarray:
        """Taps struck while holds are active elsewhere.

        v1: `(size/dt) * held^p` -- note rate times a bare hold count.
        v2 keeps the rate but replaces the count by the *mean, over the notes
        actually struck at this row*, of that note's own lock mass.  So a tap
        in column 0 while column 1 is held costs more than while column 6 is
        held, which is the same geometry the Rbar `r_lock` / `r_straddle`
        terms use on the release side.
        """
        s, n = self.s, self.n
        tot = np.zeros(n)
        for k in range(K):
            pos = s.col_chain[k]["pos"]
            if pos.size == 0:
                continue
            np.add.at(tot, pos, self._lockmass(pos, k))
        mean_lm = tot / np.maximum(s.size, 1)
        raw = (s.size / s.dt) * mean_lm
        return _wmean(s, np.maximum(raw, 0.0), self._f("cbv_w"))

    # --------------------------------------------------------------- assembly
    def terms(self) -> dict:
        """Six weighted, gated, smoothed curves; sums exactly to `curve()`."""
        out = {}
        for name in _lncoord_v2_TERMS:
            w = self._f("cbv_" + name)
            if not self._gate(name) or w == 0.0:
                out[name] = np.zeros(self.n)
                continue
            raw = getattr(self, "_t_" + name)()
            out[name] = w * np.maximum(raw, 0.0)
        return out

    def curve(self) -> np.ndarray:
        t = self.terms()
        acc = np.zeros(self.n)
        for v in t.values():
            acc = acc + v
        return np.maximum(np.nan_to_num(acc, nan=0.0, posinf=0.0,
                                        neginf=0.0), 0.0)


def _lncoord_v2_compute_cbar_v2(b, s, p, stats=None) -> np.ndarray:
    """Entry point used by `core.model.compute_curves` when use_cbar_v2 != 0."""
    return _lncoord_v2_CbarV2(b, s, p, stats).curve()


# --------------------------------------------------------------------------
# LIMITATIONS (documented, not silently ignored)
#
# * s.is_ln / b.ln_end are *row*-level aggregates (any note at the row is an
#   LN / max ln_end at the row).  A chord that puts a tap on column k and an
#   LN on column j makes column k look like an LN head at that row.  This
#   affects `shield` and `overlap` only, and only inside mixed chords.  The
#   row grid in core/batch.py has no per-note column array (b.col is
#   referenced by _jump_channel but never defined), so fixing it means
#   changing the Batch layout -- out of scope here.
#
# * `cbv_str_*` / `cbv_ls_*` / `cbv_ov_*` share the same pair-group
#   partition; only their weights differ.  Deliberate: the geometry is one
#   fact, the three channels price it differently.


# ===================== core/components/read.py =====================


import numpy as np



def _read_compute_read_curve(b, s, p: dict) -> np.ndarray:
    """Return the read-regularity curve on the flat row grid.

    Shape: (s.n,), values in [0, 1] where 1 = regular (easy to read).
    """
    n = s.n
    reg = np.zeros(n)
    tol = float(p.get("read_tol", 0.18))
    win = int(p.get("read_n_win", 16))

    # per-chart: compute interval regularity
    for ci in range(b.n_charts):
        a, z = int(s.starts[ci]), int(s.starts[ci + 1])
        if z - a < 3:
            continue
        t = b.t[a:z]  # local times, ms
        d = np.diff(t)  # interval lengths
        if d.size == 0:
            continue
        # local median with window
        nd = d.size
        med = np.zeros(nd)
        for i in range(nd):
            lo = max(0, i - win)
            hi = min(nd, i + win + 1)
            med[i] = np.median(d[lo:hi])
        med = np.maximum(med, 1.0)  # avoid div by zero
        # regularity flag
        reg_i = (np.abs(d - med) <= tol * med).astype(np.float64)
        # assign to rows: reg_i[i] applies to interval [t_i, t_{i+1})
        # which corresponds to row a+i (the row at time t_i)
        reg[a:a + nd] = reg_i

    return reg


def _read_compute_read_gate(b, s, p: dict) -> np.ndarray:
    """门控：4 键谱面取 1.0，7 键谱面取 0.0。"""
    # key_count is per-chart; broadcast to rows
    kc = b.key_count[s.chart]  # (n,)
    return (kc == 4).astype(np.float64)


def _read_compute_read(b, s, p: dict) -> np.ndarray:
    """Compute the smoothed, mean-centered read curve.

    Returns a curve that can be used as a multiplicative modifier:
      D = D * (1 + d_read * read_curve)
    """
    reg = _read_compute_read_curve(b, s, p)
    # smooth with the same window as Pbar
    W = float(p.get("pbar_smooth_window", 1000.0))
    reg_s = _wmean(s, reg, W)
    # mean-center per chart (make it a relative tilt, not a level shift)
    out = reg_s.copy()
    for ci in range(b.n_charts):
        a, z = int(s.starts[ci]), int(s.starts[ci + 1])
        if z - a > 0:
            out[a:z] -= np.mean(reg_s[a:z])
    return out


_HB_MOD = type('m', (), {'compute_churn': _hb_compute_churn, 'compute_holdage': _hb_compute_holdage, 'compute_chord2': _hb_compute_chord2, 'compute_recov': _hb_compute_recov})

_eye_compute_eye_curve = _eye_compute_eye_curve

_lncoord_v2_compute_cbar_v2 = _lncoord_v2_compute_cbar_v2

_read_compute_read, _read_compute_read_gate = _read_compute_read_curve, _read_compute_read_gate

_read_compute_read = _read_compute_read

_BATCH_MOD_REF = type('m', (), {'cached_parse': cached_parse})

DEFAULTS = _PARAM_DEFAULTS



# ------------------------------------------------------------------ driver
def _load_release_params():
    return make(dict(_PARAMS_JSON["params"]))


class _SingleChart:
    """把单个 Notes 包装成 build_batch 所需的 Chart 接口。"""

    def __init__(self, notes, chart_id="chart"):
        self.id = chart_id
        self._notes = notes
        self.path = ""


def rate_notes(notes) -> float:
    """对已解析的 Notes 对象计算星数。"""
    chart = _SingleChart(notes)
    real_parse_file = globals()["parse_file"]

    def _fake_parse_file(path, rate=1.0):
        void = (path, rate)
        return notes

    globals()["parse_file"] = _fake_parse_file
    try:
        b = build_batch([chart], parse_cache=False)
    finally:
        globals()["parse_file"] = real_parse_file
    p = _load_release_params()
    s = build_struct(b, p)
    cur = compute_curves(b, s, p)
    D = combine(s, cur, p)
    sr = aggregate(s, D, p, hold_w=True)
    return float(postprocess(s, sr, p)[0])


def rate_content(text: str, rate: float = 1.0) -> float:
    """解析 .osu 文本内容并计算星数。"""
    notes = parse_content(text, rate=rate)
    if notes is None:
        raise ValueError("not an osu!mania chart")
    return rate_notes(notes)


def rate_file(path: str, rate: float = 1.0) -> float:
    """解析一个 .osu 文件并计算星数。"""
    notes = parse_file(path, rate=rate)
    if notes is None:
        raise ValueError("not an osu!mania chart: %s" % (path,))
    return rate_notes(notes)
