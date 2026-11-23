"""Public regression cases using packaged v1.3.0 models, generated charts, and analytic time controls."""
from dataclasses import replace
from pathlib import Path
import json, math, subprocess, sys, unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from spm_rating import load_model, rate_file, rate_content, rate_notes
from spm_rating import clock, engine, role, v130_rev2 as core
from spm_rating.api import align


class RatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {name: load_model(name) for name in ['detail', 'compact']}
        cls.chart = ROOT / 'examples/Control_P_120.osu'
        cls.notes = engine.get().parser.parse_file(str(cls.chart))

    def test_completed_control_values(self):
        expected = {
            'detail': {'P_120': 2.652102726544104, 'Q_120': 2.772107696703669, 'P_180': 3.382057101601599, 'Q_180': 3.6428809227987915},
            'compact': {'P_120': 2.7646824970570743, 'Q_120': 2.7947954030665265, 'P_180': 3.4156390087800985, 'Q_180': 3.474714186204438},
        }
        for name, values in expected.items():
            model = self.models[name]
            scores = {}
            for chart, value in values.items():
                with self.subTest(model=name, chart=chart):
                    got = rate_file(ROOT / 'examples' / ('Control_' + chart + '.osu'), model)['raw_score']
                    scores[chart] = got
                    self.assertAlmostEqual(got, value, places=11)
            self.assertLess(scores['P_120'], scores['Q_120'])
            self.assertLess(scores['Q_120'], scores['P_180'])
            self.assertLess(scores['P_180'], scores['Q_180'])

    def test_file_content_and_reference_scale(self):
        for model in self.models.values():
            a = rate_file(self.chart, model, scale='reference')
            b = rate_content(self.chart.read_text(encoding='utf-8-sig'), model)
            self.assertEqual(a['raw_score'], b['raw_score'])
            alignment = model.alignment['7']
            self.assertAlmostEqual(a['score'], align(alignment, a['raw_score']), places=12)

    def test_context_whole_and_mirror(self):
        notes = self.notes
        end = float(max(notes.ln_end) + 1.)
        for model in self.models.values():
            full = rate_notes(notes, model, include_curve=True)
            context = rate_notes(notes, model, start_ms=float(min(notes.time) - 1), end_ms=end, include_curve=True)
            self.assertEqual(full['raw_score'], context['raw_score'])
            np.testing.assert_array_equal(full['field'], context['field'])
            self.assertEqual(rate_notes(engine.mirror(notes), model)['raw_score'], full['raw_score'])

    def test_native_clock_after_speed_changes(self):
        for rate in [.99, 1., 1.01, 1.5]:
            notes = engine.get().parser.parse_file(str(self.chart), rate=rate)
            self.assertIsNone(clock.plan(notes))

    def test_fractional_context_and_small_limit(self):
        notes = self.notes
        shift = 1e-6 * (notes.col - 3.) / 6.
        child = replace(notes, time=notes.time + shift, ln_end=np.where(notes.is_ln, notes.ln_end + shift, notes.ln_end))
        model = self.models['detail']
        base = rate_notes(notes, model)['raw_score']
        full = rate_notes(child, model)['raw_score']
        context = rate_notes(child, model, start_ms=float(min(child.time) - 2), end_ms=float(max(child.ln_end) + 2))['raw_score']
        self.assertLess(abs(full / base - 1), 1e-6)
        self.assertAlmostEqual(full, context, places=11)

    def test_shared_phase_analytic_product(self):
        # E[(3+I(U>1-u))*(7+I(U>1-v))] = 21+3v+7u+min(u,v).
        u, v = .23, .68
        notes = replace(
            self.notes, time=np.array([3 + u, 7 + v]), col=np.array([0, 1]),
            ln_end=np.zeros(2), is_ln=np.zeros(2, dtype=bool), n_notes=2, rate=1., duration=4 + v - u,
        )
        plan = clock.plan(notes)
        value = sum(weight * state.time[0] * state.time[1] for weight, state, _, _ in clock.states(notes, plan))
        self.assertAlmostEqual(value, 21 + 3 * v + 7 * u + min(u, v), places=12)

    def test_added_role_cost_direction(self):
        for name, model in self.models.items():
            with self.subTest(model=name):
                variant = core.variant(model.compact)
                st = core.state_from_notes(self.notes, variant, model.parameters)
                v, _, _ = variant.native(model.parameters)
                v_zero = v.copy()
                v_zero[core.I_ROLE_PRICE] = 0.0
                f1, _ = core.field(st.a, st.b, st.t, st.key, v, model.compact, variant.hill, variant.xsoft, False, 0.0)
                f0, _ = core.field(st.a, st.b, st.t, st.key, v_zero, model.compact, variant.hill, variant.xsoft, False, 0.0)
                ph = np.empty(0)
                sl = ev = np.empty(0, np.int64)
                r1 = core.aggregate(f1, st.gp, st.amount, st.held, ph, sl, ev, 0, st.win_ms, st.env_half, len(st.head_pos), v, *variant.shape(model.parameters))
                r0 = core.aggregate(f0, st.gp, st.amount, st.held, ph, sl, ev, 0, st.win_ms, st.env_half, len(st.head_pos), v_zero, *variant.shape(model.parameters))
                self.assertTrue(bool(np.all(f1 >= f0 - 1e-12)))
                self.assertGreater(r1, r0)

    def test_duplicate_and_conflict(self):
        notes = self.notes
        duplicate = replace(
            notes, time=np.r_[notes.time, notes.time[0]], col=np.r_[notes.col, notes.col[0]],
            ln_end=np.r_[notes.ln_end, notes.ln_end[0]], is_ln=np.r_[notes.is_ln, notes.is_ln[0]], n_notes=notes.n_notes + 1,
        )
        self.assertEqual(role.canonical(duplicate).n_notes, notes.n_notes)
        conflict = replace(duplicate, ln_end=np.r_[notes.ln_end, notes.ln_end[0] + 123.])
        with self.assertRaises(ValueError):
            role.canonical(conflict)

    def test_four_and_six_key_inputs(self):
        for keys in [4, 6]:
            heads = []
            for j in range(24):
                x = int((j % keys + .5) * 512 / keys)
                time = 1000 + 150 * j
                heads.append(f'{x},192,{time},1,0,0:0:0:0:')
            content = 'osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:' + str(keys) + '\nOverallDifficulty:7\n[HitObjects]\n' + '\n'.join(heads)
            result = rate_content(content, self.models['detail'])
            self.assertEqual(result['key_count'], keys)
            self.assertTrue(math.isfinite(result['raw_score']))

    def test_cancellation_and_invalid_input(self):
        with self.assertRaises(InterruptedError):
            rate_notes(self.notes, self.models['detail'], cancel=lambda: True)
        with self.assertRaises(ValueError):
            rate_notes(None, self.models['detail'])
        with self.assertRaises(ValueError):
            rate_file(self.chart, self.models['detail'], rate=0)

    def test_command_line(self):
        result = subprocess.run([sys.executable, '-m', 'spm_rating', str(self.chart), '--model', 'compact'], cwd=ROOT, capture_output=True, text=True, encoding='utf8')
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout)
        self.assertEqual(row['model'], 'compact')
        self.assertEqual(row['units'], 'dan_sr')


if __name__ == '__main__':
    unittest.main()
