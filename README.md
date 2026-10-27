# SPM Rating

[简体中文](README.zh-CN.md)

SPM Rating estimates the difficulty of osu!mania charts. It builds a difficulty curve along the timeline and combines local demands into a chart score. The model develops the [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) approach, with particular attention to repeated presses, chords, and long-note coordination.

## Versions

| Version | Location | Purpose |
|---|---|---|
| **v1.2.0-rc3** | [Candidate package](candidates/v1.2.0-rc3/) | Detailed and compact models, whole-chart and segment scoring, speed changes, and a desktop scorer |
| **v1.1.1** | `core/`, `params/`, `interface/` | Existing stable interface and regression cases |

The candidate supports 4-key, 6-key, and 7-key charts. Its detailed and compact models have weighted development ranking correlations of **0.988334** and **0.988271**. The corresponding v1.1.1 result under this evaluation protocol is **0.981346**. The [evaluation notes](candidates/v1.2.0-rc3/docs/evaluation.md) give sample counts, weights, and model tradeoffs.

## Try the candidate

Use Python 3.13. From the repository root:

```bash
cd candidates/v1.2.0-rc3
python -m pip install -r requirements.txt
python -m spm_rating examples/Control_P_120.osu
python -m spm_rating examples/Control_Q_120.osu --model compact
```

On Windows, `open_scorer.cmd` opens the desktop interface. The [candidate guide](candidates/v1.2.0-rc3/README.md) covers file and directory scoring, CSV export, Python calls, local segments, and output scales. The [model guide](candidates/v1.2.0-rc3/docs/model.md) explains the components and their source files.

## Stable v1.1.1 interface

Install the root `requirements.txt`, then:

```python
import sys
sys.path.insert(0, "interface/python")
import spm_interface as spm
result = spm.analyze("path/to/chart.osu", input_type="path")
print(result["star"])
```

Its contract is in [interface/manifest.json](interface/manifest.json). The existing [model](docs/mechanism.md), [history](docs/history.md), and [maintenance](docs/maintenance.md) documents describe v1.1.1.

## License and attribution

Code uses the [MIT license](LICENSE). [NOTICE](NOTICE) records the architectural attribution and ownership of the stable regression charts. The candidate's small P/Q examples are generated playtest controls. osu! is a trademark of ppy; this is an independent project.
