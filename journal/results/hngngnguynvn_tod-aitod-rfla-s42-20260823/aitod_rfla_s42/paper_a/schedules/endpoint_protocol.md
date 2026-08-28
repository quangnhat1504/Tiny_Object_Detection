# SA-ALW Schedule Endpoint and Sensitivity Protocol

Status: `REFERENCE_ENDPOINTS_PREREGISTERED; D1_SCALE_BOUNDS_FITTED` (TinyPerson
P10/P90 bounds frozen in `tinyperson_train_p10_p90.json`, train-only, audit
sha256 `2ae4fb56...093ff9`; AI-TOD-v2 bounds still pending)

## Reference Pilot Schedule

The G3 pilot uses one effect-size-defined reference schedule. It is not selected
from validation or test performance:

- `s_min/s_max`: P10/P90 of valid positive train boxes after the frozen detector
  transform, fitted separately for each dataset.
- `beta_min=8`, `beta_max=10`: large targets recover canonical ALW; the smallest
  targets receive a 20 percent narrower HLA relative-distance margin when the
  quality ratio is 0.60.
- `w_min=1`, `w_max=1.5`: large targets recover ALW; the smallest targets receive
  50 percent more center-term weight.
- `schedule_form=linear`.

The endpoint names follow the current implementation: `beta_max` and `w_max`
apply below `s_min`, while `beta_min` and `w_min` apply above `s_max`.

## Bounded Validation Sensitivity

After a G3 signal, use seed 42 and change exactly one axis at a time around the
reference. The reference run is reused when all hashes match.

| Axis | Candidates | New runs |
|---|---|---:|
| Train percentile bounds | P05/P95, P10/P90, P20/P80 | 2 |
| Small-target beta | 9, 10, 12 with large-target beta 8 | 2 |
| Small-target position weight | 1.25, 1.5, 2.0 with large-target weight 1 | 2 |
| Schedule form | linear, log-linear | 1 |

Total incremental sensitivity budget: seven validation-only runs. No Cartesian
product is allowed.

For beta 9/10/12, the relative-distance margin is respectively 11.1, 20.0, and
33.3 percent narrower than beta 8. Position endpoints represent 25, 50, and 100
percent center-term emphasis. The log-linear alternative is

```text
u_log(s) = clip((log(s_max) - log(s)) /
                (log(s_max) - log(s_min)), 0, 1).
```

It preserves endpoints and places the midpoint at the geometric mean of the
train bounds, matching the multiplicative scale representation used by the
log-shape term.

## Selection and Stop Rule

- Select by validation COCO AP from independently reloaded checkpoints.
- Treat candidates within `0.001` AP of the best as tied; choose the weaker
  adaptation, then the linear form, then P10/P90.
- Report all seven sensitivity rows, including negative results.
- Freeze the chosen schedule before any three-seed core matrix or final-test
  package.
- If no sensitivity candidate exceeds the selected G3 reference, retain the
  reference without another sweep.
- Never transfer `s_min/s_max` between datasets. Effect endpoints may remain
  shared only if this protocol was frozen before D2 validation.
