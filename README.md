[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dingyifeng123456-spec/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dingyifeng123456-spec/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — make.fun entry

Forecasts hours 2–24 of Grand Theft Auto VI's release-day (2026-11-19 UTC)
Twitch category viewers from the hour-1 print, as five quantiles per hour.

Built for the make.fun bounty: https://make.fun/bounty/gta6-twitch-ccu-forecast-01

## Program

`main.py` (Python 3.13, stdlib only, no network, deterministic). Reads
`$MAKEFUN_INPUT_DIR` (else `/input`) for `*.in` cases, writes
`$MAKEFUN_OUTPUT_DIR` (else `/output`)/`<id>.out`.

## Method

Per forecast hour, a local-linear kernel regression of each comparable
launch's release-day log-ratio `ln(viewers_h / viewers_1)` on log hour-1
viewers: launches whose hour-1 print is near this case's weigh most
(Gaussian kernel, bandwidth 1.2 in log space). Rolling-midnight launches
weigh ×2 and console-only launches ×4, because the target is a
console-only rolling-midnight launch — its hour 1 is taken while the US
is still locked, so the day ramps later, unlike PC simultaneous launches
whose hour 1 is already the global peak. Quantiles are normal on the log
scale, `q = p1 * exp(mu_h + z_tau * sigma_h)`, with `sigma_h` the
kernel-weighted residual SD (floored at 0.25, smoothed across hours).
Sparse comparable hours are skipped, never interpolated.

## Validation

Leave-one-out cross-validation over the 37 comparable launches in the
kit (exact scorer pinball loss), same pseudo-case setup for the baseline:

| subset | this model | persistence baseline |
|---|---|---|
| all 37 launches | 0.385 | 0.930 |
| rolling-midnight (n=12) | 0.231 | 0.357 |
| console-only rolling (n=4) | 0.171 | 0.301 |
| hour-1 print > 25k | 0.240 | 0.309 |

Typical miss 37% (rolling) / 28% (console-only rolling) vs baseline
72% / 62%; 90% coverage 0.87 / 0.97. Practice cases p00–p03 all format-
valid, ~0.08 s per case, byte-identical across repeat runs.

## License

MIT (see LICENSE).
