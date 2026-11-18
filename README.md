# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Crisyehetharia/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Crisyehetharia/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts the Twitch concurrent-viewer count of the Grand Theft Auto VI category for each hour 2–24 of release day (2026-11-19, UTC), as five quantiles (5%, 25%, 50%, 75%, 95%) per hour. The model builds an intraday launch-day shape from past big launches' hourly curves, anchors the level on the first hour's print, and widens the spread with horizon and cross-launch disagreement.

## Run it

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output)
```

`main.py` is Python 3.13, standard library only, no network. It reads every `*.in` file in the input directory and writes one `<id>.out` per case. Runtime is ~0.1 s per case (limit 8 s); two runs are byte-identical.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kefgqpce&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## The model

1. **Intraday shape.** For every past big launch in the case's `comparables`, take its launch-day hourly viewers as log-ratios to its own hour-1 print. Only launches with a **rolling midnight-local unlock** are used: GTA VI unlocks at midnight local, country by country, and simultaneous-global launches have a completely different hour 1 and curve. The shape is the per-hour **median** log-ratio across those launches, with the interquartile range kept as the shape's uncertainty. It captures the real mechanics: a small hour 1 (only NZ/AU/UK unlocked), a dip at 02:00–04:00 UTC, a step when the US East Coast unlocks (05:00 UTC for GTA VI), a slow rise through the European day, and a peak in US prime time (19:00–22:00 UTC).

2. **Level (launch size).** Hour 1 alone does not fix the level: across launches the day's peak is roughly proportional to sqrt(hour-1) — small launches grow ~9× from hour 1, huge ones only ~1.2×. So the model fits `log(peak) = a + b·log(hour-1)` over all comparables with a usable launch day and shifts the forecast so its peak matches that prediction (ramped in with the shape, so early hours stay anchored on hour 1). The predicted peak is floored at 1.5× hour 1 and capped at 50× hour 1.

3. **Spread.** Lognormal quantiles, `q = median · exp(z · s_h)`, `z = (−1.645, −0.674, 0, 0.674, 1.645)`, `s_h = 0.25 + 0.005·(h−1) + 0.2·IQR[h]` — a base uncertainty growing slowly with horizon plus the cross-launch disagreement on that hour.

## Local validation

Practice answers were re-fetched from SullyGnome (the source the kit's `NOTICE.txt` names) and the pinball loss on the log scale was computed exactly as the rules define it.

| case | this model | house baseline |
|---|---|---|
| p00 Marvel's Wolverine | 0.175 | 0.744 |
| p01 Red Dead Redemption 2 | 0.154 | 0.287 |
| p02 God of War Ragnarök | 0.147 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.195 | 0.303 |
| **mean** | **0.168** | **0.387** |

90% coverage 0.78–1.00 across the four cases (mean 0.92); typical median miss ~29% (baseline ~83%). The baseline's 0.387 reproduces the house number in the rules ("about 0.387"), which confirms the practice answers used here are the right ones.

Guardrails: the program never crashes on a weird case (missing hour-1 print, missing or short comparables, tiny counts) — it degrades to a flat persistence forecast; every output is finite, in (0, 1e9], non-decreasing, 23 rows, hours 2–24, ≤ 65,536 bytes.
