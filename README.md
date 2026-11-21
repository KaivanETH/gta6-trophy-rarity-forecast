# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KaivanETH/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KaivanETH/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, five quantiles of its Exophase earn rate at the case's target. The page only moves when Exophase republishes it, so a target inside a plateau is the last read, widened over the horizon because a page that is not the documented launch snapshot will republish somewhere inside it. On the documented launch-night plateau the model applies the house's own republish arithmetic (638 to 3,868 tracked, 305 to 408 other earned counts) trophy by trophy, and prices the one thing that arithmetic cannot know -- how those 408 whole earned counts split over the 54 trophies -- at the dispersion `tools/backtest.py` calibrates from the live board's own measurement of an earlier, narrower version of this model. Once reads have moved it fits logit rarity against log time, caps and shrinks the move towards the trophy's tier and towards the recent plateau, and sets the logit spread from the fit residual, the distance moved and the horizon. The band is narrow only where the evidence is frozen, honest everywhere else: a crossed tail costs nineteen times what an over-wide band does.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=73rfmaq9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
