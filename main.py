"""gta6-trophy-rarity-forecast-01: forecast every GTA VI PS5 trophy's Exophase
earn rate at a case's target, as five quantiles, scored by pinball loss on the
logit scale. Python 3.13, standard library only.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), one
case file in the input directory per run. Outside the grader, MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR stand in for /input and /output. For each `<id>.in`
found (in name order) the program writes `<id>.out`:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

The model, in brief (three audit revision rounds; the development log records
the claim-by-claim verification and the ablation behind every choice):
earn rates are cumulative and saturate, and the recorded page only changes
when Exophase refreshes it, so the history is first reduced to its distinct
page states. For each trophy, a weighted least-squares line is fitted to the
logit of its earn rate against ln(1 + hours since the first read), over the
last 48 hours of page states (weights: square root of how long each state
stayed on the page, normalized to mean 1 so the slope's standard error is
scale-correct). The fitted slope is shrunk toward its tier's pooled slope —
the median fitted slope of the tier, itself shrunk toward the global median
by the number of valid fits, so a tier with no usable fit (the platinum,
usually) borrows the field's slope instead of silently getting 0. The
forecast is anchored on the last observed rate and the extrapolated move is
damped by 0.55. Trophies with no usable fit take their tier's pooled slope
too, and their spread additionally carries the between-trophy slope spread
(MAD of the valid slopes), since borrowing a slope is itself uncertain.
Earners never decrease, so the forecast rate is floored at
last-earners / projected-players (a damped log-players fit whose slope is
floored at 0 and whose growth is capped, so a denominator surge cannot be
extrapolated as a rate collapse).

A still-invisible silver/gold trophy may simply not have started (story
finale, completion grind) — or may be a genuinely extreme trophy that stays
at ~0 all week. The delayed-wave hedge separates the two situations with a
cohort signal: it engages only in proportion to the SHARE of silver/gold
trophies stalled near zero (a whole tier stalled at once is the wave
signature; a lone stalled trophy among progressing peers is not), fades in
continuously as the trophy's last rate falls toward zero, and scales with
the horizon, so it is inert on hour-scale cases. When it engages it pulls
the centre toward a fixed target rate (2%) — a pull, never a free addition,
so the centre cannot overshoot the target — and widens the upper spread,
while a floor-reach guard keeps the lower spread wide enough that q05
still reaches the 0.01% floor: a trophy that truly stays at zero is never
left many sigma away. The platinum's quantiles are finally capped at the
rarest other trophy's, level by level: a platinum requires every other
trophy, so its rate cannot exceed theirs (an invariant the recorded page
satisfies in every read, and the cap only applies while it does). The remaining
spread combines the fit's residual scale, the slope's standard error over
the extrapolation distance, a floor, and a random-walk term whose horizon
is measured from the last page change; near the floor the spread is
asymmetric (wider up, narrower down) in proportion to how close the centre
sits to the floor. Quantiles are that law's, on the logit scale, mapped
back to percent and clamped to the scorer's [0.01, 99.99]. Every degenerate
case (no history, a single page state, a trophy never seen) falls back to
the last known rate with the same spread law, so the program always writes
a valid forecast.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

STD_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Frozen model configuration (three revision rounds: chosen by ablation on
# the normal scenario family plus six out-of-family stress families —
# delayed story wave, denominator shock, stale page, persistent-zero
# gold/platinum, mixed wave + persistent-zero, and reviewer dilution —
# against an exact replica of the official scorer; see the development
# log's REVISION sections).
FIT_WINDOW_H = 48.0      # fit the trend on the last 48h of page states
SHRINK_K = 12.0          # slope shrinkage strength toward the tier pool
TIER_SHRINK_K = 6.0      # tier pool's own shrinkage toward the global pool
DAMP = 0.55              # damping of the extrapolated logit move
DAMP_P = 0.85            # damping of the log-players projection
SIG_RW = 0.35            # random-walk spread, logit units per sqrt(day)
SIG_FLOOR = 0.25         # spread floor, logit units
SIG_MULT = 0.85          # overall spread scale (calibration)
MAX_JUMP = 4.5           # cap on |forecast logit - last observed logit|
TAU_FLOOR = 0.15         # floor for the between-trophy slope spread (MAD)
BP_GCAP = 5.0            # cap on projected player growth (x last count)
ASYM_UP = 1.3            # upper-spread multiplier at the floor
ASYM_DOWN = 0.85         # lower-spread multiplier at the floor
ASYM_HI_PCT = 2.0        # centre at/above this rate: no asymmetry
HEDGE_TIERS = ("silver", "gold")
HEDGE_RAMP_LO = 0.05     # last rate at/below this: full eligibility
HEDGE_RAMP_HI = 0.5      # last rate at/above this: no eligibility
COHORT_LO = 0.3          # stalled share of the tier at/below this: no hedge
COHORT_HI = 0.7          # stalled share at/above this: full hedge
HEDGE_H0 = 1.0           # horizon scale: days / (days + HEDGE_H0)
HEDGE_TARGET_PCT = 2.0   # target-pull ceiling: the lift pulls the centre
                         # toward this rate and can never push past it
HEDGE_UP = 2.0           # upper-spread multiplier at full engagement
REACH = 0.75             # q05 reaches this share of the way to the floor


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the kit starter's round4 does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


FLOOR_LG = logit_pct(0.01)
HEDGE_TARGET_LG = logit_pct(HEDGE_TARGET_PCT)


def page_states(case):
    """Distinct page states of the history: (hours since first read,
    players_tracked, rates). A read identical to the previous kept one is the
    same state of the page and adds no information."""
    hist = case["history"]
    t0 = parse_ts(hist[0]["at"])
    states = []
    prev = None
    for h in hist:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            states.append(((parse_ts(h["at"]) - t0) / 3600.0, h["players_tracked"], list(h["rates"])))
            prev = key
    return states


def last_known_rates(case):
    """Each trophy's rate in the newest read that lists it (None if never)."""
    out = [None] * len(case["trophies"])
    for h in case["history"]:
        for i, r in enumerate(h["rates"]):
            if r is not None:
                out[i] = r
    return out


def last_known_players(case):
    """players_tracked of the newest read that lists each trophy."""
    out = [None] * len(case["trophies"])
    for h in case["history"]:
        for i, r in enumerate(h["rates"]):
            if r is not None:
                out[i] = h["players_tracked"]
    return out


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x about the weighted means.
    Returns (slope, resid_sd, se_slope, x_mean, y_mean, sxx). The weights
    are first normalized to mean 1: without that, resid_sd (which divides
    by sum(w)) and sxx (which does not) sit on different scales and
    se_slope inherits the weights' arbitrary units."""
    n = len(xs)
    sw0 = sum(ws)
    ws = [w * n / sw0 for w in ws]
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(1.0, n - 2)
    sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / sw * (n / dof)) if n > 2 else 0.0
    se_b = sd / math.sqrt(sxx) if sxx > 1e-12 else float("inf")
    return b, sd, se_b, mx, my, sxx


def forecast_case(case):
    levels = case.get("levels") or STD_LEVELS
    zs = [NormalDist().inv_cdf(t) for t in levels]
    # The 0.95 z-value directly, not -zs[0]: if a case's levels did not
    # start at 0.05, -zs[0] would silently be some other quantile's z and
    # the floor-reach guard would be scaled by it.
    z95 = NormalDist().inv_cdf(0.95)
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    def row(mu, sig_up, sig_down):
        return [round4(clamp_pct(inv_logit_pct(mu + z * (sig_up if z > 0 else sig_down))))
                for z in zs]

    if not case.get("history"):
        return [{"id": t["id"], "q": row(logit_pct(0.5), 1.5, 1.5)} for t in trophies]

    states_all = page_states(case)
    states = states_all
    lr = last_known_rates(case)
    lrp = last_known_players(case)
    t_first = parse_ts(case["history"][0]["at"])
    t_ckpt_h = (parse_ts(case["at"]) - t_first) / 3600.0
    x_tgt = math.log(1.0 + t_ckpt_h + case["horizon_minutes"] / 60.0)

    # Keep only the states inside the fit window (the last state always stays).
    if FIT_WINDOW_H:
        kept = [s for s in states if s[0] >= t_ckpt_h - FIT_WINDOW_H]
        if len(kept) >= 3:
            states = kept

    # Weight of a state: sqrt of the hours it stayed on the page.
    ws = []
    for k, s in enumerate(states):
        nxt = states[k + 1][0] if k + 1 < len(states) else t_ckpt_h
        ws.append(math.sqrt(max(nxt - s[0], 0.01)))

    # The spread's random-walk horizon runs from the last page change.
    stale_days = (t_ckpt_h - states[-1][0]) / 24.0
    horizon_eff = horizon_days + max(0.0, stale_days)
    sig_base = math.sqrt(SIG_FLOOR ** 2 + (SIG_RW * math.sqrt(horizon_eff)) ** 2)

    # Per-trophy trend fits on the logit scale.
    fits = []
    for i in range(len(trophies)):
        xs, ys, w2 = [], [], []
        for k, (th, _p, rates) in enumerate(states):
            r = rates[i]
            if r is None:
                continue
            xs.append(math.log(1.0 + th))
            ys.append(logit_pct(r))
            w2.append(ws[k])
        if len(xs) >= 3 and len(set(ys)) >= 2:
            fits.append((xs, ys, wls(xs, ys, w2)))
        else:
            fits.append(None)

    def pool_stats(idxs):
        """(median fitted slope, number of valid fits) over idxs. A median
        cannot be seized by one near-collinear fit the way an
        inverse-variance mean can (with 2-decimal rates, a trophy whose
        points happen to line up gets a near-zero standard error and would
        otherwise take over the pool)."""
        bs = sorted(fits[i][2][0] for i in idxs if fits[i] is not None)
        return (bs[len(bs) // 2], len(bs)) if bs else (None, 0)

    all_idx = list(range(len(trophies)))
    g_med, _gn = pool_stats(all_idx)
    global_slope = g_med if g_med is not None else 0.0
    tier_groups = {}
    for i, t in enumerate(trophies):
        tier_groups.setdefault(t["tier"], []).append(i)
    # A tier's pool is its median slope shrunk toward the global median by
    # the number of valid fits; a tier with no valid fit at all (the
    # platinum, usually) takes the global slope outright. (The round-1 code
    # stored 0.0 for such a tier, and the .get() fallback to the global
    # slope could never fire because the tier key always exists.)
    tier_slope = {}
    for gname, idxs in tier_groups.items():
        med, n_valid = pool_stats(idxs)
        tier_slope[gname] = global_slope if med is None else (
            (n_valid * med + TIER_SHRINK_K * global_slope) / (n_valid + TIER_SHRINK_K))
    resid_pool = sorted(f[2][1] for f in fits if f is not None)
    resid_med = resid_pool[len(resid_pool) // 2] if resid_pool else 0.3

    # Between-trophy slope spread: how much the borrowed pooled slope itself
    # varies across trophies (robust MAD estimate, floored).
    tau_slope = 0.0
    bs_all = sorted(f[2][0] for f in fits if f is not None)
    if len(bs_all) >= 4:
        med_b = bs_all[len(bs_all) // 2]
        devs = sorted(abs(b - med_b) for b in bs_all)
        tau_slope = max(1.4826 * devs[len(devs) // 2], TAU_FLOOR)

    # Players projection for the earner floor: a damped log-players fit over
    # the full recorded history (the count itself is smooth and high-SNR);
    # its slope is floored at 0 and its growth capped, so the projection
    # stays a sane denominator. (Fitting the recent window instead was
    # measured worse: in this data the recent window IS the surge, so the
    # windowed fit projects even more players and weakens the floor.)
    xs_p = [math.log(1.0 + s[0]) for s in states_all]
    ys_p = [math.log(max(s[1], 1)) for s in states_all]
    ws_p = []
    for k, s in enumerate(states_all):
        nxt = states_all[k + 1][0] if k + 1 < len(states_all) else t_ckpt_h
        ws_p.append(math.sqrt(max(nxt - s[0], 0.01)))
    if len(xs_p) >= 3:
        b_p = max(wls(xs_p, ys_p, ws_p)[0], 0.0)
        ln_tgt = ys_p[-1] + DAMP_P * b_p * (x_tgt - xs_p[-1])
        ln_tgt = min(ln_tgt, ys_p[-1] + math.log(BP_GCAP))
        n_tgt = max(math.exp(ln_tgt), states_all[-1][1])
    else:
        n_tgt = states_all[-1][1]
    x_last_global = math.log(1.0 + states[-1][0])

    # Delayed-wave hedge engagement: the share of silver/gold trophies
    # stalled below the eligibility band is the cohort signal.
    hedge_pool = [i for i, t in enumerate(trophies) if t["tier"] in HEDGE_TIERS]
    if hedge_pool:
        stalled = sum(1 for i in hedge_pool
                      if (lr[i] if lr[i] is not None else 0.0) < HEDGE_RAMP_HI)
        share = stalled / len(hedge_pool)
        cohort_f = min(1.0, max(0.0, (share - COHORT_LO) / (COHORT_HI - COHORT_LO)))
    else:
        cohort_f = 0.0
    h_scale = horizon_days / (horizon_days + HEDGE_H0)
    asym_hi_lg = logit_pct(ASYM_HI_PCT)

    rows = []
    for i, t in enumerate(trophies):
        r_last = lr[i]
        y_last = logit_pct(r_last if r_last is not None else 0.0)
        f = fits[i]
        if f is None:
            # No usable fit: still take the tier's pooled slope rather than
            # freezing at the last rate (a flat line so far is weak evidence
            # about the next four days, especially for late-game trophies),
            # and carry the slope-borrowing uncertainty in the spread.
            g = tier_slope.get(t["tier"], global_slope)
            mu = y_last + DAMP * g * (x_tgt - x_last_global)
            sigma = math.sqrt(sig_base ** 2
                              + (tau_slope * DAMP * (x_tgt - x_last_global)) ** 2)
        else:
            xs, ys, (b, sd, se_b, mx, my, sxx) = f
            g = tier_slope.get(t["tier"], global_slope)
            w_shrink = len(xs) / (len(xs) + SHRINK_K)
            b_s = w_shrink * b + (1.0 - w_shrink) * g
            mu = ys[-1] + DAMP * b_s * (x_tgt - xs[-1])
            sig_fit = sd if sd > 0 else resid_med
            if sd == 0.0 and sxx > 1e-12:
                # Exactly collinear points (2-decimal rounding): the fit's
                # own residual scale is 0, so borrow the pool's scale for
                # the slope's standard error instead of reporting 0.
                se_b = resid_med / math.sqrt(sxx)
            sigma = math.sqrt(sig_fit ** 2 + (se_b * (x_tgt - mx)) ** 2 + sig_base ** 2)
        # Earner floor: earners never decrease, so the rate cannot fall
        # below last-earners / projected-players. This is what stops a
        # launch-weekend surge in tracked players (pure denominator growth)
        # from being extrapolated as a collapsing rate.
        if r_last:
            e_last = r_last / 100.0 * (lrp[i] or states_all[-1][1])
            if e_last > 0:
                floor_mu = logit_pct(e_last / n_tgt * 100.0)
                if mu < floor_mu:
                    mu = floor_mu
        jump = mu - y_last
        if jump > MAX_JUMP:
            mu = y_last + MAX_JUMP
        elif jump < -MAX_JUMP:
            mu = y_last - MAX_JUMP
        mu = min(logit_pct(99.99), max(logit_pct(0.01), mu))
        # Delayed-wave hedge (round-2 gates, round-3 form): eligibility
        # fades in continuously as the last rate falls toward zero, is
        # gated by the stalled-cohort share and scaled by the horizon.
        # The lift is a TARGET-PULL, not a free addition: it closes the
        # fraction w_e * h_scale of the gap between mu and the target
        # logit, so it can never push the centre past the target. (The
        # round-2 free addition of up to +6.0 logit, applied after the
        # jump cap, put a 0.05% trophy's median at 6-13% and its q95 at
        # 55-90%+ while a 0.50% trophy sat at 0.50% — a >10x non-monotone
        # bump across the eligibility band, measured in the round-3
        # verification.) The upper spread widens with engagement; the
        # floor-reach guard below keeps q05 anchored to the floor for the
        # same trophies.
        r_eff = r_last if r_last is not None else 0.0
        w_e = 0.0
        if t["tier"] in HEDGE_TIERS:
            w_e = min(1.0, max(0.0, (HEDGE_RAMP_HI - r_eff)
                               / (HEDGE_RAMP_HI - HEDGE_RAMP_LO))) * cohort_f
        mu = mu + w_e * h_scale * max(0.0, HEDGE_TARGET_LG - mu)
        mu = min(logit_pct(99.99), max(logit_pct(0.01), mu))
        # Asymmetric spread: in proportion to how close the centre sits to
        # the floor, the honest uncertainty points up, not down.
        prox = min(1.0, max(0.0, (asym_hi_lg - mu) / (asym_hi_lg - FLOOR_LG)))
        sig_up = sigma * (1.0 + prox * (ASYM_UP - 1.0))
        sig_down = sigma * (1.0 - prox * (1.0 - ASYM_DOWN))
        sig_up *= (1.0 + w_e * h_scale * (HEDGE_UP - 1.0))
        sig_up *= SIG_MULT
        sig_down *= SIG_MULT
        # Floor-reach guard: for a fully eligible trophy, q05 sits REACH of
        # the way from mu down to the floor logit, so an outcome of exactly
        # 0.00% is never more than (1-REACH) of the centre's height above
        # the floor away — the round-1 hedge left it ~6.9 sigma away.
        need = w_e * REACH * (mu - FLOOR_LG) / z95
        if sig_down < need:
            sig_down = need
        rows.append({"id": t["id"], "q": row(mu, sig_up, sig_down)})

    # Physical consistency cap for the platinum (round 3): on PS5 the
    # platinum requires every other trophy, so its earn rate can never
    # exceed the rarest other trophy's rate — an invariant the practice
    # data satisfies in every read. Enforce it per quantile level, and
    # only when the last observed page satisfies the same invariant, so a
    # data anomaly disables the cap instead of corrupting the forecast.
    # (Capping level k at the min of the other rows' level k preserves
    # monotonicity: both sequences are non-decreasing in k.)
    plat_i = next((i for i, t in enumerate(trophies) if t["tier"] == "platinum"), None)
    if plat_i is not None and lr[plat_i] is not None:
        others_lr = [lr[j] for j in range(len(trophies)) if j != plat_i]
        if all(o is not None for o in others_lr) and lr[plat_i] <= min(others_lr) + 1e-12:
            prow = rows[plat_i]["q"]
            for k in range(len(prow)):
                cap = min(rows[j]["q"][k] for j in range(len(rows)) if j != plat_i)
                if prow[k] > cap:
                    prow[k] = cap
    return rows


def fallback_rows(case):
    """Persistence with a wide spread: used only if the model itself fails,
    so a case is never lost to a crash."""
    levels = case.get("levels") or STD_LEVELS
    zs = [NormalDist().inv_cdf(t) for t in levels]
    lr = last_known_rates(case) if case.get("history") else [None] * len(case["trophies"])
    rows = []
    for i, t in enumerate(case["trophies"]):
        mu = logit_pct(lr[i] if lr[i] is not None else 0.5)
        q = [round4(inv_logit_pct(mu + z * 1.2)) for z in zs]
        rows.append({"id": t["id"], "q": q})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            rows = forecast_case(case)
        except Exception:
            rows = fallback_rows(case)
        out = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
