import math

import numpy as np


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def one_step_effects(y, base, codes, weights, n_groups):
    # first guess for each player: how far above the base probability he is, with its sampling variance
    p = expit(base)
    w = weights * p * (1 - p)
    info = np.bincount(codes, weights=w, minlength=n_groups)
    info_sq = np.bincount(codes, weights=weights * w, minlength=n_groups)
    miss = np.bincount(codes, weights=weights * (y - p), minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = np.where(info > 0, miss / info, 0.0)
        var = np.where(info > 0, info_sq / info ** 2, np.inf)
    return effect, var


def tau2_from_moments(effects, variances, floor=1e-4):
    # spread of the true effects = spread we see minus the spread that is just noise
    ok = np.isfinite(variances) & (variances < 1.0)
    if ok.sum() < 5:
        return floor
    return max(float(np.var(effects[ok]) - np.mean(variances[ok])), floor)


def fit_logistic_effects(y, base, groups, tau2, weights=None, iters=15):
    # logistic regression with one effect per player, pulled toward 0 by a ridge penalty of 1 / tau2.
    # groups is {name: (codes, n_players)}, for example the QB and the receiver of every pass
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    effects = {name: np.zeros(n) for name, (codes, n) in groups.items()}
    for _ in range(iters):
        for name, (codes, n) in groups.items():
            eta = base + sum(effects[g][groups[g][0]] for g in groups)
            p = expit(eta)
            grad = np.bincount(codes, weights=weights * (y - p), minlength=n) - effects[name] / tau2[name]
            hess = np.bincount(codes, weights=weights * p * (1 - p), minlength=n) + 1.0 / tau2[name]
            effects[name] = effects[name] + grad / hess
    return effects


def poisson_prior(counts, exposure):
    # gamma prior for a rate per unit of exposure: returns (shape, rate), so the mean is shape / rate
    counts, exposure = np.asarray(counts, float), np.asarray(exposure, float)
    ok = exposure > 0
    counts, exposure = counts[ok], exposure[ok]
    mean = counts.sum() / exposure.sum()
    n = len(counts)
    spread = (exposure * (counts / exposure - mean) ** 2).sum() - (n - 1) * mean
    spread /= exposure.sum() - (exposure ** 2).sum() / exposure.sum()
    var = max(spread, mean * 1e-3)
    return mean * mean / var, mean / var


def shrink_rate(counts, exposure, shape, rate):
    return (np.asarray(counts, float) + shape) / (np.asarray(exposure, float) + rate)


def shrink_mean(means, n_eff, within_var, min_n=30, floor=1e-6, max_tau=None):
    # normal prior on a player's true average. n_eff is the (recency weighted) number of plays behind each mean.
    # the spread of true averages is measured on players with enough plays, then everybody is pulled in by it
    means, n_eff = np.asarray(means, float), np.asarray(n_eff, float)
    ok = n_eff > 0
    big = n_eff >= min_n
    if big.sum() < 5:
        big = ok
    grand = np.average(means[big], weights=n_eff[big])
    tau2 = max(float(np.var(means[big]) - np.mean(within_var / n_eff[big])), floor)
    if max_tau is not None:
        tau2 = min(tau2, max_tau ** 2)
    weight = np.where(ok, n_eff / (n_eff + within_var / tau2), 0.0)
    return grand + weight * (means - grand), tau2


def dirichlet_strength(counts, grid=(2, 3, 5, 8, 12, 20, 30, 50, 80, 130, 200, 400)):
    # how many "pseudo throws" the league average mix is worth, picked by the likelihood of the real counts
    counts = np.asarray(counts, float)
    share = counts.sum(axis=0) / counts.sum()
    best, best_ll = grid[0], -math.inf
    for k in grid:
        alpha = k * share
        ll = 0.0
        for row in counts:
            n = row.sum()
            if n <= 0:
                continue
            ll += math.lgamma(k) - math.lgamma(n + k)
            ll += sum(math.lgamma(c + a) - math.lgamma(a) for c, a in zip(row, alpha))
        if ll > best_ll:
            best, best_ll = k, ll
    return best


def shrink_shares(counts, strength):
    counts = np.asarray(counts, float)
    share = counts.sum(axis=0) / counts.sum()
    return (counts + strength * share) / (counts.sum(axis=1, keepdims=True) + strength)


def heldout_logloss(y, base, groups, effects, rows, weights):
    eta = base[rows] + sum(effects[name][codes[rows]] for name, (codes, n) in groups.items())
    p = np.clip(expit(eta), 1e-9, 1 - 1e-9)
    return float(-np.average(y[rows] * np.log(p) + (1 - y[rows]) * np.log(1 - p), weights=weights[rows]))


def choose_tau(y, base, groups, weights, fold, grid, passes=2):
    # one penalty at a time: fit on half the games, score on the other half (both ways), keep the best value,
    # then go around again. grid is {name: [tau values]}
    y, base, weights = np.asarray(y, float), np.asarray(base, float), np.asarray(weights, float)
    names = list(grid)
    best = {n: grid[n][len(grid[n]) // 2] for n in names}

    def loss_of(tau):
        tau2 = {n: t * t for n, t in tau.items()}
        loss = 0.0
        for held in (0, 1):
            train, test = np.where(fold != held)[0], np.where(fold == held)[0]
            sub = {n: (c[train], k) for n, (c, k) in groups.items()}
            eff = fit_logistic_effects(y[train], base[train], sub, tau2, weights[train], iters=6)
            loss += heldout_logloss(y, base, groups, eff, test, weights)
        return loss

    for _ in range(passes):
        for n in names:
            scores = {t: loss_of(dict(best, **{n: t})) for t in grid[n]}
            best[n] = min(scores, key=scores.get)
    return best


def fit_linear_effects(y, groups, tau2, sigma2, weights=None, iters=15):
    # same idea for a number instead of a yes or no: ridge regression with one effect per player or team.
    # sigma2 is the noise variance of one play, so the penalty per group is sigma2 / tau2
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    effects = {name: np.zeros(n) for name, (codes, n) in groups.items()}
    for _ in range(iters):
        for name, (codes, n) in groups.items():
            rest = y - sum(effects[g][groups[g][0]] for g in groups if g != name)
            top = np.bincount(codes, weights=weights * rest, minlength=n)
            bottom = np.bincount(codes, weights=weights, minlength=n) + sigma2 / tau2[name]
            effects[name] = top / bottom
    return effects


def choose_tau_linear(y, groups, weights, fold, grid, sigma2, passes=2):
    y, weights = np.asarray(y, float), np.asarray(weights, float)
    names = list(grid)
    best = {n: grid[n][len(grid[n]) // 2] for n in names}

    def loss_of(tau):
        tau2 = {n: t * t for n, t in tau.items()}
        loss = 0.0
        for held in (0, 1):
            train, test = np.where(fold != held)[0], np.where(fold == held)[0]
            sub = {n: (c[train], k) for n, (c, k) in groups.items()}
            eff = fit_linear_effects(y[train], sub, tau2, sigma2, weights[train], iters=6)
            pred = sum(eff[n][groups[n][0][test]] for n in groups)
            loss += float(np.average((y[test] - pred) ** 2, weights=weights[test]))
        return loss

    for _ in range(passes):
        for n in names:
            scores = {t: loss_of(dict(best, **{n: t})) for t in grid[n]}
            best[n] = min(scores, key=scores.get)
    return best
