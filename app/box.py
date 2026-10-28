def passer_rating(att, comp, yards, td, ints) -> float:
    if att <= 0:
        return 0.0

    def clamp(x):
        return max(0.0, min(2.375, x))

    a = clamp((comp / att - 0.3) * 5)
    b = clamp((yards / att - 3) * 0.25)
    c = clamp(td / att * 20)
    d = clamp(2.375 - ints / att * 25)
    return (a + b + c + d) / 6 * 100


def ratio(a, b) -> float:
    return a / b if b else 0.0


def build_box(stats: dict, roster: list) -> dict:
    # stats is {name: {stat: number}}. Only players who took part in that kind of play are listed, best yards first,
    # so a backup who never threw is left out and a receiver who threw a trick pass is in
    def g(n, k):
        return stats.get(n, {}).get(k, 0)

    pos = dict(roster)
    players = list(dict.fromkeys([n for n, _ in roster] + list(stats)))

    def who(*keys):
        return [n for n in players if any(g(n, k) for k in keys)]

    passers, rushers, catchers = who("att", "sacked"), who("carries"), who("targets")

    passing = []
    for n in dict.fromkeys(passers):
        att, comp, yds = g(n, "att"), g(n, "comp"), g(n, "pass_yards")
        passing.append({"name": n, "pos": pos.get(n, "?"), "comp": comp, "att": att, "yards": yds, "pct": ratio(comp, att) * 100,
                        "td": g(n, "pass_td"), "int": g(n, "int"), "sacks": g(n, "sacked"),
                        "rating": passer_rating(att, comp, yds, g(n, "pass_td"), g(n, "int"))})
    rushing = []
    for n in dict.fromkeys(rushers):
        car, yds = g(n, "carries"), g(n, "rush_yards")
        rushing.append({"name": n, "pos": pos.get(n, "?"), "car": car, "yards": yds, "avg": ratio(yds, car),
                        "td": g(n, "rush_td"), "long": g(n, "long_rush")})
    receiving = []
    for n in dict.fromkeys(catchers):
        rec, tgt, yds = g(n, "rec"), g(n, "targets"), g(n, "rec_yards")
        receiving.append({"name": n, "pos": pos.get(n, "?"), "rec": rec, "tgt": tgt, "yards": yds, "avg": ratio(yds, rec),
                          "td": g(n, "rec_td"), "long": g(n, "long_rec")})
    passing.sort(key=lambda r: (-r["yards"], -r["att"]))
    rushing.sort(key=lambda r: (-r["yards"], -r["car"]))
    receiving.sort(key=lambda r: (-r["yards"], -r["tgt"]))
    return {"passing": passing, "rushing": rushing, "receiving": receiving}
