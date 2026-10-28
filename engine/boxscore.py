from dataclasses import dataclass

from engine.text import clock_text


@dataclass(frozen=True)
class Row:
    label: str
    text: tuple
    value: tuple
    better: int = 1
    core: bool = True


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def passer_rating(comp: int, att: int, yards: int, td: int, ints: int) -> float:
    if att == 0:
        return 0.0

    def part(x):
        return max(0.0, min(2.375, x))

    a = part((comp / att - 0.3) * 5)
    b = part((yards / att - 3) * 0.25)
    c = part(td / att * 20)
    d = part(2.375 - ints / att * 25)
    return (a + b + c + d) / 6 * 100


def derive(s) -> dict:
    g = s.__getitem__
    d = {k: g(k) for k in ("plays", "yards", "first_downs", "pass_att", "pass_comp", "pass_yards", "pass_td", "sack",
                           "sack_yards", "rush_att", "rush_yards", "rush_td", "third_att", "third_conv", "fourth_att",
                           "fourth_conv", "rz_trips", "rz_td", "big_plays", "top", "drives", "punt", "penalty",
                           "penalty_yards", "interceptions_thrown", "fumbles_lost", "fg_att", "fg_made", "td")}
    d["pass_net"] = d["pass_yards"] - d["sack_yards"]
    d["turnovers"] = d["interceptions_thrown"] + d["fumbles_lost"]
    d["comp_pct"] = ratio(d["pass_comp"], d["pass_att"])
    d["ypa"] = ratio(d["pass_yards"], d["pass_att"])
    d["ypc"] = ratio(d["rush_yards"], d["rush_att"])
    d["ypp"] = ratio(d["yards"], d["plays"])
    d["third_pct"] = ratio(d["third_conv"], d["third_att"])
    d["rating"] = passer_rating(d["pass_comp"], d["pass_att"], d["pass_yards"], d["pass_td"], d["interceptions_thrown"])
    return d


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def sections(you, opp) -> list:
    a, b = derive(you), derive(opp)

    def both(fn):
        return fn(a), fn(b)

    optional = {"Plays", "20+ yd plays", "Pass TD", "Carries", "Rush TD", "Drives", "Punts", "Yds / attempt", "Yds per play"}

    def row(label, key=None, fmt=str, better=1, fn=None):
        if fn is None:
            fn = lambda d: d[key]
        va, vb = both(fn)
        return Row(label, (fmt(va), fmt(vb)), (va, vb), better, label not in optional)

    def of(label, num, den, better=1):
        return Row(label, tuple(f"{d[num]}/{d[den]} {pct(ratio(d[num], d[den]))}" if d[den] else "0/0 -" for d in (a, b)),
                   (ratio(a[num], a[den]), ratio(b[num], b[den])), better)

    return [
        ("OFFENSE", [
            row("Total yards", "yards"),
            row("Plays", "plays", better=0),
            row("Yds per play", fmt=lambda x: f"{x:.1f}", fn=lambda d: d["ypp"]),
            row("First downs", "first_downs"),
            row("20+ yd plays", "big_plays"),
        ]),
        ("PASSING", [
            Row("Comp / Att", tuple(f"{d['pass_comp']}/{d['pass_att']}" for d in (a, b)),
                (a["pass_comp"], b["pass_comp"]), 1),
            row("Completion %", fmt=pct, fn=lambda d: d["comp_pct"]),
            row("Pass yards", "pass_yards"),
            row("Yds / attempt", fmt=lambda x: f"{x:.1f}", fn=lambda d: d["ypa"]),
            row("Pass TD", "pass_td"),
            row("Passer rating", fmt=lambda x: f"{x:.1f}" if x else "-", fn=lambda d: d["rating"]),
            Row("Sacks taken", tuple(f"{d['sack']}-{d['sack_yards']}" for d in (a, b)), (a["sack"], b["sack"]), -1),
        ]),
        ("RUSHING", [
            row("Rush yards", "rush_yards"),
            row("Carries", "rush_att", better=0),
            row("Yds / carry", fmt=lambda x: f"{x:.1f}", fn=lambda d: d["ypc"]),
            row("Rush TD", "rush_td"),
        ]),
        ("SITUATIONS", [
            of("3rd down", "third_conv", "third_att"),
            of("4th down", "fourth_conv", "fourth_att"),
            of("Red zone TD", "rz_td", "rz_trips"),
        ]),
        ("BALL CONTROL", [
            row("Turnovers", "turnovers", better=-1),
            Row("Penalties", tuple(f"{d['penalty']}-{d['penalty_yards']}" for d in (a, b)),
                (a["penalty_yards"], b["penalty_yards"]), -1),
            Row("Time of poss", tuple(clock_text(d["top"]) for d in (a, b)), (a["top"], b["top"]), 1),
            row("Drives", "drives", better=0),
            row("Punts", "punt", better=0),
        ]),
    ]
