"""Scripted match scenario: the ground truth the intelligence engine must rediscover.

Every tactical phase / scripted episode carries `flags`. Disabling a flag removes that
element from the simulation, which is how counterfactual ("what if?") runs are produced.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

L, W = 105.0, 68.0

# (line, lateral). line: 0 GK, 1 defence ... 3 attack. lateral: +1 = team's right touchline.
FORMATIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "4-3-3": {"GK": (0, 0), "RB": (1, .85), "RCB": (1, .3), "LCB": (1, -.3), "LB": (1, -.85),
              "RCM": (2, .45), "CM": (1.9, 0), "LCM": (2, -.45),
              "RW": (3, .8), "ST": (3.1, 0), "LW": (3, -.8)},
    "4-2-3-1": {"GK": (0, 0), "RB": (1, .85), "RCB": (1, .3), "LCB": (1, -.3), "LB": (1, -.85),
                "RDM": (1.6, .3), "LDM": (1.6, -.3),
                "RAM": (2.45, .8), "CAM": (2.45, 0), "LAM": (2.45, -.8), "ST": (3.1, 0)},
    "4-4-2": {"GK": (0, 0), "RB": (1, .85), "RCB": (1, .3), "LCB": (1, -.3), "LB": (1, -.85),
              "RM": (2, .85), "RCM": (2, .28), "LCM": (2, -.28), "LM": (2, -.85),
              "RS": (3, .25), "LS": (3, -.25)},
    "4-4-1": {"GK": (0, 0), "RB": (1, .85), "RCB": (1, .3), "LCB": (1, -.3), "LB": (1, -.85),
              "RM": (2, .85), "RCM": (2, .28), "LCM": (2, -.28), "LM": (2, -.85), "ST": (3, 0)},
}

ROLE_MAP = {
    ("4-3-3", "4-2-3-1"): {"RCM": "RDM", "LCM": "LDM", "CM": "CAM", "RW": "RAM", "LW": "LAM"},
    ("4-2-3-1", "4-3-3"): {"RDM": "RCM", "LDM": "LCM", "CAM": "CM", "RAM": "RW", "LAM": "LW"},
}


@dataclass(frozen=True)
class TeamTactic:
    label: str
    formation: str
    block: float = 30.0         # own-frame x of defensive line with the ball on halfway
    depth: float = 30.0         # defensive line -> forward line distance
    width_in: float = 50.0
    width_out: float = 38.0
    press_n: int = 1            # players hunting the ball out of possession
    press_strength: float = 0.5
    press_reach: float = 70.0   # press only when ball is within this distance of own goal
    counter: bool = False
    hold_forwards: bool = False
    pass_acc: float = 0.86
    tempo: float = 1.0
    directness: float = 0.5
    right_press: float = 0.0    # extra pressing when ball is on the team's right side


TEAMS = {
    "A": {"name": "Ashford City", "short": "ASH", "color": "#ef4444"},
    "B": {"name": "Belmont Rovers", "short": "BEL", "color": "#38bdf8"},
}

# (shirt, name, starting role or None for bench)
ROSTERS = {
    "A": [(1, "Ward", "GK"), (2, "Okoye", "RB"), (4, "Lindqvist", "RCB"), (5, "Moreau", "LCB"),
          (3, "Fenwick", "LB"), (8, "Takeda", "RCM"), (6, "Brennan", "CM"), (10, "Calder", "LCM"),
          (7, "Ashworth", "RW"), (9, "Varga", "ST"), (11, "Pellegrini", "LW"), (17, "Nkemelu", None)],
    "B": [(1, "Harlow", "GK"), (2, "Quinn", "RB"), (5, "Delacroix", "RCB"), (4, "Rask", "LCB"),
          (3, "Osei", "LB"), (7, "Marsh", "RM"), (8, "Ibarra", "RCM"), (6, "Holm", "LCM"),
          (11, "Cardoso", "LM"), (9, "Doyle", "RS"), (10, "Sato", "LS"), (14, "Byrne", None)],
}

A_BAL = TeamTactic("Balanced 4-3-3", "4-3-3", block=30, depth=30, width_in=50, width_out=38,
                   press_n=1, press_strength=.55, press_reach=70)
A_PRESS = TeamTactic("High press", "4-3-3", block=43, depth=26, width_in=50, width_out=36,
                     press_n=3, press_strength=.95, press_reach=105, directness=.6)
A_HIGHLINE = TeamTactic("Possession, high line", "4-3-3", block=39, depth=30, width_in=56,
                        width_out=40, press_n=1, press_strength=.45, press_reach=60, tempo=.9,
                        directness=.4)
A_SHIFT = TeamTactic("4-2-3-1 right-side press", "4-2-3-1", block=40, depth=30, width_in=58,
                     width_out=40, press_n=2, press_strength=.8, press_reach=90, right_press=.9)
A_DOM = TeamTactic("Territorial dominance", "4-2-3-1", block=47, depth=26, width_in=60,
                   width_out=40, press_n=3, press_strength=.9, press_reach=105, directness=.55)
B_BAL = TeamTactic("Balanced 4-4-2", "4-4-2", block=30, depth=30, width_in=50, width_out=38,
                   press_n=1, press_strength=.55, press_reach=70)
B_COUNTER = TeamTactic("Low block + counter", "4-4-2", block=20, depth=24, width_in=48,
                       width_out=34, press_n=1, press_strength=.5, press_reach=42, counter=True,
                       hold_forwards=True, directness=.9)
B_TEN = TeamTactic("10 men low block", "4-4-1", block=19, depth=24, width_in=46, width_out=34,
                   press_n=1, press_strength=.45, press_reach=45, pass_acc=.8, directness=.7)

# (start_second, tactic, required_flags)
PHASES = {
    "A": [(0, A_BAL, ()), (900, A_PRESS, ("A_press",)), (1800, A_HIGHLINE, ()),
          (2700, A_BAL, ()), (3900, A_SHIFT, ("A_sub", "A_shift")),
          (4800, A_DOM, ("A_sub", "A_shift", "B_red"))],
    "B": [(0, B_BAL, ()), (1800, B_COUNTER, ("B_counter",)), (2700, B_BAL, ()),
          (4560, B_TEN, ("B_red",))],
}

SUBSTITUTIONS = [  # (t, team, off_shirt, on_shirt, flags)
    (3660, "A", 7, 17, ("A_sub",)),
]
RED_CARDS = [  # (t, team, shirt, role remap for remaining players, flags)
    (4560, "B", 4, {"LCM": "LCB", "LS": "LCM", "RS": "ST"}, ("B_red",)),
]


@dataclass
class Script:
    id: str
    earliest: int
    flags: Tuple[str, ...]
    cond: Callable  # (sim) -> bool
    steps: List[Tuple[int, str, dict]]


SCRIPTS = [
    Script("press_bypass", 1230, ("A_press",),
           lambda s: s.poss == "B" and s.own_ball("B")[0] < 32,
           [(0, "pass", {"to": ["GK"]}), (2, "pass", {"to": ["LS", "RS"], "through": 6}),
            (3, "carry", {"dist": 8}), (4, "carry", {"dist": 7}), (5, "pass", {"to": ["RS", "LS"]}),
            (6, "carry", {"dist": 5}), (7, "shot", {"goal": False, "outcome": "saved"})]),
    Script("press_goal", 1410, ("A_press",),
           lambda s: s.poss == "B" and s.own_ball("B")[0] < 36,
           [(0, "pass", {"to": ["LCB", "RCB"]}), (2, "pass", {"to": ["GK"]}),
            (4, "pass", {"to": ["RB", "LB"], "intercept": True}), (5, "carry", {"dist": 5}),
            (6, "pass", {"to": ["ST"]}), (7, "shot", {"goal": True})]),
    Script("counter_goal", 2200, ("B_counter",),
           lambda s: s.poss == "A" and s.own_ball("A")[0] > 58,
           [(0, "pass", {"to": ["RCM", "LCM", "CM"], "intercept": True}), (1, "carry", {"dist": 8}),
            (2, "pass", {"to": ["RS", "LS"], "through": 10}), (3, "carry", {"dist": 8}),
            (4, "carry", {"dist": 8}), (5, "carry", {"dist": 7}), (6, "carry", {"dist": 5}),
            (7, "shot", {"goal": True})]),
    Script("shift_goal", 4130, ("A_sub", "A_shift"),
           lambda s: s.poss == "B" and s.own_ball("B")[1] < 28 and s.own_ball("B")[0] < 60,
           [(0, "tackle", {"by": "RAM"}), (1, "carry", {"dist": 7}), (2, "carry", {"dist": 7}),
            (3, "carry", {"dist": 5}), (4, "pass", {"to": ["ST"]}), (5, "carry", {"dist": 3}),
            (6, "shot", {"goal": True})]),
    Script("late_goal", 5100, ("B_red",),
           lambda s: s.poss == "A" and s.own_ball("A")[0] > 62,
           [(0, "pass", {"to": ["RAM", "LAM"]}), (2, "pass", {"to": ["CAM"]}),
            (3, "pass", {"to": ["LAM", "RAM"]}), (5, "pass", {"to": ["ST"]}),
            (6, "carry", {"dist": 4}), (7, "shot", {"goal": True})]),
]

ALL_FLAGS = ("A_press", "B_counter", "A_sub", "A_shift", "B_red")

# Ground-truth description of the script, exported for validation / demo.
GROUND_TRUTH = [
    {"start": 0, "end": 900, "label": "Balanced play"},
    {"start": 900, "end": 1800, "label": "Ashford high press", "team": "A"},
    {"start": 1800, "end": 2700, "label": "Belmont low block + counterattacks", "team": "B"},
    {"start": 2700, "end": 3600, "label": "Balanced play"},
    {"start": 3660, "end": 3660, "label": "Ashford substitution: Ashworth -> Nkemelu", "team": "A"},
    {"start": 3900, "end": 4500, "label": "Ashford switch to 4-2-3-1, right-side press", "team": "A"},
    {"start": 4560, "end": 4560, "label": "Belmont red card (Rask)", "team": "B"},
    {"start": 4800, "end": 5400, "label": "Ashford territorial dominance", "team": "A"},
]


def tactic_at(team: str, t: float, flags) -> TeamTactic:
    current = PHASES[team][0][1]
    for start, tac, req in PHASES[team]:
        if t >= start and all(f in flags for f in req):
            current = tac
    return current
