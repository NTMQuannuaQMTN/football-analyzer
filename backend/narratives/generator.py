"""EXPLAIN / RENDER layer: facts -> words.

`NarrativeGenerator` is the seam for Microsoft Foundry / Azure OpenAI. Any implementation
must only rephrase facts in `moment` (evidence values come from analytics); it never invents
numbers. `FoundryNarrativeGenerator` shows where the hosted model plugs in.
"""
import os
import re
from typing import Optional

TEAM_NAMES = {"A": "Ashford City", "B": "Belmont Rovers"}
SHORT = {"A": "Ashford", "B": "Belmont"}


class NarrativeGenerator:
    def generate(self, moment: dict, match_context: dict) -> dict:
        raise NotImplementedError

    def player_summary(self, player: dict, stats: dict, moments: list) -> dict:
        raise NotImplementedError


def _fmt(v, unit=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        v = round(v, 1)
    return f"{'+' if isinstance(v, (int, float)) and v > 0 else ''}{v}{unit}"


class DeterministicNarrativeGenerator(NarrativeGenerator):
    """Template-based narratives. Reliable, reproducible, zero-latency."""

    def generate(self, mo: dict, ctx: dict) -> dict:
        team, opp = SHORT[mo["team"]], SHORT["B" if mo["team"] == "A" else "A"]
        d = mo.get("deltas", {})
        t = mo["type"]
        ev = {e["label"]: e for e in mo["evidence"]}
        if t == "TACTICAL_TURNING_POINT" and mo.get("subtype") == "high_press":
            fan_h = f"{team.upper()} TURN UP THE HEAT"
            fan = (f"{team} are hunting the ball high up the pitch. {opp} keep getting pushed back, "
                   f"and {team} have won it back {mo['after']['final_third_recoveries']} times in the danger zone.")
            if mo["outcome"]["type"] == "GOAL":
                fan += " And a goal followed!"
            why = (f"While the press was on, {opp} were mostly pinned in their own third and {team} won the ball back "
                   f"{mo['after']['final_third_recoveries']} times in the final third; {mo['after']['shots']} shots followed. "
                   f"The sequence suggests the press created these chances.")
            lower = f"{team} high press | Pressing {_fmt(d.get('press_intensity'))} | Line {_fmt(d.get('def_line_oop_m'), 'm')}"
        elif t == "TACTICAL_TURNING_POINT" and mo.get("subtype") == "dominant":
            fan_h = f"{team.upper()} TAKE OVER"
            fan = (f"{team} have pushed up and are camped in {opp}'s half — "
                   f"{mo['after']['territory_pct']}% territory and {mo['after']['shots']} shots in this spell.")
            why = (f"After {team} moved their block up, territory changed {_fmt(d.get('territory_pct'), ' pts')} "
                   f"and shots {_fmt(d.get('shots'))}.")
            lower = f"{team} take control | Territory {_fmt(d.get('territory_pct'), ' pts')} | Shots {_fmt(d.get('shots'))}"
        elif t == "TACTICAL_TURNING_POINT":
            fan_h = f"{team.upper()} SIT DEEP AND STRIKE"
            fan = (f"{team} have handed {opp} the ball, packed their own half and are waiting to break. "
                   f"Every turnover is a chance to run at a high defensive line.")
            why = (f"{team} dropped their block {abs(d.get('def_line_oop_m', 0))}m deeper and ceded the ball; "
                   f"in the following period {opp} pushed forward and {team} broke on the counter. "
                   f"The timing is consistent with a deliberate low-block-and-counter plan.")
            lower = f"{team} low block | Possession {_fmt(d.get('possession_pct'), '%')} | Line {_fmt(d.get('def_line_oop_m'), 'm')}"
        elif t == "FORMATION_SHIFT":
            f0, f1 = mo["detection"]["formation_before"], mo["detection"]["formation_after"]
            fan_h = f"⚡ {team.upper()} JUST TOOK CONTROL"
            fan = (f"The switch to {f1} has changed this game. {team} are pressing harder down the right "
                   f"and have won the ball {mo['after']['final_third_recoveries']} times in {opp}'s final third.")
            if mo["outcome"]["type"] == "GOAL":
                fan += " A goal followed within minutes."
            why = (f"After moving from {f0} to {f1}, the share of right-side pressures changed "
                   f"{_fmt(d.get('right_press_share_pct'), ' pts')}, the line moved {_fmt(d.get('def_line_oop_m'), 'm')} "
                   f"and territory changed {_fmt(d.get('territory_pct'), ' pts')}. The sequence suggests the new shape drove the improvement.")
            lower = f"{team} {f0} → {f1} | Press {_fmt(d.get('press_intensity'))} | Territory {_fmt(d.get('territory_pct'), ' pts')}"
        elif t == "GOAL_BUILD_UP":
            s = mo["after"]["score"]
            fan_h = f"GOAL! {TEAM_NAMES['A']} {s['A']}-{s['B']} {TEAM_NAMES['B']}"
            ctx_txt = {"high press": "It started with the press", "counterattack": "A lightning counter",
                       "sustained possession": "Patient build-up"}[mo["subtype"]]
            fan = f"{ctx_txt} - {mo['after']['passes']} passes, {mo['after']['duration_s']} seconds, and it's in the net."
            why = " → ".join(c["label"] for c in mo["chain"])
            lower = f"GOAL {team} | {mo['subtype'].title()} | xG {mo['after']['xg']}"
        elif t == "PRESSURE_BREAK":
            fan_h = f"{opp.upper()}'S PRESS BEATEN"
            fan = f"One ball over the top and {team} are through - {mo['after']['players_bypassed']} pressers taken out of the game."
            why = "Pressing high leaves space behind; here one long pass bypassed the press and a shot followed."
            lower = f"Press broken | {mo['after']['players_bypassed']} players bypassed | xG {mo['after']['xg']}"
        elif t == "COUNTERATTACK":
            fan_h = f"{team.upper()} ON THE BREAK"
            fan = f"From defence to a shot in {mo['after']['seconds']} seconds!"
            why = f"{opp}'s line was at {mo['before']['opp_def_line_m']}m when the ball turned over; {team} covered {mo['after']['progression_m']}m with {mo['after']['passes']} pass(es)."
            lower = f"Counter | {mo['after']['progression_m']}m in {mo['after']['seconds']}s | xG {mo['after']['xg']}"
        elif t == "SUBSTITUTION_IMPACT":
            fan_h = f"FRESH LEGS FOR {team.upper()}"
            fan = f"{mo['title']}. {ev['Player activity']['detail']}."
            why = "A structural change followed shortly after the substitution." if mo["links"] else "A fresh player in the press."
            lower = f"SUB {team} | {mo['title']}"
        elif t == "RED_CARD_IMPACT":
            a = mo["after"]["opponent"]
            fan_h = f"RED CARD! {team.upper()} DOWN TO TEN"
            fan = f"{opp} now own the ball - {a['territory_pct']}% territory and {a['shots']} shots since the sending off."
            why = f"With ten men, {team} reorganised into {mo['after']['team']['formation']} and conceded territory."
            lower = f"RED CARD | {opp} territory {_fmt(d.get('opp_territory_pct'), ' pts')}"
        else:
            fan_h, fan, why, lower = mo["title"].upper(), mo["title"], "", mo["title"]
        cov = mo.get("coverage") or {}
        vis_note = None
        if cov.get("inferred_avg", 0) > 0.05:
            vis_note = (f"On average {cov['inferred_avg']:.0f} of {cov['observed_avg'] + cov['inferred_avg']:.0f} {SHORT[cov['team']]} "
                        f"players were outside the camera view; their positions were estimated from previous trajectories "
                        f"and team shape (mean confidence {cov['inferred_mean_confidence']:.0%}).")
        return {
            "headline": mo["title"],
            "visibility_note": vis_note,
            "fan_stats": self.fan_stats(mo),
            "why_it_matters": why,
            "analyst": {"summary": why, "bullets": [f"{e['label']}: {e['detail']}" for e in mo["evidence"]]},
            "fan": {"headline": fan_h, "text": fan},
            "broadcast": {"lower_third": lower, "ticker": f"{mo['minute_label']} {mo['title']}"},
            "generator": "deterministic-v1",
        }

    @staticmethod
    def fan_stats(mo: dict) -> list:
        a, t = mo["after"], mo["type"]
        if t == "GOAL_BUILD_UP":
            out = [(a["passes"], "passes"), (f"{a['duration_s']:.0f}s", "start to finish"), (a["xg"], "chance quality (xG)")]
        elif t in ("PRESSURE_BREAK",):
            out = [(a["players_bypassed"], "players bypassed"), (f"{a['seconds_to_final_third']}s", "to the final third"), (a["xg"], "chance quality (xG)")]
        elif t == "COUNTERATTACK":
            out = [(f"{a['progression_m']:.0f}m", "covered"), (f"{a['seconds']}s", "start to shot"), (a["passes"], "passes")]
        elif t == "RED_CARD_IMPACT":
            o = a["opponent"]
            out = [(f"{o['territory_pct']:.0f}%", "opponent territory"), (o["shots"], "opponent shots"), (o["goals"], "goals since")]
        elif t == "SUBSTITUTION_IMPACT":
            ev = {e["label"]: e for e in mo["evidence"]}
            out = [(ev["Player activity"]["value"], "pressures by the sub"), (f"{a['territory_pct']:.0f}%", "team territory"), (a["shots"], "team shots")]
        else:
            out = [(a.get("final_third_recoveries"), "dangerous regains"), (f"{a.get('territory_pct', 0):.0f}%", "territory"), (a.get("shots"), "shots")]
        return [{"value": v, "label": l} for v, l in out]

    def player_summary(self, p: dict, s: dict, moments: list) -> dict:
        bits = []
        if s.get("right_side_actions", 0) >= 5:
            bits.append(f"{s['right_side_actions']} pressing actions on the right side")
        if s.get("final_third_recoveries"):
            bits.append(f"{s['final_third_recoveries']} final-third recoveries")
        if s.get("progressive_passes"):
            bits.append(f"{s['progressive_passes']} progressive passes")
        if s.get("goals"):
            bits.append(f"{s['goals']} goal(s)")
        if s.get("assists"):
            bits.append(f"{s['assists']} assist(s)")
        key = [m["title"] for m in moments]
        return {"headline": f"{p['name'].upper()} — LIVE IMPACT",
                "impact": ("; ".join(bits) or "Quiet involvement so far") + ".",
                "moments": key, "generator": "deterministic-v1"}


def _fact_numbers(obj, out):
    """Every number in the moment's structured data (the only numbers an explanation may use)."""
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 1))
        return out
    if isinstance(obj, str):
        for n in re.findall(r"\d+(?:\.\d+)?", obj):
            out.add(round(float(n), 1))
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            _fact_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _fact_numbers(v, out)
    return out


def numbers_supported(text: str, facts: set) -> bool:
    """True if every number in `text` is (within rounding) a number from the facts; formation names are ignored."""
    clean = re.sub(r"\b\d(?:-\d){2,4}\b", "", text)
    for num in re.findall(r"\d+(?:\.\d+)?", clean):
        v = round(float(num), 1)
        if not (v in facts or round(v) in facts or any(abs(v - f) <= 0.6 for f in facts)
                or round(v / 100, 2) in {round(f, 2) for f in facts}):
            return False
    return True


FOUNDRY_SYSTEM = (
    "You are a football analyst writing for a broadcast studio. You receive one key moment as JSON computed by a "
    "match-analysis engine. Rewrite it as (1) 'why_it_matters': two or three sentences for analysts, and (2) "
    "'fan_text': one or two short sentences for fans. Rules: use ONLY facts and numbers present in the JSON, never "
    "invent or compute statistics; do not claim causation - say 'followed', 'preceded', 'was associated with' or "
    "'the sequence suggests'; if hidden_state is present you may say how many players the camera did not show and "
    "compare camera-only, reconstructed and actual values exactly as given. Reply with a JSON object with keys "
    "why_it_matters and fan_text only."
)


class FoundryNarrativeGenerator(NarrativeGenerator):
    """Microsoft Foundry / Azure OpenAI narratives, constrained to engine evidence.

    Env: AZURE_AI_ENDPOINT (full chat-completions URL incl. api-version, Azure OpenAI deployment or Foundry models
    endpoint), AZURE_AI_KEY, optional AZURE_AI_MODEL (Foundry model name). The LLM only rephrases: every number in its
    reply must exist in the moment JSON, otherwise the deterministic text is kept (narrative.source says which).
    """

    def __init__(self, fallback: Optional[NarrativeGenerator] = None):
        self.endpoint = os.getenv("AZURE_AI_ENDPOINT")
        self.key = os.getenv("AZURE_AI_KEY")
        self.model = os.getenv("AZURE_AI_MODEL")
        self.fallback = fallback or DeterministicNarrativeGenerator()
        self.stats = {"llm": 0, "rejected_numbers": 0, "errors": 0}

    def _call(self, payload: dict) -> Optional[dict]:
        import json
        import urllib.request
        body = {"messages": [{"role": "system", "content": FOUNDRY_SYSTEM},
                             {"role": "user", "content": json.dumps(payload, default=str)}],
                "temperature": 0.2, "max_tokens": 350, "response_format": {"type": "json_object"}}
        if self.model:
            body["model"] = self.model
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "api-key": self.key or "",
                                              "Authorization": f"Bearer {self.key or ''}"})
        with urllib.request.urlopen(req, timeout=12) as r:
            out = json.loads(r.read())
        return json.loads(out["choices"][0]["message"]["content"])

    def generate(self, moment, match_context):
        base = self.fallback.generate(moment, match_context)
        base["source"] = "template"
        if not (self.endpoint and self.key):
            return base
        facts_obj = {k: v for k, v in moment.items() if k not in ("narrative", "counterfactual", "replay", "related_events")}
        facts = _fact_numbers(facts_obj, set())
        try:
            reply = self._call({"teams": {k: v["name"] for k, v in match_context["teams"].items()}, "moment": facts_obj})
        except Exception:
            self.stats["errors"] += 1
            return base
        why, fan = (reply or {}).get("why_it_matters"), (reply or {}).get("fan_text")
        if not (isinstance(why, str) and isinstance(fan, str)) or not (numbers_supported(why, facts) and numbers_supported(fan, facts)):
            self.stats["rejected_numbers"] += 1
            return base
        self.stats["llm"] += 1
        base["why_it_matters"] = why
        base["analyst"] = {**base.get("analyst", {}), "summary": why}
        base["fan"] = {**base["fan"], "text": fan}
        base["source"] = "foundry"
        return base

    def player_summary(self, p, s, moments):
        return self.fallback.player_summary(p, s, moments)


def get_generator() -> NarrativeGenerator:
    if os.getenv("NARRATIVE_BACKEND") == "foundry":
        return FoundryNarrativeGenerator()
    return DeterministicNarrativeGenerator()
