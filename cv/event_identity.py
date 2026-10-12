"""Resolve factual goalscorers from lineup event labels, never from proximity guesses."""
import re


def labelled_scorer(caption,side,half,time_in_half):
    minute=(45 if half==2 else 0)+int(time_in_half//60)+1
    candidates=[]
    for player in caption.get("lineup",{}).get(side,{}).get("players",[]):
        for fact in player.get("facts",[]):
            if fact.get("description","").strip().lower()!="goal":
                continue
            match=re.match(r"^(\d+)(?:\+(\d+))?'",fact.get("time",""))
            if match and int(match[1])+int(match[2] or 0)==minute:
                candidates.append(player)
    # Two same-minute goals cannot be assigned from this coarse source.
    return candidates[0] if len(candidates)==1 else None
