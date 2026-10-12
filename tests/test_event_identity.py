from cv.event_identity import labelled_scorer


def test_scorer_comes_from_goal_fact_and_ambiguous_labels_abstain():
    scorer=dict(shirt_number="14",facts=[dict(description="Goal",time="31' scorer (assist)")])
    caption=dict(lineup=dict(home=dict(players=[scorer])))
    assert labelled_scorer(caption,"home",1,1811.677)==scorer
    assert labelled_scorer(caption,"home",1,1740) is None
    caption['lineup']['home']['players'].append(dict(shirt_number="8",facts=[dict(description="Goal",time="31'")]))
    assert labelled_scorer(caption,"home",1,1811.677) is None


def test_second_half_stoppage_time():
    scorer=dict(shirt_number="9",facts=[dict(description="Goal",time="90+3' scorer")])
    caption=dict(lineup=dict(away=dict(players=[scorer])))
    assert labelled_scorer(caption,"away",2,47*60+15)==scorer
