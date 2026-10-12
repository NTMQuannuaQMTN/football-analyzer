import json

from backend import catalog


def test_provider_match_id_is_not_displayed_as_a_date():
    metadata = {'source': {'game': 'skillcorner/2440', 'provider': 'SkillCorner', 'date': '2019-11-10', 'competition': 'Premier League'}}
    details = catalog.video_details(metadata)
    assert details['date'] == '2019-11-10'
    assert details['competition'] == 'Premier League'
    assert 'SkillCorner' in details['label'] and 'SoccerNet' not in details['label']
    assert catalog.video_details({'source': {'game': 'skillcorner/2440'}})['date'] is None


def test_lobby_preserves_unknown_score_and_exported_date(tmp_path, monkeypatch):
    path = tmp_path/'partial.json'
    path.write_text(json.dumps({'meta': {
        'duration': 60, 'score_known': False, 'initial_score': {'A': 0, 'B': 0},
        'teams': {team: {'name': team, 'color': '#fff'} for team in 'AB'},
        'source': {'game': 'skillcorner/2440', 'date': '2019-11-10', 'provider': 'SkillCorner', 'competition': 'Premier League'}},
        'events': []}))
    monkeypatch.setattr(catalog, '_catalog_static', lambda: ({'id': 'private-partial', 'kind': 'computer_vision', 'label': 'legacy wrong provider', 'parts': {'clip': str(path)}, 'info': {}},))
    card = catalog.lobby()[0]
    assert card['score_known'] is False
    assert card['date'] == '2019-11-10'
    assert card['label'] == 'Computer vision · SkillCorner'
    assert card['note'].endswith('score N/A')


def test_soccernet_date_can_be_recovered_without_a_provider_id_guess():
    details = catalog.video_details({'source': {'game': 'england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool'}})
    assert details['date'] == '2016-08-14'
    assert details['label'] == 'Computer vision · SoccerNet'


def test_experimental_video_label_survives_lobby_provenance_refresh():
    details=catalog.video_details({'source':{'game':'skillcorner/2440','date':'2019-11-10','experimental':True}})
    assert details['label']=='Experimental reconstruction · EPL'
    assert details['date']=='2019-11-10'


def test_synthetic_lobby_does_not_build_reconstruction(monkeypatch):
    generated = []
    original = catalog.generate_match
    def record(seed):
        raw = original(seed)
        generated.append(raw['meta'])
        return raw
    def unexpected(*args, **kwargs):
        raise AssertionError('Lobby should not build full match intelligence')
    monkeypatch.setattr(catalog, 'generate_match', record)
    monkeypatch.setattr(catalog, 'intel', unexpected)
    monkeypatch.setattr(catalog, '_catalog_static', lambda: ({'id': 'sim-7', 'kind': 'synthetic', 'label': 'Synthetic', 'parts': {'full': None}, 'info': {}},))
    catalog._synthetic_meta.cache_clear()
    first = catalog.lobby()[0]
    second = catalog.lobby()[0]
    assert len(generated) == 1
    assert first['score'] == generated[0]['final_score'] == second['score']
    assert first['score_known'] is True
    catalog._synthetic_meta.cache_clear()
