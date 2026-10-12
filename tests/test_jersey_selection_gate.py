from cv.train_jersey_audit import select_threshold


def test_no_jersey_threshold_is_selected_when_precision_or_sample_count_fails():
    assert select_threshold({'0.5':dict(decided_tracks=100,correct_tracks=89),'0.95':dict(decided_tracks=2,correct_tracks=2)}) is None


def test_eligible_jersey_threshold_uses_actual_counts_and_widest_coverage():
    validation={'0.5':dict(decided_tracks=100,correct_tracks=90),'0.7':dict(decided_tracks=30,correct_tracks=29),
                '0.95':dict(decided_tracks=20,correct_tracks=20)}
    assert select_threshold(validation)==.5
