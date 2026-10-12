import json

from cv import soccernet
from cv.detect_track import replay_segments


def test_camera_labels_describe_the_ending_shot(tmp_path,monkeypatch):
    rows=[dict(gameTime="1 - 00:10",position="10000",label="Main camera center",replay="real-time"),
          dict(gameTime="1 - 00:20",position="20000",label="Close-up player or field referee",replay="real-time"),
          dict(gameTime="1 - 00:30",position="30000",label="Main camera right",replay="replay"),
          dict(gameTime="2 - 00:05",position="5000",label="Main camera left",replay="real-time")]
    (tmp_path/"Labels-cameras.json").write_text(json.dumps(dict(annotations=rows)))
    monkeypatch.setattr(soccernet,"game_dir",lambda game:tmp_path)
    assert soccernet.main_camera_segments("test",1)==[(0.,10.)]
    assert replay_segments("test",1)==[(20.,30.)]
    assert soccernet.main_camera_segments("test",2)==[(0.,5.)]
    assert soccernet.camera_segments("test",1)[-1][:2]==(20.,30.)
