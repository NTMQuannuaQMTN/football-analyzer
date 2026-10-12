import numpy as np

from cv.train_person_embedding import cross_clip_retrieval


def test_identity_retrieval_excludes_same_clip_and_requires_gallery_identity():
    identities={f'{clip}:{i}':(f'game:half:team:{i}','game:half:team') for clip in ['a','b'] for i in [1,2]}
    tracks=np.array([t for t in identities for _ in range(6)])
    z=np.array([[1,0] if t.endswith(':1') else [0,1] for t in tracks],float)
    result=cross_clip_retrieval(z,tracks,identities)
    assert result['queries']==4 and result['top1']==1
    one=tracks[np.char.startswith(tracks,'a:')]
    assert cross_clip_retrieval(z[:len(one)],one,identities)['queries']==0
