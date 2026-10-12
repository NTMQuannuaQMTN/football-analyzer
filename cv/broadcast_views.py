"""Shot categories, not camera accuracy or replay timestamps inferred from pixels."""
import math


def view_kind(annotation):
    label=annotation['label']
    if label in ('Main camera center','Main camera left','Main camera right'):return 'sideline'
    if label.startswith('Close-up'):return 'closeup'
    if label in ('Main behind the goal','Spider camera'):return 'alternate'
    return 'other'


def shot_split(start,end,review_times):
    """Entire annotated shots stay together; manual review shots never train."""
    if any(start<=t<end for t in review_times):return 'review'
    block=int((start+end)/2//300)
    return 'val' if block in (2,6) else 'test' if block in (4,8) else 'train'


def samples(segments,review_times,stride=3.,boundary_margin=1.):
    if stride<=0 or boundary_margin<0:raise ValueError('Invalid sampling settings')
    result=[]
    for shot,(start,end,annotation) in enumerate(segments):
        if annotation.get('replay','real-time')!='real-time':continue
        part=shot_split(start,end,review_times)
        kind=view_kind(annotation)
        t=math.ceil((start+boundary_margin)/stride)*stride
        while t<end-boundary_margin:
            # Buffer temporal split boundaries even when neighbouring shots differ.
            if min(t%300,300-t%300)>=15:
                result.append(dict(video_s=round(t,3),shot=shot,part=part,kind=kind,
                    target=int(kind=='sideline'),label=annotation['label']))
            t+=stride
    return result
