"""Give reset-local anonymous tracker IDs a causal, shot-qualified namespace."""


class TrackNamespace:
    def __init__(self):
        self.supported = False
        self.shot = -1
        self.next_id = 1
        self.identities = {}

    def camera(self, supported):
        supported = bool(supported)
        if supported and not self.supported:
            self.shot += 1
        self.supported = supported

    def identity(self, native_id):
        native_id = int(native_id)
        if native_id < 0:
            return native_id
        if not self.supported:
            raise ValueError('Cannot qualify a measured track in an unsupported camera interval')
        key = self.shot, native_id
        if key not in self.identities:
            self.identities[key] = self.next_id
            self.next_id += 1
        return self.identities[key]


def qualify_rows(rows):
    """Copy IDs only; preserve all measurements and leave the cache untouched."""
    namespace = TrackNamespace()
    output = []
    for row in rows:
        if 'live' not in row:
            raise ValueError('Shot qualification needs cached live/camera-supported flags; use explicit legacy mode for older caches')
        namespace.camera(row['live'])
        detections = []
        for original in row['dets']:
            detection = dict(original)
            if detection['cls'] in ('person', 'referee') and detection['id'] >= 0:
                native_id = detection.get('native_id', detection['id'])
                detection.update(id=namespace.identity(native_id), native_id=native_id, shot_id=namespace.shot)
            detections.append(detection)
        output.append({**row, 'dets': detections, 'shot_id': namespace.shot if row['live'] else None})
    return output
