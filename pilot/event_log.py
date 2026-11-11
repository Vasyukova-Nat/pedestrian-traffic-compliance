"""Журнал событий по определениям из статьи.

Событие = интервал кадров, в течение которого конкретный пешеход (track_id)
находится вне зоны перехода. Повторное уведомление в пределах открытого
события подавляется. Событие закрывается при возврате в зону или при
потере трека дольше GAP_FRAMES.
"""


class Event:
    def __init__(self, track_id, start_frame, center):
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = None
        self.centers = [center]
        self.close_reason = None

    @property
    def duration(self):
        return (self.end_frame - self.start_frame + 1) if self.end_frame is not None else None

    def as_dict(self):
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration": self.duration,
            "close_reason": self.close_reason,
        }


class EventLog:
    def __init__(self, gap_frames, min_event_frames):
        self.gap_frames = gap_frames
        self.min_event_frames = min_event_frames
        self.open = {}         # track_id -> Event
        self.last_seen = {}    # track_id -> frame_idx
        self.events = []       # закрытые события
        self.notifications = []  # (frame_idx, track_id, kind)

    def update(self, frame_idx, detections):
        """detections: список (track_id, center, in_crosswalk)."""
        seen_ids = set()
        for track_id, center, in_crosswalk in detections:
            seen_ids.add(track_id)
            self.last_seen[track_id] = frame_idx

            if not in_crosswalk:
                if track_id not in self.open:
                    ev = Event(track_id, frame_idx, center)
                    self.open[track_id] = ev
                    # единственное уведомление на открытие события
                    self.notifications.append((frame_idx, track_id, "event_open"))
                else:
                    self.open[track_id].centers.append(center)
            else:
                # вернулся в зону — закрываем событие
                if track_id in self.open:
                    self._close(track_id, frame_idx, "returned_to_crosswalk")

        # потеря трека: закрываем события тех ID, что давно не видны
        for track_id in list(self.open.keys()):
            if frame_idx - self.last_seen.get(track_id, frame_idx) > self.gap_frames:
                self._close(track_id, self.last_seen.get(track_id, frame_idx),
                            "track_lost")

    def _close(self, track_id, end_frame, reason):
        ev = self.open.pop(track_id)
        ev.end_frame = end_frame
        ev.close_reason = reason
        self.events.append(ev)

    def finalize(self, last_frame):
        for track_id in list(self.open.keys()):
            self._close(track_id, last_frame, "end_of_video")

    def significant_events(self):
        return [e for e in self.events if e.duration and e.duration >= self.min_event_frames]