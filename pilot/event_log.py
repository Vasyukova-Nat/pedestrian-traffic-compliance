"""Журнал событий по определениям из статьи.

Событие = интервал кадров, в течение которого конкретный пешеход (track_id)
находится вне зоны перехода. Повторное уведомление в пределах открытого
события подавляется. Событие закрывается при возврате в зону или при
потере трека дольше GAP_FRAMES.

Границы интервала: событие покрывает только кадры вне ROI.
start_frame — первый кадр вне ROI, end_frame — последний кадр вне ROI.
Кадр возврата в ROI в длительность не входит.

Уведомление (event_open) выдаётся не в момент технического открытия
кандидата, а при подтверждении: когда длительность события достигает
MIN_EVENT_FRAMES. Техническое открытие кандидата пишется отдельным
типом записи "event_candidate".

Пропуски трека: короткие пропуски (< GAP_FRAMES) не продлевают событие
и не входят в длительность; длительность считается по числу кадров,
в которых пешеход реально наблюдался вне ROI. При потере трека событие
закрывается на последнем кадре наблюдения вне ROI. finalize закрывает
открытые события на последнем наблюдении, а не на последнем кадре видео.
"""


class Event:
    def __init__(self, track_id, start_frame, center):
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = None
        self.centers = [center]
        self.close_reason = None
        self.frames_outside = 1   # число кадров, реально наблюдавшихся вне ROI
        self.last_outside_frame = start_frame
        self.notified = False

    @property
    def duration(self):
        # длительность = число кадров, реально наблюдавшихся вне ROI
        return self.frames_outside

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
        self.last_seen = {}    # track_id -> frame_idx (любое наблюдение)
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
                    # техническое открытие кандидата — отдельный тип записи
                    self.notifications.append((frame_idx, track_id, "event_candidate"))
                else:
                    ev = self.open[track_id]
                    ev.centers.append(center)
                    ev.frames_outside += 1
                    ev.last_outside_frame = frame_idx

                ev = self.open[track_id]
                # подтверждение уведомления при достижении порога
                if not ev.notified and ev.duration >= self.min_event_frames:
                    ev.notified = True
                    self.notifications.append((frame_idx, track_id, "event_open"))
            else:
                # вернулся в зону — закрываем событие на последнем кадре вне ROI
                if track_id in self.open:
                    ev = self.open[track_id]
                    self._close(track_id, ev.last_outside_frame,
                                "returned_to_crosswalk")

        # потеря трека: закрываем события тех ID, что давно не видны
        for track_id in list(self.open.keys()):
            if frame_idx - self.last_seen.get(track_id, frame_idx) > self.gap_frames:
                ev = self.open[track_id]
                self._close(track_id, ev.last_outside_frame, "track_lost")

    def _close(self, track_id, end_frame, reason):
        ev = self.open.pop(track_id)
        ev.end_frame = end_frame
        ev.close_reason = reason
        self.events.append(ev)

    def finalize(self, last_frame):
        # закрываем на последнем кадре наблюдения вне ROI, а не на последнем кадре видео
        for track_id in list(self.open.keys()):
            ev = self.open[track_id]
            self._close(track_id, ev.last_outside_frame, "end_of_video")

    def significant_events(self):
        return [e for e in self.events if e.duration and e.duration >= self.min_event_frames]