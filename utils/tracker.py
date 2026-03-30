import math

class ObjectCounter:
    def __init__(self):
        self.counted_ids = set()
        self.total_count = 0

    def count(self, track_id):
        if track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.total_count += 1
        return self.total_count
