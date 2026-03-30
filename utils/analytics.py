from collections import deque
from datetime import datetime

def traffic_density(vehicle_count: int) -> str:
    """
    Determine traffic density based on the number of vehicles.

    Args:
        vehicle_count (int): Number of vehicles detected.

    Returns:
        str: 'Low', 'Medium', or 'High' traffic density.
    """
    if vehicle_count < 10:
        return "Low"
    elif vehicle_count < 25:
        return "Medium"
    else:
        return "High"

class TrafficHistory:
    """
    Stores historical traffic data for vehicles and persons, with timestamps.

    Attributes:
        vehicles (deque): Vehicle counts over time.
        persons (deque): Person counts over time.
        timestamps (deque): Corresponding timestamps.
        peak_vehicles (int): Maximum number of vehicles observed.
    """

    def __init__(self, maxlen: int = 100):
        """
        Initialize the TrafficHistory object with fixed-length deques.

        Args:
            maxlen (int, optional): Maximum number of historical records to store. Defaults to 100.
        """
        self.vehicles = deque(maxlen=maxlen)
        self.persons = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        self.peak_vehicles = 0

    def record(self, vehicles: int, persons: int):
        """
        Record a new data point for vehicles and persons.

        Args:
            vehicles (int): Number of vehicles detected.
            persons (int): Number of persons detected.
        """
        self.vehicles.append(vehicles)
        self.persons.append(persons)
        self.timestamps.append(datetime.now())

        # Update peak vehicles if current count exceeds previous peak
        if vehicles > self.peak_vehicles:
            self.peak_vehicles = vehicles

    def as_chart_data(self):
        """
        Prepare historical data for charting.

        Returns:
            list[dict]: List of dictionaries with keys 'time', 'vehicles', 'persons'.
        """
        return [
            {"time": t.strftime("%H:%M:%S"), "vehicles": v, "persons": p}
            for t, v, p in zip(self.timestamps, self.vehicles, self.persons)
        ]

