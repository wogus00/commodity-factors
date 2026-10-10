"""NYMEX settlement dates, with explicit provenance for future projections.

2026 closures are from CME's published trading-day schedule. Other years use
the regular holiday pattern and are labeled projected, rather than verified.
Ad hoc exchange closures can be supplied in configs/settlement_calendar.json.
Globex trading on a holiday does not imply a daily settlement is published.
"""

from datetime import date, timedelta

from dateutil.easter import easter


def _nth_weekday(year, month, weekday, nth):
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (nth - 1))


def _observed(value):
    return value + timedelta(days=1 if value.weekday() == 6 else -1 if value.weekday() == 5 else 0)


class SettlementCalendar:
    def __init__(self, config):
        self.config = config

    def closures(self, year):
        verified = self.config.get("verified_years", {}).get(str(year))
        if verified:
            closed = {date.fromisoformat(day) for day in verified["closed"]}
        else:
            # New Year's on Saturday does not close the preceding Friday.
            new_year = date(year, 1, 1)
            closed = {
                new_year + timedelta(days=1 if new_year.weekday() == 6 else 0),
                _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
                easter(year) - timedelta(days=2),
                _nth_weekday(year, 5, 0, 5) if _nth_weekday(year, 5, 0, 5).month == 5
                else _nth_weekday(year, 5, 0, 4),
                _observed(date(year, 6, 19)), _observed(date(year, 7, 4)),
                _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4),
                _observed(date(year, 12, 25)),
            }
        closed.update(date.fromisoformat(day) for day in self.config.get("extra_closed", []))
        closed.difference_update(date.fromisoformat(day) for day in self.config.get("extra_open", []))
        return closed

    def is_session(self, value):
        value = date.fromisoformat(str(value)[:10])
        return value.weekday() < 5 and value not in self.closures(value.year)

    def next_session(self, value):
        day = date.fromisoformat(str(value)[:10]) + timedelta(days=1)
        while not self.is_session(day):
            day += timedelta(days=1)
        verified = str(day.year) in self.config.get("verified_years", {})
        basis = "CME NYMEX settlement calendar" if verified else "Projected NYMEX settlement session; regular holiday rules"
        return day.isoformat(), basis

    def previous_session(self, value):
        day = date.fromisoformat(str(value)[:10])
        while not self.is_session(day):
            day -= timedelta(days=1)
        return day.isoformat()
