"""Shared participant-baseline solar clock for right-closed hourly energy."""
import numpy as np
import pandas as pd


def solar_clock(ts):
    """Return midpoint solar minutes at 135E and solar declination (radians).

    Add 4 * (longitude - 135) minutes for the site's apparent solar time.
    Preserve the participant baseline's endpoint-based annual approximation.
    """
    ts = pd.DatetimeIndex(ts).tz_convert('Asia/Seoul')
    doy = ts.day_of_year.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                       - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
            - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
            - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    return (hour - 0.5)*60 + eqtime, decl
