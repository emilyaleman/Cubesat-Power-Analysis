import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, degrees, pi, sin, cos, atan2
from CoreScript import JulianDateConverter
from CoreScript import SunPosition
from CoreScript import Orbit
from CoreScript import Satellite
from CoreScript import PowerAnalyzer
from CoreScript import compute_lat_lon


def run_simulation(
    is_sso,
    size_u,
    nadir_condition,
    nadir_face,
    pitch_deg,
    yaw_deg,
    roll_deg,
    panel_faces,
    occupancy,
    efficiency,
    altitude_km,
    inclination_deg,
    date_str,
    LTDN,
    num_orbits
):
    # 1. Convert date to Julian Date
    year, month, day = date_str.split("-")
    JD = JulianDateConverter.to_julian_date(int(year), int(month), float(day))

    # 2. Compute sun vector and RA (alpha)
    sun_vector, sun_RA_deg = SunPosition.sun_vector_in_inertial(JD)

    # 3. RAAN calculation based on orbit type
    if is_sso:
        RAAN_deg = (sun_RA_deg + 15 * (LTDN - 12)) % 360
        orbit = Orbit(altitude_km, is_sso=True)
    else:
        RAAN_deg = float(inclination_deg)  # RAAN is provided explicitly in non-SSO case
        orbit = Orbit(altitude_km, inclination_deg, is_sso=False)

    # 4. Initialize Satellite and PowerAnalyzer
    sat = Satellite(size_u, panel_faces, occupancy, efficiency)
    pa = PowerAnalyzer(
        satellite=sat,
        orbit=orbit,
        RAAN_deg=RAAN_deg,
        sun_vector=sun_vector,
        nadir_condition=nadir_condition,
        nadir_face=nadir_face,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=roll_deg
    )

    # 5. Prepare time array and initialize storage
    times = np.linspace(0, num_orbits * orbit.period, 500 * num_orbits) #why a step size so large?
    total_powers = []
    powers_faces = {face: [] for face in panel_faces}
    ra_values, dec_values, beta_values = [], [], []

    eclipse = orbit.eclipse_window(sun_vector)
    if eclipse:
        eclipse_start, eclipse_end = eclipse
    else:
        eclipse_start, eclipse_end = None, None

    # 6. Loop and simulate
    for t in times:
        JD_t = JD + (t / 86400.0)
        sun_vec_t, alpha_deg = SunPosition.sun_vector_in_inertial(JD_t)
        xs, ys, zs = sun_vec_t
        alpha_t = atan2(ys, xs)
        delta_t = np.arcsin(zs)
        lat, lon = compute_lat_lon(orbit, JD, radians(RAAN_deg), orbit.inclination, t)
        latitudes.append(lat)
        longitudes.append(lon)


        sun_lvlh_t = SunPosition.inertial_to_LVLH(sun_vec_t, orbit.inclination, radians(RAAN_deg), (t / orbit.period) * 2 * pi)
        beta_t = np.arcsin(sun_lvlh_t[2])
        beta_values.append(degrees(beta_t))

        in_eclipse = eclipse and (eclipse_start <= t <= eclipse_end)
        if in_eclipse:
            total_powers.append(0)
            for face in panel_faces:
                powers_faces[face].append(0)
        else:
            total_power, power_per_face = pa.compute_power(t)
            total_powers.append(total_power)
            for face in panel_faces:
                powers_faces[face].append(power_per_face[face])

    # 7. Plot total power
    fig, ax = plt.subplots()
    ax.plot(times / 60, total_powers)
    ax.set_title("Total Power vs Time")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Power (W)")
    ax.grid(True)
    average_power = np.mean(total_powers)
    ax.axhline(y=average_power, color='red', linestyle='--', label=f'Avg Power = {average_power:.2f} W')
    ax.legend()

    # 8. Plot power per face
    fig2, ax2 = plt.subplots()
    for face in panel_faces:
        ax2.plot(times / 60, powers_faces[face], label=f'{face}')
    ax2.set_title("Power per Face vs Time")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Power (W)")
    ax2.grid(True)
    ax2.legend()

    # 9. Create output DataFrame
    df = pd.DataFrame({
    "Time (min)": times / 60,
    "Total Power (W)": total_powers,
    "Beta (deg)": beta_values,
    "Latitude (deg)": latitudes,
    "Longitude (deg)": longitudes})

    return fig, fig2, df
