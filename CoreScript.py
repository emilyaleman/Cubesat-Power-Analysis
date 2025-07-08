
def gst_from_julian(JD):
    D = JD - 2451545.0
    GST = 280.46061837 + 360.98564736629 * D
    return radians(GST % 360)

def satellite_eci_position(orbit, RAAN_rad, inclination_rad, mean_anomaly_rad):
    r_orbital = np.array([orbit.a * cos(mean_anomaly_rad), orbit.a * sin(mean_anomaly_rad), 0])
    C_Omega = np.array([
        [cos(RAAN_rad), -sin(RAAN_rad), 0],
        [sin(RAAN_rad),  cos(RAAN_rad), 0],
        [0, 0, 1]
    ])
    C_i = np.array([
        [1, 0, 0],
        [0, cos(inclination_rad), -sin(inclination_rad)],
        [0, sin(inclination_rad),  cos(inclination_rad)]
    ])
    return C_Omega @ C_i @ r_orbital

def eci_to_ecef(r_eci, JD):
    gst = gst_from_julian(JD)
    R = np.array([
        [cos(-gst), -sin(-gst), 0],
        [sin(-gst),  cos(-gst), 0],
        [0, 0, 1]
    ])
    return R @ r_eci

def ecef_to_latlon(r_ecef):
    x, y, z = r_ecef
    lon = atan2(y, x)
    lat = np.arcsin(z / sqrt(x**2 + y**2 + z**2)) 
    return degrees(lat), degrees(lon)

def compute_lat_lon(orbit, JD, RAAN_rad, inclination_rad, t):
    mean_anomaly_rad = (t / orbit.period) * 2 * pi
    r_eci = satellite_eci_position(orbit, RAAN_rad, inclination_rad, mean_anomaly_rad)
    r_ecef = eci_to_ecef(r_eci, JD + (t / 86400))
    return ecef_to_latlon(r_ecef)

    
def main():
   # Main Input Section 
    is_sso = input("Is this an SSO orbit? (Y/N): ").strip().upper() == 'Y'
    size_u = float(input("Enter CubeSat size (U): "))
    nadir_condition = input("Is the CubeSat Nadir Pointing? (Y/N): ").strip().upper() == 'Y'
    if nadir_condition:
        nadir_face = input("Which face points to Nadir? (e.g. +Z): ").strip().upper()
    else:
        nadir_face = None  # or set a default, or skip reorientation

    # Angulos de actitud con respecto al nadir
    pitch_deg = float(input("Enter pitch angle in degrees (+X axis or default 0): ") or "0")
    yaw_deg   = float(input("Enter yaw angle in degrees (+Y axis, default 0): ") or "0")
    roll_deg  = float(input("Enter roll angle in degrees (+Z axis, default 0): ") or "0")

    panel_faces = input("Enter panel faces (e.g. +Z,+X,-X,+Y): ").replace(" ", "").split(",")
    occupancy = float(input("Enter panel occupancy (0-1): "))
    efficiency = float(input("Enter panel efficiency (0-1): "))
    altitude_km = float(input("Enter orbit altitude (km): "))

    if not is_sso:
        inclination_deg = float(input("Enter orbit inclination (deg): "))
        orbit = Orbit(altitude_km, inclination_deg, is_sso=False)
    else:
        orbit = Orbit(altitude_km, 0, is_sso=True)  
        

    date_input = input("Enter date (YYYY-MM-DD.dddd): ")
    year, month, day = date_input.split("-")
    JD = JulianDateConverter.to_julian_date(int(year), int(month), float(day))
    sun_vector, sun_longitude_deg = SunPosition.sun_vector_in_inertial(JD)

        #sun_longitude deg should be alpha in degrees 

    if is_sso:
        LTDN = float(input("Enter Local Time of Descending Node (hours, e.g. 10.5): "))
        RAAN_deg = (sun_longitude_deg + 15 * (LTDN - 12)) % 360
        print(f"Sun longitude: {sun_longitude_deg:.2f} degrees")
        print(RAAN_deg)
        print(f"Computed RAAN for SSO: {RAAN_deg:.2f} degrees")
        orbit = Orbit(altitude_km, is_sso=True)  # Notice: no inclination_deg passed
    else:
        inclination_deg = float(input("Enter orbit inclination (deg): "))
        RAAN_deg = float(input("Enter RAAN (deg): "))
        orbit = Orbit(altitude_km, inclination_deg, is_sso=False)

        # Create the Satellite and PowerAnalyzer
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
    sun_vector_lvlh = SunPosition.inertial_to_LVLH(sun_vector, orbit.inclination, radians(RAAN_deg), 0)
    print(f"Sun vector in LVLH: {sun_vector_lvlh}")
    beta_rad = np.arcsin(sun_vector_lvlh[2])    # signed beta (in radians)
    beta_deg = degrees(beta_rad)
    beta = abs(beta_deg)
    print(f"Beta angle: {beta_deg:.2f} degrees (absolute: {beta:.2f} degrees)")

    eclipse = orbit.eclipse_window(sun_vector)
    if eclipse:
        eclipse_start, eclipse_end = eclipse
    else:
        eclipse_start, eclipse_end = None, None

    num_orbits = int(input("Enter number of orbits to simulate: "))
    times = np.linspace(0, num_orbits * orbit.period, 500 * num_orbits)
    total_powers = []
    powers_faces = {face: [] for face in panel_faces}
    lat_values = []
    lon_values = []

    beta_values = []

    for t in times:
        JD_t = JD + (t / 86400.0)
        sun_vector_inertial_t, lambda_s_t = SunPosition.sun_vector_in_inertial(JD_t)
        # print(SunPosition.sun_vector_in_inertial(JD_t))
        xs, ys, zs = sun_vector_inertial_t
        alpha_t = atan2(ys, xs)
        delta_t = np.arcsin(zs)
            # print(alpha_t, delta_t)
        lat, lon = compute_lat_lon(orbit, JD, radians(RAAN_deg), orbit.inclination, t)
        lat_values.append(lat)
        lon_values.append(lon)
        sun_vector_lvlh_t = SunPosition.inertial_to_LVLH(sun_vector_inertial_t, orbit.inclination, radians(RAAN_deg), (t / orbit.period) * 2 * pi)
        beta_t = np.arcsin(sun_vector_lvlh_t[2])
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

    average_power = np.mean(total_powers)
    plt.axhline(y=average_power, color='red', linestyle='--', label=f'Average Power = {average_power:.2f} W')

        # Plot total power
    plt.plot(times / 60, total_powers)
    plt.title("Total Power vs Time (Nadir Pointing, 1 Orbit)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Power (W)")
    plt.grid()
    plt.show()


        # Plot power per face
    for face in panel_faces:
        plt.plot(times / 60, powers_faces[face], label=f'{face}')
    plt.title("Power per Face vs Time (Nadir Pointing, 1 Orbit)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid()
    plt.show()

    import csv

        # Resultados en CSV
    output_filename = "simulation_output.csv"

    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (min)", "Total Power (W)", "Beta (deg)", "RA (deg)", "DEC (deg)"])
        for i in range(len(times)):
            writer.writerow([
                times[i] / 60,
                total_powers[i], 
                beta_values[i],
                lat_values[i],
                lon_values[i]
            ])

    print(f"\nSimulation output saved to: {output_filename}")
    print(sun_vector)
     

if __name__ == "__main__":
    main()



    


    

