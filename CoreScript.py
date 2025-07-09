
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, degrees, pi, sqrt, acos, atan2

# Construct rotation matrix from body frame to RSN (Radial-velocity-normal) frame
def construct_rotation_from_rsn(nadir_face, velocity_face):
    base_normals = {
        '+X': np.array([1, 0, 0]), '-X': np.array([-1, 0, 0]),
        '+Y': np.array([0, 1, 0]), '-Y': np.array([0, -1, 0]),
        '+Z': np.array([0, 0, 1]), '-Z': np.array([0, 0, -1])
    }

    if nadir_face not in base_normals or velocity_face not in base_normals:
        raise ValueError("Invalid face provided for RSN alignment.")

    # Define RSN frame: R = -x (nadir), S = +y (velocity), N = R x S
    R = -np.array([1, 0, 0])
    S = np.array([0, 1, 0])
    N = np.cross(R, S)
    RSN_matrix = np.column_stack((R, S, N))

    # Define body frame based on user input
    b_r = base_normals[nadir_face]
    b_s = base_normals[velocity_face]
    b_n = np.cross(b_r, b_s)
    body_matrix = np.column_stack((b_r, b_s, b_n))

    return RSN_matrix @ body_matrix


# Julian Date Converter
class JulianDateConverter:
    @staticmethod
    def to_julian_date(year, month, day_decimal):
        if month <= 2:
            year -= 1
            month += 12
        A = int(year / 100)
        B = 2 - A + int(A / 4)
        JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day_decimal + B - 1524.5
        return JD

# Sun Position Model
class SunPosition:

    @staticmethod
    def sun_vector_in_inertial(JD):
        JT = (JD - 2451545.0) / 36525
        L0 = 280.46646 + 36000.76983 * JT + 0.0003032 * JT**2
        L0 %= 360
        M = 357.52911 + 35999.05029 * JT - 0.0001537 * JT**2
        M %= 360
        M_rad = radians(M)
        C = (1.914602 - 0.004817 * JT - 0.000014 * JT**2) * sin(M_rad) + \
            (0.019993 - 0.000101 * JT) * sin(2 * M_rad) + \
            0.00289 * sin(3 * M_rad)
        Theta = L0 + C
        V_s = M + C
        Omega = 125.04 - 1934.136 * JT
        lambda_s = Theta - 0.00569 - 0.00478 * sin(radians(Omega))
        lambda_rad = radians(lambda_s)
        epsilon0 = (23 + 26/60 + 21.448/3600) - ((46.8150 * JT) / 3600) - ((0.00059 * JT**2) / 3600) + ((0.001813 * JT**3) / 3600)
        epsilon = epsilon0 + 0.00256 * cos(radians(Omega))
        epsilon_rad = radians(epsilon)
        alpha = atan2(cos(epsilon_rad) * sin(lambda_rad), cos(lambda_rad))
        delta = np.arcsin(sin(epsilon_rad) * sin(lambda_rad))
        xs = cos(delta) * cos(alpha)
        ys = cos(delta) * sin(alpha)
        zs = sin(delta)
        return np.array([xs, ys, zs]), degrees(alpha) % 360

    

    @staticmethod
    def inertial_to_LVLH(sun_vector_inertial, inclination_rad, RAAN_rad, mean_anomaly_rad):
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
        C_theta = np.array([
            [cos(mean_anomaly_rad), -sin(mean_anomaly_rad), 0],
            [sin(mean_anomaly_rad),  cos(mean_anomaly_rad), 0],
            [0, 0, 1]
        ])
        C_inertial_to_orbital = C_theta @ C_i @ C_Omega
        C_inertial_to_LVLH = C_inertial_to_orbital.T
        return C_inertial_to_LVLH @ sun_vector_inertial

        #cambio de ejes de rotacion sera haciendo un transpose del primer plano a uno segundo 
        #  C_2 = R_1^-T

# Orbit Class
class Orbit:
    def __init__(self, altitude_km, inclination_deg=None, is_sso=False):
        # self.earth_radius = 6378.0
        # self.mu = 398600.0
        # self.J2 = 1.08263e-3
        # self.altitude = altitude_km
        # self.sma = self.earth_radius + altitude_km
        # self.period = 2 * pi * sqrt(self.sma**3 / self.mu)
        # if is_sso:
        #     n = 2 * pi / self.period
        #     p = self.sma
        #     factor = - (4 * pi**2 * p**2) / (365.2411984 * 24 * 3600 * 3 * self.J2 * self.earth_radius**2 * n)
        #     self.inclination = acos(factor)
        # else:
        #     self.inclination = radians(inclination_deg)

        # Constantes
        self.Re = 6378.0               # Radio de la Tierra en km
        self.mu = 398600.0             # Constante gravitacional
        self.J2 = 1.08263e-3           # Oblatez ecuatorial
        self.altitude = altitude_km
        self.a = self.Re + altitude_km # Semi-eje mayor
        self.n = sqrt(self.mu / self.a**3)  # Movimiento medio (rad/s)
        self.is_sso = is_sso

        # Si es órbita Sun-síncrona, calcula la inclinación automáticamente
        if self.is_sso:
            dRAAN_dt = 2 * pi / (365.2422 * 86400)  # Precesión solar deseada (rad/s)
            factor = -2 * dRAAN_dt / (3 * self.J2 * (self.Re / self.a)**2 * self.n)

            if abs(factor) > 1:
                raise ValueError("No se puede calcular inclinación SSO para esta altitud.")

            self.inclination = acos(factor)  # en radianes
            print(f"Computed SSO inclination: {round(self.inclination * 180 / pi, 2)} degrees")

        else:
            self.inclination = radians(inclination_deg)

        # periodo 
        self.period = 2 * pi * sqrt(self.a**3 / self.mu)

    def eclipse_window(self, sun_vector):
        beta = pi/2 - acos(np.dot(sun_vector, [0, 0, 1]))   #is this equivalent?
        Re = self.Re
        h = self.altitude
        rho = np.arcsin(Re / (Re + h))
        try:
            cos_phi_half = np.cos(rho) / np.cos(beta)
        except ZeroDivisionError:
            return None, None
        if abs(cos_phi_half) > 1:
            return None, None
        phi_half = np.arccos(cos_phi_half)
        eclipse_duration = 2 * phi_half * (self.period / (2 * pi))
        start = (self.period - eclipse_duration) / 2
        end = start + eclipse_duration

        if self.is_sso:
            print(f"Computed SSO inclination: {degrees(self.inclination):.2f} degrees")
        return (start, end)
    

# Satellite Class
class Satellite:
    def __init__(self, size_u, panel_faces, occupancy, efficiency):
        self.size_u = size_u
        self.panel_faces = panel_faces
        self.occupancy = occupancy
        self.efficiency = efficiency
        self.base_size = 0.1
        self.panel_areas = self._calculate_panel_areas()

    def _calculate_panel_areas(self):
        x, y, z = self.base_size, self.base_size, self.base_size * self.size_u
        areas = {
            '+X': y * z, '-X': y * z,
            '+Y': x * z, '-Y': x * z,
            '+Z': x * y, '-Z': x * y
        }
        return {face: areas[face] * self.occupancy for face in self.panel_faces}


class PowerAnalyzer:
    SOLAR_CONSTANT = 1367

    def __init__(self, satellite, orbit, RAAN_deg, sun_vector, nadir_condition, nadir_face, velocity_face, pitch_deg=0, yaw_deg=0, roll_deg=0):
        self.sat = satellite
        self.orbit = orbit
        self.inclination = orbit.inclination
        self.RAAN = np.radians(RAAN_deg)
        self.sun_vector_inertial = sun_vector
        self.velocity_face = velocity_face

        self.base_normals = {
            '+X': np.array([1, 0, 0]), '-X': np.array([-1, 0, 0]),
            '+Y': np.array([0, 1, 0]), '-Y': np.array([0, -1, 0]),
            '+Z': np.array([0, 0, 1]), '-Z': np.array([0, 0, -1])
        }

        if nadir_condition and nadir_face and velocity_face:
            R_rsn = construct_rotation_from_rsn(nadir_face, velocity_face)
        else:
            R_rsn = np.identity(3)
        
        # Attitude rotation matrices
        pitch = radians(pitch_deg)
        yaw = radians(yaw_deg)
        roll = radians(roll_deg)
        
        R_pitch = np.array([
            [1, 0, 0],
            [0, cos(pitch), sin(pitch)],
            [0, -sin(pitch),  cos(pitch)]])
        
        R_yaw = np.array([
            [cos(yaw), 0, -sin(yaw)],
            [0, 1, 0],
            [sin(yaw), 0, cos(yaw)]])
        
        R_roll = np.array([
            [cos(roll), sin(roll), 0],
            [-sin(roll),  cos(roll), 0],
            [0, 0, 1]])
        
        R_attitude = R_roll @ R_pitch @ R_yaw
        self.R_total = R_attitude @ R_rsn.T  


        self.rotated_normals = {face: self.R_total @ vec for face, vec in self.base_normals.items()}


    def sun_vector_in_LVLH(self, t):   #multiplied by R_rsn to get body 
        M = (t / self.orbit.period) * 2 * np.pi
        return SunPosition.inertial_to_LVLH(self.sun_vector_inertial, self.inclination, self.RAAN, M) @ self.R_total

    def compute_power(self, t):
        sun_vector_b = self.sun_vector_in_LVLH(t) 
        total_power = 0
        power_per_face = {}
        for face in self.sat.panel_faces:
            normal = self.rotated_normals[face]
            cos_theta = np.dot(sun_vector_b, normal) / np.linalg.norm(sun_vector_b)
            if cos_theta > 0:
                power_face = self.SOLAR_CONSTANT * self.sat.panel_areas[face] * self.sat.efficiency * cos_theta
                total_power += power_face
                power_per_face[face] = power_face
            else:
                power_per_face[face] = 0
        return total_power, power_per_face

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
        velocity_face = input("Which face points to Velocity direction? (e.g. +X): ").strip().upper()
    else:
        nadir_face, velocity_face = None, None


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
        velocity_face=velocity_face,
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
