import streamlit as st
import pandas as pd
from Script_to_function import run_simulation

from PIL import Image

logo = Image.open("SFS27792.png")
st.image(logo, width=150)
st.title("CubeSat Power Simulation")

# --- Inputs ---
is_sso = st.radio("Is this an SSO orbit?", ["Yes", "No"]) == "Yes"
size_u = st.slider("CubeSat size (U)", 1, 6, 3)
nadir_condition = st.checkbox("Is the CubeSat Nadir Pointing?", value=True)
nadir_face = st.selectbox("Which face points to Nadir?", ['+X', '-X', '+Y', '-Y', '+Z', '-Z']) if nadir_condition else None

pitch_deg = st.number_input("Pitch angle (degrees)", value=0.0)
yaw_deg = st.number_input("Yaw angle (degrees)", value=0.0)
roll_deg = st.number_input("Roll angle (degrees)", value=0.0)

panel_faces = st.multiselect("Panel faces", ['+X', '-X', '+Y', '-Y', '+Z', '-Z'], default=['+Z'])
occupancy = st.slider("Panel occupancy", 0.0, 1.0, 0.5)
efficiency = st.slider("Panel efficiency", 0.0, 1.0, 0.6)
altitude_km = st.number_input("Altitude (km)", value=500.0)

if not is_sso:
    inclination_deg = st.number_input("Inclination (degrees)", value=98.0)
    LTDN = None
else:
    inclination_deg = 0  # Dummy value for constructor
    LTDN = st.number_input("Local Time of Descending Node (LTDN, hrs)", value=10.5)

date_str = st.text_input("Date (YYYY-MM-DD.dddd)", value="2025-06-13")
num_orbits = st.number_input("Number of orbits to simulate", min_value=1, value=1)

# --- Run ---
if st.button("Run Simulation"):
    st.write("Running simulation...")

    try:
        fig, fig2, df = run_simulation(
            is_sso=is_sso,
            size_u=size_u,
            nadir_condition=nadir_condition,
            nadir_face=nadir_face,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            panel_faces=panel_faces,
            occupancy=occupancy,
            efficiency=efficiency,
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            date_str=date_str,
            LTDN=LTDN,
            num_orbits=int(num_orbits)
        )

        st.pyplot(fig)
        st.pyplot(fig2)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "simulation_output.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")

