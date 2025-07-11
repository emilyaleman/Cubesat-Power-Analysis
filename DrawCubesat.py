# import plotly.graph_objects as go
# import numpy as np

# def draw_cubesat(nadir_face='+Z', velocity_face='+X'):
#     # Define 1x1x1 cube vertices
#     x = [0, 1]
#     y = [0, 1]
#     z = [0, 3]  # For 3U CubeSat

#     # Define face centers for arrows
#     face_normals = {
#         '+X': ([1, 0.5, 1.5], [1, 0.5, 1.5], [0, 3]),
#         '-X': ([0, 0.5, 1.5], [0, 0.5, 1.5], [0, 3]),
#         '+Y': ([0.5, 1.5, 1], [1, 1, 0.5], [0, 3]),
#         '-Y': ([0.5, 1.5, 1], [0, 0, 0.5], [0, 3]),
#         '+Z': ([0.5, 0.5, 1], [0.5, 0.5, 1], [3, 3, 3]),
#         '-Z': ([0.5, 0.5, 1], [0.5, 0.5, 1], [0, 0, 0])
#     }

#     # Define colors for all faces
#     face_colors = {'+X': 'lightgray', '-X': 'lightgray', '+Y': 'lightgray',
#                    '-Y': 'lightgray', '+Z': 'lightgray', '-Z': 'lightgray'}
#     if nadir_face in face_colors:
#         face_colors[nadir_face] = 'red'
#     if velocity_face in face_colors:
#         face_colors[velocity_face] = 'blue'

#     fig = go.Figure()

#     # Add cube as mesh
#     fig.add_trace(go.Mesh3d(
#         x=[0, 1, 1, 0, 0, 1, 1, 0],
#         y=[0, 0, 1, 1, 0, 0, 1, 1],
#         z=[0, 0, 0, 0, 3, 3, 3, 3],
#         i=[0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
#         j=[1, 2, 3, 2, 3, 3, 0, 0, 5, 6, 7, 4],
#         k=[2, 3, 0, 3, 0, 1, 1, 2, 6, 7, 4, 5],
#         opacity=0.6,
#         color='lightgray',
#         flatshading=True
#     ))

#     # Add arrows for nadir and velocity
#     for face, label, color in [(nadir_face, "Nadir", "red"), (velocity_face, "Velocity", "blue")]:
#         dx, dy, dz = direction_vectors(face)
#         cx, cy, cz = face_center(face)
#         fig.add_trace(go.Cone(x=[cx], y=[cy], z=[cz],
#                               u=[dx], v=[dy], w=[dz],
#                               sizemode="absolute", sizeref=0.4,
#                               colorscale=[[0, color], [1, color]],
#                               showscale=False,
#                               anchor="tail",
#                               name=label))

#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectratio=dict(x=1, y=1, z=3)
#         ),
#         title="3U CubeSat with Orientation Arrows"
#     )

#     return fig

import matplotlib.pyplot as plt

def draw_cubesat(nadir_face='+Z', velocity_face='+X'):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw cube outline
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, linewidth=2))
    ax.text(0, 0, 'CubeSat', fontsize=12, ha='center', va='center')

    # Define direction labels and vectors
    arrows = {
        'Nadir': {'vector': [0, -1], 'color': 'red', 'face': nadir_face},
        'Velocity': {'vector': [1, 0], 'color': 'blue', 'face': velocity_face},
        'X': {'vector': [1, 0], 'color': 'gray'},
        'Y': {'vector': [0, 1], 'color': 'gray'},
        'Z': {'vector': [0.7, 0.7], 'color': 'gray'},  # for fun
    }

    # Draw arrows
    for label, props in arrows.items():
        vec = props['vector']
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1, fc=props['color'], ec=props['color'])
        ax.text(vec[0]*1.2, vec[1]*1.2, f"{label}\n({props.get('face', '')})", color=props['color'],
                ha='center', va='center', fontsize=10)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("CubeSat Orientation (Schematic View)")
    plt.tight_layout()

    return fig

