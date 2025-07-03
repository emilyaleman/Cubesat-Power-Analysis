import plotly.graph_objects as go
import numpy as np

def draw_interactive_cubesat(size_u=3, nadir_face='+Z', panel_faces=None):
    if panel_faces is None:
        panel_faces = []

    height = size_u
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, height], [1, 0, height], [1, 1, height], [0, 1, height]
    ], dtype=float)

    faces = {
        '+Z': [[4, 5, 6], [4, 6, 7]],
        '-Z': [[0, 1, 2], [0, 2, 3]],
        '+X': [[1, 2, 6], [1, 6, 5]],
        '-X': [[0, 3, 7], [0, 7, 4]],
        '+Y': [[2, 3, 7], [2, 7, 6]],
        '-Y': [[0, 1, 5], [0, 5, 4]]
    }

    def get_color(face):
        if face == nadir_face:
            return 'red'
        elif face in panel_faces:
            return 'gold'
        else:
            return 'lightgray'

    face_centers = {
        '+Z': [0.5, 0.5, height], '-Z': [0.5, 0.5, 0],
        '+X': [1, 0.5, height/2], '-X': [0, 0.5, height/2],
        '+Y': [0.5, 1, height/2], '-Y': [0.5, 0, height/2]
    }
    directions = {
        '+Z': [0, 0, 1], '-Z': [0, 0, -1],
        '+X': [1, 0, 0], '-X': [-1, 0, 0],
        '+Y': [0, 1, 0], '-Y': [0, -1, 0]
    }

    fig = go.Figure()

    # Draw faces
    for face, tris in faces.items():
        i, j, k = zip(*tris)
        fig.add_trace(go.Mesh3d(
            x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=i, j=j, k=k,
            color=get_color(face), opacity=0.5,
            name=face, showscale=False
        ))

    # Add Nadir arrow
    if nadir_face in face_centers:
        start = np.array(face_centers[nadir_face], dtype=float)
        direction = np.array(directions[nadir_face], dtype=float)
        end = start + 0.6 * direction
        label_pos = start + 0.75 * direction

        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode='lines',
            line=dict(color='white', width=5),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
            mode='text',
            text=["Nadir"],
            textfont=dict(color='white', size=16),
            showlegend=False
        ))

    # Rotate CubeSat so Nadir is down
    up = np.array([0, 0, 1])
    target = np.array(directions[nadir_face], dtype=float)
    axis = np.cross(target, up)
    angle = np.arccos(np.clip(np.dot(target, up), -1, 1))
    if np.linalg.norm(axis) > 0 and angle != 0:
        axis = axis / np.linalg.norm(axis)
        ux, uy, uz = axis
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c + ux**2*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s, c + uy**2*(1-c), uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz**2*(1-c)]
        ])
        rotated = vertices @ R.T
        for trace in fig.data:
            if isinstance(trace, go.Mesh3d):
                trace.update(x=rotated[:,0], y=rotated[:,1], z=rotated[:,2])

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=1.5),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
