import plotly.graph_objects as go
import numpy as np

def draw_interactive_cubesat(size_u=3, nadir_face='+Z', velocity_face='+X'):
    # CubeSat dimensions
    width, depth, height = 1, 1, size_u

    # Define cube vertices
    a = [0, 0, 0]
    b = [1, 0, 0]
    c = [1, 1, 0]
    d = [0, 1, 0]
    e = [0, 0, height]
    f = [1, 0, height]
    g = [1, 1, height]
    h = [0, 1, height]
    vertices = np.array([a, b, c, d, e, f, g, h])
    x, y, z = vertices.T

    # Face triangles
    faces = {
        '+Z': [[4, 5, 6], [4, 6, 7]],
        '-Z': [[0, 1, 2], [0, 2, 3]],
        '+X': [[1, 2, 6], [1, 6, 5]],
        '-X': [[0, 3, 7], [0, 7, 4]],
        '+Y': [[2, 3, 7], [2, 7, 6]],
        '-Y': [[0, 1, 5], [0, 5, 4]]
    }

    # All gray by default
    face_colors = {face: 'lightgray' for face in faces}
    face_colors[nadir_face] = 'red'
    face_colors[velocity_face] = 'navy'

    # Face centers and outward direction vectors
    face_centers = {
        '+Z': [0.5, 0.5, height],
        '-Z': [0.5, 0.5, 0],
        '+X': [1, 0.5, height / 2],
        '-X': [0, 0.5, height / 2],
        '+Y': [0.5, 1, height / 2],
        '-Y': [0.5, 0, height / 2]
    }

    directions = {
        '+Z': [0, 0, 1],
        '-Z': [0, 0, -1],
        '+X': [1, 0, 0],
        '-X': [-1, 0, 0],
        '+Y': [0, 1, 0],
        '-Y': [0, -1, 0]
    }

    # --- Rotation matrix helper ---
    def get_rotation_matrix(from_vec, to_vec):
        from_vec = from_vec / np.linalg.norm(from_vec)
        to_vec = to_vec / np.linalg.norm(to_vec)
        v = np.cross(from_vec, to_vec)
        c = np.dot(from_vec, to_vec)
        s = np.linalg.norm(v)
        if s == 0:
            return np.identity(3)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return np.identity(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
    
    # --- Compute rotation only if nadir_face is specified ---
    if nadir_face is not None:
        rot_mat = get_rotation_matrix(np.array(directions[nadir_face]), np.array([0, 0, -1]))
    else:
        rot_mat = np.identity(3)


    # Rotate all vertices and centers
    rotated_vertices = vertices @ rot_mat.T
    x, y, z = rotated_vertices.T
    rotated_face_centers = {f: (np.array(c) @ rot_mat.T) for f, c in face_centers.items()}
    rotated_directions = {f: (np.array(d) @ rot_mat.T) for f, d in directions.items()}

    # --- Plotting ---
    fig = go.Figure()

    for face, triangles in faces.items():
        i, j, k = zip(*triangles)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=face_colors[face],
            opacity=0.6,
            name=face,
            showscale=False
        ))

    for label, face in [('Nadir', nadir_face), ('Velocity', velocity_face)]:
        if face is None:
            continue  # Skip drawing arrow if no face is defined
        if face not in rotated_face_centers or face not in rotated_directions:
            continue  # Avoid crash if something is misaligned
    
        start = rotated_face_centers[face]
        vec = rotated_directions[face]
        end = start + 0.6 * vec
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+text',
            line=dict(color='white', width=6),
            text=['', f'→ {label}'],
            textposition='middle right',
            textfont=dict(size=14),
            showlegend=False
        ))


    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
