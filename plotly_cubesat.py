import plotly.graph_objects as go
import numpy as np

def draw_interactive_cubesat(size_u=3, nadir_face='+Z'):
    # Define CubeSat dimensions (1U x 1U x size_u U)
    width, depth, height = 1, 1, size_u

    # Define the 8 corner vertices
    a, b, c, d = [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    e, f, g, h = [0, 0, height], [1, 0, height], [1, 1, height], [0, 1, height]
    vertices = np.array([a, b, c, d, e, f, g, h])
    x, y, z = vertices.T

    # Triangular faces for each CubeSat panel
    faces = {
        '+Z': [[4, 5, 6], [4, 6, 7]],
        '-Z': [[0, 1, 2], [0, 2, 3]],
        '+X': [[1, 2, 6], [1, 6, 5]],
        '-X': [[0, 3, 7], [0, 7, 4]],
        '+Y': [[2, 3, 7], [2, 7, 6]],
        '-Y': [[0, 1, 5], [0, 5, 4]]
    }

    # Assign face colors (nadir face in red)
    face_colors = {
        '+Z': 'lightblue', '-Z': 'lightgray',
        '+X': 'lightgreen', '-X': 'orange',
        '+Y': 'violet', '-Y': 'pink'
    }

    fig = go.Figure()

    # Plot each face of the CubeSat
    for face, triangles in faces.items():
        i, j, k = zip(*triangles)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='red' if face == nadir_face else face_colors[face],
            opacity=0.5,
            name=face,
            showscale=False
        ))

    # Define center points and outward vectors for each face
    face_centers = {
        '+Z': [0.5, 0.5, height],
        '-Z': [0.5, 0.5, 0],
        '+X': [1, 0.5, height / 2],
        '-X': [0, 0.5, height / 2],
        '+Y': [0.5, 1, height / 2],
        '-Y': [0.5, 0, height / 2]
    }

    outward_directions = {
        '+Z': [0, 0, 1],
        '-Z': [0, 0, -1],
        '+X': [1, 0, 0],
        '-X': [-1, 0, 0],
        '+Y': [0, 1, 0],
        '-Y': [0, -1, 0]
    }

    # Draw Nadir vector arrow from selected face
    if nadir_face in face_centers:
        start = np.array(face_centers[nadir_face])
        direction = np.array(outward_directions[nadir_face])
        arrow_length = 0.8
        end = start + arrow_length * direction
        label_pos = start + 1.0 * direction

        # Arrow (white line)
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='white', width=6),
            showlegend=False
        ))

        # Label at tip
        fig.add_trace(go.Scatter3d(
            x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
            mode='text',
            text=["Nadir"],
            textposition="top center",
            textfont=dict(color='white', size=16),
            showlegend=False
        ))

    # Clean layout with zoomed-out view
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=size_u * 1.1),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
