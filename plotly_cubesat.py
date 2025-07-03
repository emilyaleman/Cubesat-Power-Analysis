import plotly.graph_objects as go
import numpy as np

def draw_interactive_cubesat(size_u=3, nadir_face='+Z'):
    width, depth, height = 1, 1, size_u

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

    faces = {
        '+Z': [[4, 5, 6], [4, 6, 7]],
        '-Z': [[0, 1, 2], [0, 2, 3]],
        '+X': [[1, 2, 6], [1, 6, 5]],
        '-X': [[0, 3, 7], [0, 7, 4]],
        '+Y': [[2, 3, 7], [2, 7, 6]],
        '-Y': [[0, 1, 5], [0, 5, 4]]
    }

    face_colors = {
        '+Z': 'lightblue', '-Z': 'lightgray',
        '+X': 'lightgreen', '-X': 'orange',
        '+Y': 'violet', '-Y': 'pink'
    }

    fig = go.Figure()

    for face, triangles in faces.items():
        i, j, k = zip(*triangles)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=face_colors[face] if face != nadir_face else 'red',
            opacity=0.5,
            name=face,
            showscale=False
        ))

    face_centers = {
        '+Z': [0.5, 0.5, height + 0.2],
        '-Z': [0.5, 0.5, -0.2],
        '+X': [1.2, 0.5, height / 2],
        '-X': [-0.2, 0.5, height / 2],
        '+Y': [0.5, 1.2, height / 2],
        '-Y': [0.5, -0.2, height / 2]
    }

    directions = {
        '+Z': [0, 0, -0.5], '-Z': [0, 0, 0.5],
        '+X': [-0.5, 0, 0], '-X': [0.5, 0, 0],
        '+Y': [0, -0.5, 0], '-Y': [0, 0.5, 0],
    }

    if nadir_face in face_centers:
        start = np.array(face_centers[nadir_face])
        end = start + np.array(directions[nadir_face])
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+text',
            line=dict(color='black', width=6),
            text=['', 'Nadir'],
            textposition='top center',
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig
