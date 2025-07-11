import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def draw_cubesat(size_u=3, nadir_face='+Z', velocity_face='+X'):
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

    vertices = [a, b, c, d, e, f, g, h]

    faces = {
        '+Z': [e, f, g, h],
        '-Z': [a, b, c, d],
        '+X': [b, c, g, f],
        '-X': [a, d, h, e],
        '+Y': [c, d, h, g],
        '-Y': [a, b, f, e]
    }

    direction_vectors = {
        '+X': np.array([1, 0, 0]),
        '-X': np.array([-1, 0, 0]),
        '+Y': np.array([0, 1, 0]),
        '-Y': np.array([0, -1, 0]),
        '+Z': np.array([0, 0, 1]),
        '-Z': np.array([0, 0, -1]),
    }

    # All faces gray by default
    face_colors = {face: 'lightgray' for face in faces}
    if nadir_face in face_colors:
        face_colors[nadir_face] = 'red'
    if velocity_face in face_colors:
        face_colors[velocity_face] = 'blue'

    def face_center(face_verts):
        return np.mean(np.array(face_verts), axis=0)

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw cube faces
    for face, verts in faces.items():
        poly = Poly3DCollection([verts], facecolors=face_colors[face], edgecolor='black', alpha=0.8)
        ax.add_collection3d(poly)

    # Draw arrows for nadir and velocity only
    for face, label, color in [
        (nadir_face, 'Nadir', 'red'),
        (velocity_face, 'Velocity', 'blue'),
    ]:
        center = face_center(faces[face])
        direction = direction_vectors[face] / np.linalg.norm(direction_vectors[face])
        arrow_length = 0.7
        arrow_vector = direction * arrow_length

        ax.quiver(*center, *arrow_vector, color=color, arrow_length_ratio=0.2)
        ax.text(*(center + arrow_vector * 1.1), label, color=color, fontsize=10, ha='center')

    # Format plot
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, height + 0.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{size_u}U CubeSat with Orientation")

    ax.set_box_aspect([1, 1, height])
    plt.tight_layout()
    return fig
