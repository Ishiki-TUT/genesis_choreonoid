import numpy as np
import genesis as gs

# 1. initialise Genesis
gs.init(seed=0, backend=gs.gpu)  # use gs.cpu for CPU backend

# 2. create a scene
scene = gs.Scene(show_viewer=True)

# # scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# # 3. prepare a height map (here a simple bump for demo)
# hf = np.zeros((40, 40), dtype=np.int16)
# hf[10:30, 10:30] = 200 * np.hanning(20)[:, None] * np.hanning(20)[None, :]

# horizontal_scale = 0.25  # metres between grid points
# vertical_scale   = 0.005  # metres per height-field unit

# # 4. add the terrain entity
# scene.add_entity(
#     morph=gs.morphs.Terrain(
#         height_field=hf,
#         horizontal_scale=horizontal_scale,
#         vertical_scale=vertical_scale,
#     ),
# )

# scene = gs.Scene(show_viewer=True)

# terrain = scene.add_entity(
#     morph=gs.morphs.Terrain(
#         n_subterrains=(5, 2),
#         subterrain_size=(6.0, 6.0),
#         horizontal_scale=0.25,
#         vertical_scale=0.005,
#         subterrain_types=[
#             ["flat_terrain", "random_uniform_terrain"],
#             ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
#             ["wave_terrain", "pyramid_stairs_terrain"],
#             ["stairs_terrain", "stepping_stones_terrain"],
#             ["fractal_terrain","stepping_stones_terrain"],
#         ],
#     ),
# )

# scene.build(n_envs=5)  # you can still run many parallel envs

from genesis.utils.terrain import mesh_to_heightfield
import os

# path to your .obj / .glb / .stl terrain
mesh_path = os.path.join(gs.__path__[0], "assets", "meshes", "terrain_45.obj")

horizontal_scale = 2.0  # desired grid spacing (metres)
height, xs, ys = mesh_to_heightfield(mesh_path, spacing=horizontal_scale, oversample=3)

# shift the terrain so the centre of the mesh becomes (0,0)
translation = np.array([xs.min(), ys.min(), 0.0])

scene = gs.Scene(show_viewer=True)
scene.add_entity(
    morph=gs.morphs.Terrain(
        height_field=height,
        horizontal_scale=horizontal_scale,
        vertical_scale=1.0,
        pos=translation,  # optional world transform
    ),
)
scene.add_entity(gs.morphs.Sphere(pos=(10, 15, 10), radius=1))
scene.build()

# scene.build()

# run the sim so you can inspect the surface
for _ in range(10000):
    scene.step()


