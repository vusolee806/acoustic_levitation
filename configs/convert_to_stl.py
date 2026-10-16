import vedo

# Point explicitly to the assets folder
bowl_mesh = vedo.Mesh("assets/bowl.obj")

# Save the STL in the assets folder as well
bowl_mesh.write("assets/bowl.stl")
