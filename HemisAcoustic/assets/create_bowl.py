import numpy as np

def generate_bowl_mesh(filename="assets/bowl.obj", r_inner=0.086, thickness=0.004, theta_max_deg=80, rings=32, sectors=64):
    r_outer = r_inner + thickness
    theta_max = np.radians(theta_max_deg)
    
    thetas = np.linspace(0.05, theta_max, rings)
    phis = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    
    verts = []
    
    # Generate Inner Surface (concave inner face)
    for t in thetas:
        for p in phis:
            x = r_inner * np.sin(t) * np.cos(p)
            y = r_inner * np.sin(t) * np.sin(p)
            z = -r_inner * np.cos(t)  # Points downward
            verts.append((x, y, z))
            
    # Generate Outer Surface (convex outer backing)
    for t in thetas:
        for p in phis:
            x = r_outer * np.sin(t) * np.cos(p)
            y = r_outer * np.sin(t) * np.sin(p)
            z = -r_outer * np.cos(t)
            verts.append((x, y, z))
            
    faces = []
    num_ring_pts = sectors
    
    # Inner surface faces
    for i in range(rings - 1):
        for j in range(sectors):
            j_next = (j + 1) % sectors
            p1 = i * sectors + j + 1
            p2 = i * sectors + j_next + 1
            p3 = (i + 1) * sectors + j_next + 1
            p4 = (i + 1) * sectors + j + 1
            faces.append((p1, p3, p2))
            faces.append((p1, p4, p3))
            
    # Outer surface faces
    offset = rings * sectors
    for i in range(rings - 1):
        for j in range(sectors):
            j_next = (j + 1) % sectors
            p1 = offset + i * sectors + j + 1
            p2 = offset + i * sectors + j_next + 1
            p3 = offset + (i + 1) * sectors + j_next + 1
            p4 = offset + (i + 1) * sectors + j + 1
            faces.append((p1, p2, p3))
            faces.append((p1, p3, p4))
            
    # Rim top closing ring
    inner_edge_start = (rings - 1) * sectors
    outer_edge_start = offset + (rings - 1) * sectors
    for j in range(sectors):
        j_next = (j + 1) % sectors
        i1 = inner_edge_start + j + 1
        i2 = inner_edge_start + j_next + 1
        o1 = outer_edge_start + j + 1
        o2 = outer_edge_start + j_next + 1
        faces.append((i1, i2, o2))
        faces.append((i1, o2, o1))

    # Write out OBJ file
    with open(filename, "w") as f:
        f.write("# Acoustic Bowl Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)
    generate_bowl_mesh("assets/bowl.obj")
    print("Bowl mesh generated at assets/bowl.obj")
