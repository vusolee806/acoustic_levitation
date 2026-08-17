import sys
import os
import numpy as np
import torch
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt  # Thêm thư viện vẽ đồ thị

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 1. IMPORT THE NEW WRAPPER FUNCTION AND PHASE GENERATOR

from src.acoustic_physics_pytorch import get_physics_outputs, generate_twin_trap_phases
def main():
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        return

    data = mujoco.MjData(model)

    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))
    
    # Initialize phases with geometric split for a static twin trap at the origin
    phases = np.zeros(num_transducers)

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        transducer_positions[i] = model.site_pos[site_id]
        
        # Apply pi-phase shift based on spatial X-coordinate
        if transducer_positions[i][0] > 0:
            phases[i] = np.pi
            
        quat = model.site_quat[site_id]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        transducer_zaxis[i] = mat.reshape(3, 3)[:, 2]

    ball_id = model.body("test_ball").id

    # 2. CONVERT TRANSDUCER ARRAYS TO PYTORCH TENSORS ONCE BEFORE THE LOOP
    trans_pos_tensor = torch.tensor(transducer_positions, dtype=torch.float32)
    trans_zaxis_tensor = torch.tensor(transducer_zaxis, dtype=torch.float32)
    phases_tensor = torch.tensor(phases, dtype=torch.float32)

    # --- KHAI BÁO CÁC MẢNG LƯU TRỮ DỮ LIỆU ĐỒ THỊ ---
    time_log = []
    fx_log = []
    fy_log = []
    fz_log = []
    z_pos_log = []

    print("Đang chạy mô phỏng... Tắt cửa sổ MuJoCo để xem đồ thị!")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            ball_pos = data.xpos[ball_id]
            ball_vel = data.cvel[ball_id][3:6] 
            
            # --- 1. DEFINE THE TARGET FOCAL POINT ---
            # In an RL environment, this coordinate would be the action output by your neural network.
            # For now, let's set a static target to catch the dropping ball at Z = 0.05 meters.
            target_focal_point = torch.tensor([[0.0, 0.0, 0.04]], dtype=torch.float32)
            
            # --- 2. GENERATE PHASES DYNAMICALLY ---
            # Calculate the exact phase delays needed to steer the twin trap to the target
            phases_tensor = generate_twin_trap_phases(target_focal_point, trans_pos_tensor)
            
            # --- 3. CALCULATE PHYSICS ---
            # Reshape ball pos to (1, 3) and call the physics wrapper using the NEW phases
            ball_pos_np = ball_pos.copy().reshape(1, 3).astype(np.float32)
            U_array, forces_array = get_physics_outputs(
                ball_pos_np, 
                trans_pos_tensor, 
                trans_zaxis_tensor, 
                phases_tensor  # <-- Using the dynamically generated phases here!
            )
            
            # Extract the force vector for the single ball
            arf_force = forces_array[0]
            arf_force = np.clip(arf_force, -1.0, 1.0)
            
            # # Add simple drag to simulate air resistance damping
            # damping_coeff = 0.00001
            # drag_force = -damping_coeff * ball_vel
            
            # Apply the acoustic and drag forces to the internal joint (qfrc)
            dof_start = model.body_dofadr[ball_id]
            data.qfrc_applied[dof_start : dof_start + 3] = arf_force #+ drag_force
            
            # --- LƯU DỮ LIỆU Ở MỖI BƯỚC MÔ PHỎNG ---
            time_log.append(data.time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            z_pos_log.append(ball_pos[2])
            
            mujoco.mj_step(model, data)
            viewer.sync()

    # =========================================================
    # VẼ ĐỒ THỊ SAU KHI MÔ PHỎNG KẾT THÚC
    # =========================================================
    print("Đã tắt mô phỏng. Đang tạo đồ thị...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Biểu đồ 1: Sự biến thiên của lực bức xạ âm
    ax1.plot(time_log, fz_log, label='F_z (Vertical Force)', color='blue')
    ax1.plot(time_log, fx_log, label='F_x (Lateral Force)', color='red', alpha=0.6)
    ax1.plot(time_log, fy_log, label='F_y (Lateral Force)', color='green', alpha=0.6)
    ax1.set_ylabel("Force (Newtons)")
    ax1.set_title("Acoustic Radiation Force over Time")
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ 2: Quỹ đạo độ cao của hạt (Xem hạt hội tụ vào điểm lõi)
    ax2.plot(time_log, z_pos_log, label='Z Position (Height)', color='purple')
    ax2.set_xlabel("Simulation Time (Seconds)")
    ax2.set_ylabel("Height (Meters)")
    ax2.set_title("Particle Trajectory (Settling into Trap)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
