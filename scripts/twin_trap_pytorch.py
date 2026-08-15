import sys
import os
import numpy as np
import torch
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt  # Thêm thư viện vẽ đồ thị

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 1. IMPORT THE NEW WRAPPER FUNCTION
from src.acoustic_physics_pytorch import get_physics_outputs 

def main():
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_array.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        return

    data = mujoco.MjData(model)

    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))
    
    phases = np.zeros(num_transducers)
    phases[:num_transducers//2] = np.pi 

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        transducer_positions[i] = model.site_pos[site_id]
        
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
            
            # 3. RESHAPE BALL POS TO (1, 3) AND CALL THE NEW PHYSICS WRAPPER
            ball_pos_np = ball_pos.copy().reshape(1, 3).astype(np.float32)
            U_array, forces_array = get_physics_outputs(
                ball_pos_np, 
                trans_pos_tensor, 
                trans_zaxis_tensor, 
                phases_tensor
            )
            
            # Extract the force vector for the single ball
            arf_force = forces_array[0]
            arf_force = np.clip(arf_force, -1.0, 1.0)
            
            damping_coeff = 0.000001
            drag_force = -damping_coeff * ball_vel
            
            # 1. Find where the ball's joint degrees of freedom (X, Y, Z) start in the engine
            dof_start = model.body_dofadr[ball_id]
            
            # 2. Apply the acoustic and drag forces to the internal joint (qfrc), NOT the external body (xfrc)
            # This leaves xfrc_applied completely free for your mouse interactions!
            data.qfrc_applied[dof_start : dof_start + 3] = arf_force + drag_force
            
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
