import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt  # Thêm thư viện vẽ đồ thị

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.acoustic_physics import calculate_arf 

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
            
            arf_force = calculate_arf(ball_pos, transducer_positions, transducer_zaxis, phases)
            arf_force = np.clip(arf_force, -1.0, 1.0)
            
            damping_coeff = 0.05 
            drag_force = -damping_coeff * ball_vel
            
            data.xfrc_applied[ball_id] = 0.0 
            data.xfrc_applied[ball_id, :3] = arf_force + drag_force
            
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
