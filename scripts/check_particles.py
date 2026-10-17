import numpy as np

# Sử dụng đúng tên project HemisAcoustic của bạn
from HemisAcoustic.particles import particle_slim as Particle
from HemisAcoustic.particles import target_slim as Target

def main():
    print("=== 1. KHỞI TẠO HẠT VÀ MỤC TIÊU ===")
    # Hạt xuất hiện CHÍNH XÁC tại tâm hội tụ sóng âm của 79 transducer
    p = Particle(x=0.0, y=0.0, z=0.0) 
    
    # Mục tiêu nằm lệch sang phải 2cm
    t = Target(x=0.02, y=0.0, z=0.0)
    
    print(f"Vị trí ban đầu của hạt: ({p.x}, {p.y}, {p.z})")
    print(f"Vị trí mục tiêu:       ({t.x}, {t.y}, {t.z})")

    # Tính toán khoảng cách ban đầu
    initial_dist = np.linalg.norm([p.x - t.x, p.y - t.y, p.z - t.z])
    p.inital_distance = initial_dist
    p.last_timestep_dist = initial_dist
    print(f"Khoảng cách ban đầu:    {initial_dist:.4f} m\n")

    print("=== 2. MÔ PHỎNG ĐỘNG HỌC (RL cấp vận tốc) ===")
    # Giả sử mô hình RL xuất ra hành động yêu cầu hạt bay theo trục X với v = 0.1 m/s
    p.vX = 0.1
    p.vY = 0.0
    p.vZ = 0.0
    p.velocity = np.array([p.vX, p.vY, p.vZ])
    
    delta_t = 0.1  # Bước thời gian 100ms
    
    for step in range(1, 4):
        # Lưu lại trạng thái cũ
        p.last_position = [p.x, p.y, p.z]
        p.last_velocity = p.velocity.copy()

        # Cập nhật vị trí mới bằng phương trình Kinematic
        p.x += p.vX * delta_t
        p.y += p.vY * delta_t
        p.z += p.vZ * delta_t

        # Cập nhật khoảng cách tới đích
        current_dist = np.linalg.norm([p.x - t.x, p.y - t.y, p.z - t.z])
        p.last_timestep_dist = current_dist

        print(f"Step {step}:")
        print(f"  - Vị trí mới:  ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
        print(f"  - Tọa độ cũ:   ({p.last_position[0]:.4f}, {p.last_position[1]:.4f}, {p.last_position[2]:.4f})")
        print(f"  - Còn cách đích: {current_dist:.4f} m")
        
        if current_dist < 0.001:
            p.reached_target = True
            print("  -> Đã đến đích thành công!\n")
            break
        print()

    print("=== 3. KIỂM TRA HÀM RESET ===")
    p.reset()
    print(f"Sau khi gọi reset():")
    print(f"Vị trí hạt: ({p.x}, {p.y}, {p.z})")
    print(f"Vận tốc:    {p.velocity}")
    print(f"Đã đến đích: {p.reached_target}")

if __name__ == "__main__":
    main()
