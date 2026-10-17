import numpy as np

import numpy as np

def create_points_hemisphere(N, radius=0.04, z_radius=0.015, z_center=0.0, max_attempts=2000, fixed_center_start=True):
    """
    Sinh N tọa độ hạt bên trong không gian làm việc hình elip dẹt (Focal Volume).
    """
    points = []
    attempts = 0
    # fixed at start on the Z axis
    if fixed_center_start and N > 0:
        points.append(np.array([0.0, 0.0, z_center]))
        
    while len(points) < N and attempts < max_attempts:
        # 1. Bốc ngẫu nhiên với giới hạn Z bị bóp hẹp lại
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = np.random.uniform(-z_radius, z_radius)
        point = np.array([x, y, z])
        
        # 2. Kiểm tra xem điểm có nằm trong khối Elip không gian làm việc không
        # Phương trình: (x^2 / r^2) + (y^2 / r^2) + (z^2 / z_r^2) <= 1
        if (x**2 / radius**2) + (y**2 / radius**2) + (z**2 / z_radius**2) > 1.0:
            attempts += 1
            continue
            
        point[2] += z_center
        
        # 3. Kiểm tra va chạm hạt
        if all(np.linalg.norm((point - p) / np.array([0.015, 0.015, 0.03])) > 1.0 for p in points):
            points.append(point)
        
        attempts += 1
        
        if attempts == max_attempts:
            raise RuntimeError(f"Lỗi: Không sinh đủ {N} điểm an toàn.")
    
    points = np.array(points)
    
    if not fixed_center_start:
        points = points[points[:, 0].argsort()]
    
    return points

#this part is use for extended version (multiparticle control)
def create_points_multistage_hemisphere(N, radius=0.04, z_center=0.12, max_attempts=2000, distance=0.02):
    """Sinh các cặp điểm (Start - Target) nằm gần nhau trong bán cầu."""
    # Bước 1: Sinh N điểm xuất phát (Start)
    points = create_points_hemisphere(N, radius, z_center, max_attempts)
    
    new_points = []
    
    # Bước 2: Sinh N điểm đích (Target) bám sát điểm xuất phát
    for point in points:
        attempts = 0
        current_distance = distance
        
        while attempts < max_attempts:
            # Bốc một điểm ngẫu nhiên nằm lân cận điểm xuất phát
            new_point = np.random.uniform(
                low=point - current_distance,
                high=point + current_distance
            )
            
            # LƯU Ý QUAN TRỌNG: Kiểm tra xem điểm đích mới này có bị lọt ra ngoài bát bán cầu không!
            # Vì point đã được cộng z_center, ta phải trừ đi để đưa về gốc tọa độ khi tính bán kính
            centered_new_point = new_point.copy()
            centered_new_point[2] -= z_center
            
            if np.linalg.norm(centered_new_point) <= radius:
                # Nếu không lọt ra ngoài bát, kiểm tra tiếp xem có đè lên các điểm đích khác không
                if all(np.linalg.norm((new_point - p) / np.array([0.015, 0.015, 0.03])) > 1.0 for p in new_points):
                    new_points.append(new_point)
                    break
            
            attempts += 1
            
            # Cơ chế nới lỏng nếu không tìm được chỗ trống
            if attempts == 100:
                current_distance += 0.01
            elif attempts == 200:
                current_distance += 0.01

        if attempts == max_attempts:
            print(new_points)
            raise RuntimeError("Lỗi: Không thể sinh điểm đích (Target) vì không gian quá chật.")
            
    new_points = np.array(new_points)
    
    return points, new_points
