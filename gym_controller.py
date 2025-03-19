from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import socket
import numpy as np
import math
import cv2
import struct
import threading
import time

def quaternion_to_euler(x, y, z, w):
    """
    quaternion (w, x, y, z)를 euler 각도로 변환
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def load_gripper(sim, gym):
    """
    로봇 그리퍼 관련 환경 변수 설정
    """
    asset_root = "assets"
    asset_path = "franka_description/robots/franka_hand_acronym_improved_collision_rotating.urdf"
    gripper_asset_options = gymapi.AssetOptions()
    gripper_asset_options.fix_base_link = True #! important
    gripper_asset_options.armature = 0.01
    gripper_asset_options.use_physx_armature = True
    gripper_asset_options.disable_gravity = True
    gripper_asset_options.linear_damping = 1.0
    gripper_asset_options.max_linear_velocity = 1.0
    gripper_asset_options.angular_damping = 5.0
    gripper_asset_options.max_angular_velocity = 2*math.pi
    gripper_asset_options.collapse_fixed_joints = False
    gripper_asset_options.enable_gyroscopic_forces = True
    gripper_asset_options.thickness = 0.0
    gripper_asset_options.density = 1000
    gripper_asset_options.override_com = True
    gripper_asset_options.override_inertia = True
    gripper_asset_options.vhacd_enabled = True
    gripper_asset_options.vhacd_params = gymapi.VhacdParams()
    gripper_asset_options.vhacd_params.resolution = 1000000
    
    asset_gripper = gym.load_asset(sim, asset_root, asset_path, gripper_asset_options)
    return asset_gripper
    
    
# 1. Isaac Gym 초기화 및 시뮬레이션 설정
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z  # Z 축이 위쪽
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0  # 60Hz 주기

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("시뮬레이션 생성 실패")
    exit()

# 바닥면 추가
plane_params = gymapi.PlaneParams()
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 2.0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)


# 환경 생성 (하나의 환경)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
initial_tf = gymapi.Transform()
initial_tf.p = gymapi.Vec3(0, 0, 0.2)


# 그리퍼 로드 및 actor 생성 및 actor handler 설정
franka_gripper = load_gripper(sim, gym)
gripper_actor = gym.create_actor(env, franka_gripper, initial_tf, "gripper", 0, 1)
gripper_handle = gym.find_actor_handle(env, "gripper")
x_trans_handle = gym.find_actor_dof_handle(env, gripper_actor, 'hand_joint_x')
y_trans_handle = gym.find_actor_dof_handle(env, gripper_actor, 'hand_joint_y')
z_trans_handle = gym.find_actor_dof_handle(env, gripper_actor, 'hand_joint')

x_rot_handle = gym.find_actor_dof_handle(env, gripper_actor, 'hand_rotating_x')
y_rot_handle = gym.find_actor_dof_handle(env, gripper_actor, 'hand_rotating_y')
z_rot_handle = gym.find_actor_dof_handle(env, gripper_actor, 'hand_rotating')

gripper_props = gym.get_actor_dof_properties(env, gripper_actor)
gripper_props['driveMode'].fill(gymapi.DOF_MODE_POS)
gripper_props['stiffness'].fill(1300.0)
gripper_props['damping'].fill(200.0)
gripper_props["velocity"].fill(1)
gym.set_actor_dof_properties(env, gripper_actor, gripper_props)

# 전역 변수에 연결 객체 저장
ctrl_conn = None
img_conn = None

# 제어용 TCP 서버 (포트 5000)
ctrl_tcp_ip = "0.0.0.0"
ctrl_tcp_port = 5000
ctrl_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ctrl_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ctrl_server_socket.bind((ctrl_tcp_ip, ctrl_tcp_port))
ctrl_server_socket.listen(1)
print("제어용 TCP 연결 대기 중: {}:{}".format(ctrl_tcp_ip, ctrl_tcp_port))

# 이미지용 TCP 서버 (포트 4000)
img_tcp_ip = "0.0.0.0"
img_tcp_port = 4000
img_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
img_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
img_server_socket.bind((img_tcp_ip, img_tcp_port))
img_server_socket.listen(1)
print("이미지용 TCP 연결 대기 중: {}:{}".format(img_tcp_ip, img_tcp_port))

def accept_ctrl_connection():
    global ctrl_conn
    ctrl_conn, addr = ctrl_server_socket.accept()
    print("제어용 TCP 연결됨:", addr)

def accept_img_connection():
    global img_conn
    img_conn, addr = img_server_socket.accept()
    print("이미지용 TCP 연결됨:", addr)

# 별도 스레드로 연결 수락
ctrl_thread = threading.Thread(target=accept_ctrl_connection)
ctrl_thread.daemon = True
ctrl_thread.start()

img_thread = threading.Thread(target=accept_img_connection)
img_thread.daemon = True
img_thread.start()

# 두 연결이 모두 수락될 때까지 대기
while ctrl_conn is None or img_conn is None:
    time.sleep(0.1)  
    
# viewer용 카메라 설정
cam_props = gymapi.CameraProperties()
cam_props.use_collision_geometry = True
viewer = gym.create_viewer(sim, cam_props)
cam_pos = gymapi.Vec3(1, 1, 1)
can_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, env, cam_pos, can_target)


# 이미지 획득용 카메라 설정
img_cam_props = gymapi.CameraProperties()
img_cam_props.width = 1280
img_cam_props.height = 720
img_cam_props.enable_tensors = True
camera_handle = gym.create_camera_sensor(env, img_cam_props)

#카메라 위치 설정
cam_local_transform = gymapi.Transform()
cam_local_transform.p = gymapi.Vec3(1, 1, 1)
cam_local_transform.r = gymapi.Quat.from_euler_zyx(np.deg2rad(180), np.deg2rad(135), np.deg2rad(45))
gym.attach_camera_to_body(camera_handle, env, gripper_handle, cam_local_transform, gymapi.FOLLOW_POSITION)


# 4. 메인 시뮬레이션 루프
buffer = ""
while True:
    # TCP 데이터 수신 (데이터 형식: "px,py,pz, roll, pitch, yaw") 
    data = ctrl_conn.recv(1024)
    if not data:
        break  # 연결 종료 시 루프 탈출
    buffer += data.decode('utf-8')
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        try:
            values = [float(x) for x in line.split(',')]
            if len(values) == 6:
                px, py, pz, roll, pitch, yaw = values
            # numpy 배열을 생성 (위치 3개, 회전 4개)
            new_state = np.array([px, py, pz, roll, pitch, yaw], dtype=np.float32)
            
            # change degree to radian
            roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)
            
            # actor의 루트 상태 업데이트 (set_actor_root_state 사용)
            gym.set_dof_target_position(env, x_trans_handle, pz)
            gym.set_dof_target_position(env, y_trans_handle, px)
            gym.set_dof_target_position(env, z_trans_handle, py)
            gym.set_dof_target_position(env, x_rot_handle, -roll)
            gym.set_dof_target_position(env, y_rot_handle, -pitch)
            gym.set_dof_target_position(env, z_rot_handle, yaw)
            
        except Exception as e:
            print("데이터 파싱 에러:", e)

    # 시뮬레이션 진행
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    gym.render_all_camera_sensors(sim)
    
    # 이미지 캡처 및 전송
    try:
        img_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
        torch_img = gymtorch.wrap_tensor(img_tensor)
        img = torch_img.cpu().numpy()
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imshow("image", bgr_img)
        cv2.waitKey(1)
        ret, jpeg = cv2.imencode('.jpg', bgr_img)
        if ret:
            jpeg_bytes = jpeg.tobytes()
            # 전송할 데이터 길이를 4바이트 빅 엔디안 정수로 패킹 후 이미지 데이터 전송
            length = struct.pack('!I', len(jpeg_bytes))
            try:
                img_conn.sendall(length + jpeg_bytes)
            except Exception as e:
                print("이미지 전송 오류:", e)
    except Exception as e:
        print("이미지 캡처/인코딩 오류:", e)

# 연결 종료
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
