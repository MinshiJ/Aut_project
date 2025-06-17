import mujoco
import numpy as np
import time
from mujoco.viewer import launch
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import ttk, messagebox
import ctypes

# 设置DPI感知
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

"""
                                    GLOBALS
"""

PI = np.pi

class ParameterWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("模拟参数设置")
        self.root.geometry("500x800")
        
        # 设置窗口最小大小
        self.root.minsize(500, 800)
        
        # 设置窗口样式
        style = ttk.Style()
        style.configure('TLabelframe', font=('Microsoft YaHei UI', 10))
        style.configure('TLabelframe.Label', font=('Microsoft YaHei UI', 10))
        style.configure('TRadiobutton', font=('Microsoft YaHei UI', 10))
        style.configure('TLabel', font=('Microsoft YaHei UI', 10))
        style.configure('TButton', font=('Microsoft YaHei UI', 10))
        
        # 创建参数变量
        self.assembly_task = tk.StringVar(value="RJ45")
        self.compliance_stiffness = tk.StringVar(value="50.0")
        self.assembly_stiffness = tk.StringVar(value="2.0")
        self.z_stiffness = tk.StringVar(value="100")
        self.rotx_stiffness = tk.StringVar(value="100.0")
        self.roty_stiffness = tk.StringVar(value="100.0")
        self.rotz_stiffness = tk.StringVar(value="1.0")
        self.starting_pos_shift = tk.StringVar(value="0.0")
        self.simulation_repeats = tk.StringVar(value="1")
        
        # 设置字体
        self.default_font = ('Microsoft YaHei UI', 10)
        self.root.option_add('*Font', self.default_font)
        
        self.create_widgets()
        
    def create_widgets(self):
        # 装配任务选择
        task_frame = ttk.LabelFrame(self.root, text="选择装配任务", padding=10)
        task_frame.pack(fill="x", padx=10, pady=5)
        
        tasks = ["RJ45", "USB", "KET8", "KET12"]
        for task in tasks:
            ttk.Radiobutton(task_frame, text=task, variable=self.assembly_task, 
                           value=task).pack(anchor="w")
        
        # 刚度参数
        stiffness_frame = ttk.LabelFrame(self.root, text="刚度参数设置", padding=10)
        stiffness_frame.pack(fill="x", padx=10, pady=5)
        
        # 创建刚度参数输入框
        stiffness_params = [
            ("顺应方向刚度 (N/mm):", self.compliance_stiffness),
            ("装配方向刚度 (N/mm):", self.assembly_stiffness),
            ("Z方向刚度 (N/mm):", self.z_stiffness),
            ("X旋转刚度 (Nm/rad):", self.rotx_stiffness),
            ("Y旋转刚度 (Nm/rad):", self.roty_stiffness),
            ("Z旋转刚度 (Nm/rad):", self.rotz_stiffness)
        ]
        
        for label, var in stiffness_params:
            frame = ttk.Frame(stiffness_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=20).pack(side="left")
            ttk.Entry(frame, textvariable=var, width=10).pack(side="right")
        
        # 起始位置偏移
        pos_frame = ttk.LabelFrame(self.root, text="起始位置设置", padding=10)
        pos_frame.pack(fill="x", padx=10, pady=5)
        
        frame = ttk.Frame(pos_frame)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text="起始位置偏移 (mm):", width=20).pack(side="left")
        
        # 创建按钮框架
        button_frame = ttk.Frame(frame)
        button_frame.pack(side="right")
        
        # 减少按钮
        decrease_btn = ttk.Button(button_frame, text="-", width=3, 
                                 command=self.decrease_starting_pos)
        decrease_btn.pack(side="left", padx=(0, 2))
        
        # 输入框
        ttk.Entry(button_frame, textvariable=self.starting_pos_shift, width=10).pack(side="left", padx=2)
        
        # 增加按钮
        increase_btn = ttk.Button(button_frame, text="+", width=3, 
                                 command=self.increase_starting_pos)
        increase_btn.pack(side="left", padx=(2, 0))
        
        # 模拟重复次数
        repeat_frame = ttk.LabelFrame(self.root, text="模拟设置", padding=10)
        repeat_frame.pack(fill="x", padx=10, pady=5)
        
        repeat_inner_frame = ttk.Frame(repeat_frame)
        repeat_inner_frame.pack(fill="x", pady=2)
        ttk.Label(repeat_inner_frame, text="模拟重复次数:", width=20).pack(side="left")
        ttk.Entry(repeat_inner_frame, textvariable=self.simulation_repeats, width=10).pack(side="right")
        
        # 运行按钮
        self.run_button = ttk.Button(self.root, text="运行模拟", command=self.run_simulation)
        self.run_button.pack(pady=20)
        
    def get_parameters(self):
        return {
            'assembly_task': self.assembly_task.get(),
            'compliance_stiffness': float(self.compliance_stiffness.get()),
            'assembly_stiffness': float(self.assembly_stiffness.get()),
            'z_stiffness': float(self.z_stiffness.get()),
            'rotx_stiffness': float(self.rotx_stiffness.get()),
            'roty_stiffness': float(self.roty_stiffness.get()),
            'rotz_stiffness': float(self.rotz_stiffness.get()),
            'starting_pos_shift': float(self.starting_pos_shift.get()) / 1000.0,  # 转换为米
            'simulation_repeats': int(self.simulation_repeats.get())
        }
    
    def increase_starting_pos(self):
        """增加起始位置偏移0.1mm"""
        try:
            current_value = float(self.starting_pos_shift.get())
            new_value = current_value + 0.1
            self.starting_pos_shift.set(f"{new_value:.1f}")
        except ValueError:
            # 如果输入不是有效数字，重置为0.1
            self.starting_pos_shift.set("0.1")
    
    def decrease_starting_pos(self):
        """减少起始位置偏移0.1mm"""
        try:
            current_value = float(self.starting_pos_shift.get())
            new_value = current_value - 0.1
            self.starting_pos_shift.set(f"{new_value:.1f}")
        except ValueError:
            # 如果输入不是有效数字，重置为-0.1
            self.starting_pos_shift.set("-0.1")
    
    def run_simulation(self):
        # 获取当前参数并运行模拟
        params = self.get_parameters()
        run_simulation_with_parameters(params)
    
    def show(self):
        self.root.mainloop()

def calculate_ik(model, data, goal_pos, goal_euler, init_q, 
                 body_id=None, step_size=1e-4, tol=1e-4, 
                 alpha=0.1, max_iters=1e5):
    # Set current position to inital
    data.qpos[28:34] = init_q
    goal_quat = R.from_euler('xyz', goal_euler,
                             degrees=True).as_quat(scalar_first=True)
    # Calculate model
    mujoco.mj_forward(model, data)

    # Get current position and orientation of body_id
    if body_id is None:
        body_id = model.body("TCP").id
    current_pos = data.body(body_id).xpos.copy()
    current_quat = data.body(body_id).xquat.copy()

    # Calculate position and orientation error
    pos_error = np.subtract(goal_pos, current_pos)
    quat_error = quaternion_error(goal_quat, current_quat)

    iters = 0
    while np.linalg.norm(pos_error) >= tol or np.linalg.norm(quat_error) \
        >= tol and iters < max_iters:
        # Calculate Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, current_pos, body_id)

        # Stack the Jacobians
        jac = np.vstack((jacp, jacr))
        jac = jac[:, 24:30]

        # Stack the errors
        error = np.hstack((pos_error, quat_error))

        # Calculate gradient for robot joints only
        grad = alpha * jac.T @ error
        grad = normalize_gradient(grad)

        # Update only the specific joints' positions
        data.qpos[28:34] += step_size * grad

        # Check joint limits for the specific joints
        for i, j in zip([28, 29, 30, 31, 32, 33], [4, 5, 6, 7, 8, 9]):           
            data.qpos[i] = np.clip(data.qpos[i], model.jnt_range[j, 0], 
                                   model.jnt_range[j, 1])

        # Compute forward kinematics
        mujoco.mj_forward(model, data)

        # Calculate new error
        current_pos = data.body(body_id).xpos.copy()
        current_quat = data.body(body_id).xquat.copy()
        
        pos_error = np.subtract(goal_pos, current_pos)
        quat_error = quaternion_error(goal_quat, current_quat)
  
        iters += 1

    return data.qpos.copy()

def quaternion_error(target, current):
    """Compute the quaternion error."""
    return np.array([target[0]*current[1] - target[1]*current[0] -
                     target[2]*current[3] + target[3]*current[2],
                     target[0]*current[2] + target[1]*current[3] -
                     target[2]*current[0] - target[3]*current[1],
                     target[0]*current[3] - target[1]*current[2] +
                     target[2]*current[1] - target[3]*current[0]])

def normalize_gradient(grad):
    """Normalize the gradient to ensure stable steps."""
    norm = np.linalg.norm(grad)
    return grad / norm if norm > 0 else grad

def gripper_state(position=0.01, state="open"):
    # Check the gripper state (open, closed)
    global data
    if state == "open":
        data.ctrl[6] = -position
        data.ctrl[7] =  position
    if state == "closed":
        data.ctrl[6] = -position
        data.ctrl[7] =  position

def simulate_movement_duration(duration=15, frames_rendered=50):
    frames_rendered = frames_rendered   
    frame_count = 0
   
    time_start = time.monotonic()
 
    while (time.monotonic() <= time_start + duration):
        # 检查查看器是否仍在运行
        if not viewer.is_running():
            print("查看器已关闭，停止模拟")
            return False
        
        mujoco.mj_step(model, data)
        if frame_count % frames_rendered == 0:
            viewer.sync()
        frame_count += 1
    
    return True

def simulate_movement_tol(target_site_name="", tol=1e-3, frames_rendered=50):
    frames_rendered = frames_rendered   
    frame_count = 0
  
    mujoco.mj_fwdPosition(model, data)

    end_effector = data.body("TCP").xpos
    target_site = data.site(target_site_name).xpos
 
    while any(np.abs(end_effector - target_site) >= tol):
        # 检查查看器是否仍在运行
        if not viewer.is_running():
            print("查看器已关闭，停止模拟")
            return False
        
        mujoco.mj_step(model, data)
        if frame_count % frames_rendered == 0:
            viewer.sync()
        frame_count += 1
        
        # 更新末端执行器位置
        end_effector = data.body("TCP").xpos
    
    return True

def check_viewer_and_simulate_tol(target_site_name="", tol=1e-3):
    """检查查看器状态并执行位置模拟，如果查看器关闭则返回 False"""
    global simulation_continue
    if not viewer.is_running():
        print("查看器已关闭，取消模拟")
        simulation_continue = False
        return False
    result = simulate_movement_tol(target_site_name, tol)
    if not result:
        simulation_continue = False
    return result

def check_viewer_and_simulate_duration(duration=15):
    """检查查看器状态并执行时间模拟，如果查看器关闭则返回 False"""
    global simulation_continue
    if not viewer.is_running():
        print("查看器已关闭，取消模拟")
        simulation_continue = False
        return False
    result = simulate_movement_duration(duration)
    if not result:
        simulation_continue = False
    return result

def safe_simulate_tol(target_site_name="", tol=1e-3):
    """安全执行位置模拟，自动检查查看器状态"""
    global simulation_continue
    if not simulation_continue or not viewer.is_running():
        simulation_continue = False
        return False
    return simulate_movement_tol(target_site_name, tol)

def safe_simulate_duration(duration=15):
    """安全执行时间模拟，自动检查查看器状态"""
    global simulation_continue
    if not simulation_continue or not viewer.is_running():
        simulation_continue = False
        return False
    return simulate_movement_duration(duration)
 
def run_simulation_with_parameters(params):
    global model, data, viewer, simulation_continue
    
    # 获取重复次数
    repeat_count = params['simulation_repeats']
      # 重复运行模拟
    for run_num in range(1, repeat_count + 1):
        print(f"\n=== 开始第 {run_num}/{repeat_count} 次模拟 ===")
        
        # 设置场景和装配任务
        if params['assembly_task'] == "RJ45":
            xml = "scene_chamf.xml"
            assembly_task = "RJ45"
        else:
            xml = "scene_15.xml"
            assembly_task = params['assembly_task']

        # 加载模型
        model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        cam = mujoco.MjvCamera()                        
        option = mujoco.MjvOption()     
        options = model.opt

        # 获取关节ID
        left_compliance_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'compliance_dir_left')
        left_assembly_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'assembly_dir_left')
        left_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'z_dir_left')
        left_rotx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rot_x_left')
        left_roty_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rot_y_left')
        left_rotz_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rot_z_left')
        
        right_compliance_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'compliance_dir_right')
        right_assembly_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'assembly_dir_right')
        right_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'z_dir_right')
        right_rotx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rot_x_right')
        right_roty_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rot_y_right')
        right_rotz_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rot_z_right')
        
        # 设置刚度值
        # 左侧
        model.jnt_stiffness[left_compliance_id] = params['compliance_stiffness'] * 1000
        model.jnt_stiffness[left_assembly_id] = params['assembly_stiffness'] * 1000
        model.jnt_stiffness[left_z_id] = params['z_stiffness'] * 1000
        model.jnt_stiffness[left_rotx_id] = params['rotx_stiffness']
        model.jnt_stiffness[left_roty_id] = params['roty_stiffness']
        model.jnt_stiffness[left_rotz_id] = params['rotz_stiffness']
        
        # 右侧
        model.jnt_stiffness[right_compliance_id] = params['compliance_stiffness'] * 1000
        model.jnt_stiffness[right_assembly_id] = params['assembly_stiffness'] * 1000
        model.jnt_stiffness[right_z_id] = params['z_stiffness'] * 1000
        model.jnt_stiffness[right_rotx_id] = params['rotx_stiffness']
        model.jnt_stiffness[right_roty_id] = params['roty_stiffness']
        model.jnt_stiffness[right_rotz_id] = params['rotz_stiffness']        # 设置起始位置偏移
        startingpos_shift = params['starting_pos_shift']

        # 启动查看器
        viewer = mujoco.viewer.launch_passive(model, data)
        
        # 立即检查查看器是否成功启动
        if not viewer.is_running():
            print("查看器启动失败或被立即关闭，停止所有模拟")
            break
        
        # 显示接触力和接触点
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True  
        viewer.opt.label = viewer.opt.label | mujoco.mjtLabel.mjLABEL_CONTACTFORCE  
     
        robotjoints = [28, 29, 30, 31, 32, 33]
        pos_home = np.array([3*np.pi/2, -np.pi/2, -np.pi/2, 3*np.pi/2, np.pi/2, 0, -0.01, 0.01])

        # 模拟是否应该继续的标志
        simulation_continue = True

        if viewer.is_running() and simulation_continue:
            if xml == "scene_chamf.xml":
                ### Camerapos.
                viewer.cam.azimuth = -90
                viewer.cam.elevation = -10
                viewer.cam.distance = 0.2
                viewer.cam.lookat =np.array([-0.6413, 0.54745, 0.09])
                
                
                
                ### Calculating joint positions
                
                # start in home position
                data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                
                # RJ45 above
                goal = [-0.6413, 0.54745, 0.15]
                orientation = [0, 0, -90]
                init = pos_home[:6]        
                pos_RJ45_above = calculate_ik(model, data, goal, orientation, 
                                            init)[robotjoints]
                print("Position RJ45 above calculated.")

                # RJ45 gripping
                goal = [-0.6413, 0.54745, 0.084]
                orientation = [0, 0, -90]
                init = pos_RJ45_above        
                pos_RJ45_gripping = calculate_ik(model, data, goal, orientation, 
                                                init)[robotjoints]
                print("Position RJ45 above calculated.")

                # RJ45 above gripped
                goal = [-0.6413, 0.54745, 0.12]
                orientation = [0, 0, -90]
                init = pos_RJ45_gripping        
                pos_RJ45_above_gripped = calculate_ik(model, data, goal, orientation, 
                                                    init)[robotjoints]
                print("Position RJ45 above calculated.")
            
                ### SEARCH STRATEGY
                
                # Starting position for search strategy
                # Depends also on startingpos_shift
                site_name = "RJ45_searchAngled"
                data.site(site_name).xpos += np.array([startingpos_shift, 0, 0])
                model.site(site_name).pos = data.site(site_name).xpos
                startpos = data.site(site_name).xpos.copy()

                # RJ45 angled
                orientation = [10, 0, -90]
                init = pos_RJ45_gripping        
                pos_RJ45_angled = calculate_ik(model, data, startpos, orientation, 
                                            init)[robotjoints]
                print("Position RJ45 above calculated.")
            

                # RJ45 touch Z
                startpos += [0, 0, -0.0041]
                orientation = [10, 0, -90]
                init = pos_RJ45_angled        
                pos_RJ45_touchZ = calculate_ik(model, data, startpos, orientation, 
                                            init)[robotjoints]
                print("Position RJ45 touch Z calculated.")


                # RJ45 touch back
                startpos += [0, 0.003, 0]
                orientation = [10, 0, -90]
                init = pos_RJ45_touchZ        
                pos_RJ45_touchback = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                print("Position RJ45 touchback calculated.")


                # RJ45 touch front
                startpos += [0, -0.003, -0.0002]
                orientation = [10, 0, -90]
                init = pos_RJ45_touchback        
                pos_RJ45_touchfront = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                print("Position RJ45 touchfront calculated.")


                # RJ45 touch side
                startpos += [-0.004, 0, 0]
                orientation = [10, 0, -90]
                init = pos_RJ45_touchfront        
                pos_RJ45_touchside = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                print("Position RJ45 touchside calculated.")


                # RJ45 assembly
                startpos += [0, 0, -0.014]
                orientation = [10, 0, -90]
                init = pos_RJ45_touchside        
                pos_RJ45_assembly = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                print("RJ45 assembly position calculated.")
                
                
                # RJ45 assembled
                startpos += [0.0095, 0, 0.005]
                orientation = [0, 0, -90]
                init = pos_RJ45_assembly        
                pos_RJ45_assembled = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                print("RJ45 assembled position calculated.")


                ### SIMULATION 

                # timestep
                options.timestep = 5e-5 

                # start simulation in home position
                data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home

                  ### RJ45 Above
                data.ctrl[:6] = pos_RJ45_above
                gripper_state()
                if not check_viewer_and_simulate_tol(target_site_name="RJ45_above"):
                    simulation_continue = False
                    break
                print("Position RJ45 above reached.")
                
                
                ### RJ45 gripping
                data.ctrl[:6] = pos_RJ45_gripping
                gripper_state()
                if not check_viewer_and_simulate_tol(target_site_name="RJ45_gripping"):
                    simulation_continue = False
                    break
                print("Position RJ45 gripping reached.")
                  # close gripper
                gripper_state(state="closed", position=0.003)
                if not check_viewer_and_simulate_duration(5):
                    simulation_continue = False
                    break
                print("Position RJ45 gripped reached.")
                
                
                ### RJ45 Above gripped
                data.ctrl[:6] = pos_RJ45_above_gripped
                gripper_state(state="closed", position=0.003)
                if not check_viewer_and_simulate_tol(target_site_name="RJ45_above_gripped"):
                    simulation_continue = False
                    break
                print("Position RJ45 above gripped reached.")
                
                
                ### SEARCH STRATEGY
                  ### RJ45 angled
                data.ctrl[:6] = pos_RJ45_angled
                gripper_state(state="closed", position=0.003)
                if not safe_simulate_tol(target_site_name="RJ45_searchAngled", tol=1e-4):
                    simulation_continue = False
                    break
                print("Position RJ45 angled reached.")
                
                
                # RJ45 Touch Z
                data.ctrl[:6] = pos_RJ45_touchZ
                gripper_state(state="closed", position=0.003)
                if not safe_simulate_duration(10):
                    simulation_continue = False
                    break
                print("Position RJ45 touchZ reached.")
                
                
                ### Touch Back
                data.ctrl[:6] = pos_RJ45_touchback
                gripper_state(state="closed", position=0.003)
                if not safe_simulate_duration(5):
                    simulation_continue = False
                    break
                print("Position RJ45 touch back reached.")
                

                ### Touch Front
                data.ctrl[:6] = pos_RJ45_touchfront
                gripper_state(state="closed", position=0.003)
                if not safe_simulate_duration(5):
                    simulation_continue = False
                    break
                print("Position RJ45 touch front reached.")


                ### Touch Side
                data.ctrl[:6] = pos_RJ45_touchside
                gripper_state(state="closed", position=0.003)
                if not safe_simulate_duration(10):
                    simulation_continue = False
                    break
                print("Position RJ45 touch side reached.")

                gripper_axis_right_id = mujoco.mj_name2id(model, 
                                                        mujoco.mjtObj.mjOBJ_JOINT, 
                                                        'gripper_axis_right_joint')
                dof_right_id = model.jnt_dofadr[gripper_axis_right_id]
                model.dof_armature[dof_right_id] = 1e7    
                
                
                gripper_axis_left_id = mujoco.mj_name2id(model, 
                                                        mujoco.mjtObj.mjOBJ_JOINT, 
                                                        'gripper_axis_left_joint')
                dof_left_id = model.jnt_dofadr[gripper_axis_left_id]
                model.dof_armature[dof_left_id] = 1e7                ### Assembly
                data.ctrl[:6] = pos_RJ45_assembly
                gripper_state(state="closed", position=0.003)
                if not safe_simulate_duration(10):
                    simulation_continue = False
                    break
                print("RJ45 assembled.")

                model.dof_armature[dof_right_id] = 0    
                model.dof_armature[dof_left_id] = 0     
                

                ### Assembled
                data.ctrl[:6] = pos_RJ45_assembled
                gripper_state()
                if not safe_simulate_duration(10):
                    simulation_continue = False
                    break
                print("RJ45 assembly complete.")
            
            elif xml == "scene_15.xml":
                if assembly_task == "USB":
                    ### Camerapos.
                    viewer.cam.azimuth = 0
                    viewer.cam.elevation = -10
                    viewer.cam.distance = 0.15
                    viewer.cam.lookat =np.array([-0.3414, 0.4722, 0.09])

                    
            
                    ### Calculating joint positions
                    
                    # start in home position
                    data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                    
                    # USB above
                    goal = [-0.3414, 0.4722, 0.15]
                    orientation = [0, 0, 0]
                    init = pos_home[:6]        
                    pos_USB_above = calculate_ik(model, data, goal, orientation, 
                                                init)[robotjoints]
                    print("Position USB above calculated.")
                    
                    
                    # USB gripping
                    goal = [-0.3414, 0.4722, 0.097]
                    orientation = [0, 0, 0]
                    init = pos_USB_above       
                    pos_USB_gripping = calculate_ik(model, data, goal, orientation, 
                                                    init)[robotjoints]
                    print("Position USB gripping calculated.")
                    
                    # USB above gripped
                    goal = [-0.3414, 0.4722, 0.15]
                    orientation = [0, 0, 0]
                    init = pos_USB_gripping
                    pos_USB_above_gripped = calculate_ik(model, data, goal, orientation, 
                                                        init)[robotjoints]
                    print("Position USB above gripped calculated.")
                    
                    
                    ### SEARCH STRATEGY
                    
                    # Starting position for search strategy
                    # Depends also on startingpos_shift
                    site_name = "USB_searchAngled"
                    data.site(site_name).xpos += np.array([0, startingpos_shift, 0])
                    model.site(site_name).pos = data.site(site_name).xpos
                    startpos = data.site(site_name).xpos.copy()

                    
                    # USB angled 
                    orientation = [10, 0, 0]
                    init = pos_USB_above_gripped 
                    pos_USB_angled= calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                    print("Position USB angled calculated.")
                    
                    
                    # USB touch Z
                    startpos += [-0.0026, 0, -0.0098]
                    orientation = [10, 0, 0]
                    init = pos_USB_angled 
                    pos_USB_touchZ = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                    print("Position USB touchZ calculated.")

                    
                    # USB touch back
                    startpos += [0.005, 0, 0]
                    orientation = [10, 0, 0]
                    init = pos_USB_touchZ 
                    pos_USB_touchback = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("Position USB touchback calculated.")
                    
                    
                    # USB touch front
                    startpos += [-0.005, 0, 0]
                    orientation = [10, 0, 0]
                    init = pos_USB_touchback 
                    pos_USB_touchfront = calculate_ik(model, data, startpos, orientation,
                                                    init)[robotjoints]
                    print("Position USB touchfront calculated.")
                    
                    
                    # USB touch side
                    startpos += [0, -0.01, 0]
                    orientation = [10, 0, 0]
                    init = pos_USB_touchfront 
                    pos_USB_touchside = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("Position USB touchside calculated.")
                    
                    
                    # USB assembly
                    startpos += [0, 0, -0.02]
                    orientation = [10, 0, 0]
                    init = pos_USB_touchside
                    pos_USB_assembly = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("USB assembly position calculated.")
                    
                    # RJ45 assembled
                    startpos += [0, 0.014, 0.02]
                    orientation = [0, 0, 0]
                    init = pos_USB_assembly        
                    pos_USB_assembled = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("USB assembled position calculated.")

                    
                    
                    ### SIMULATION 
                    
                    # timestep
                    options.timestep = 5e-5 

                    # start simulation in home position
                    data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                    print("Home position reached.")

                      ### USB Above
                    data.ctrl[:6] = pos_USB_above
                    gripper_state()
                    if not safe_simulate_tol(target_site_name="USB_above"):
                        simulation_continue = False
                        break
                    print("Position USB Above reached.")



                    ### USB gripping 
                    data.ctrl[:6] = pos_USB_gripping
                    gripper_state()
                    if not safe_simulate_tol(target_site_name="USB_gripping"):
                        simulation_continue = False
                        break
                    print("Position USB gripping reached.")
                    
                    # close gripper
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position USB gripping reached and gripper closed.")
                    
                    
                    ### USB Above gripped
                    data.ctrl[:6] = pos_USB_above_gripped
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_tol(target_site_name="USB_above_gripped"):
                        simulation_continue = False
                        break
                    print("Position USB above gripped reached.")
                    
                    
                    ### SEARCH STRATEGY
                    
                    ### Angled
                    data.ctrl[:6] = pos_USB_angled
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_tol(target_site_name="USB_searchAngled", tol=1e-4):
                        simulation_continue = False
                        break
                    print("Position USB angled reached.")
                    
                    
                    ### Touch Z
                    data.ctrl[:6] = pos_USB_touchZ
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("Position USB touchZ reached.")


                    
                    ### Touch Back
                    data.ctrl[:6] = pos_USB_touchback
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position USB touch back reached.")
                    

                    ### Touch Front
                    data.ctrl[:6] = pos_USB_touchfront
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position USB touch front reached.")


                    ### Touch Side
                    data.ctrl[:6] = pos_USB_touchside
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("Position USB touch side reached.")

                    gripper_axis_right_id = mujoco.mj_name2id(model, 
                                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                                            'gripper_axis_right_joint')
                    dof_right_id = model.jnt_dofadr[gripper_axis_right_id]
                    model.dof_armature[dof_right_id] = 1e7    
                    
                    
                    gripper_axis_left_id = mujoco.mj_name2id(model, 
                                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                                            'gripper_axis_left_joint')
                    dof_left_id = model.jnt_dofadr[gripper_axis_left_id]
                    model.dof_armature[dof_left_id] = 1e7                    ### Assembly
                    data.ctrl[:6] = pos_USB_assembly
                    gripper_state(state="closed", position=0.0048)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("USB assembled.")

                    model.dof_armature[dof_right_id] = 0    
                    model.dof_armature[dof_left_id] = 0     

                    ### Assembled
                    data.ctrl[:6] = pos_USB_assembled
                    gripper_state()
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("USB assembly complete.")
                    # time.sleep(2)
                    
                elif assembly_task == "KET8":
                    ### Camerapos.
                    viewer.cam.azimuth = -90
                    viewer.cam.elevation = -10
                    viewer.cam.distance = 0.1
                    viewer.cam.lookat = np.array([-0.5665, 0.4725, 0.05])


                    ### Calculating joint positions
                    
                    # start in home position
                    data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                    
                    # KET8 above
                    goal = [-0.5665, 0.4725, 0.09]
                    orientation = [0, 0, -90]
                    init = pos_home[:6]        
                    pos_KET8_above = calculate_ik(model, data, goal, orientation, 
                                                init)[robotjoints]
                    print("Position KET8 above calculated.")
                    
                    
                    # KET8 gripping
                    goal = [-0.5665, 0.4725, 0.0465]
                    orientation = [0, 0, -90]
                    init = pos_KET8_above       
                    pos_KET8_gripping = calculate_ik(model, data, goal, orientation, 
                                                    init)[robotjoints]
                    print("Position KET8 gripping calculated.")
                    
                    # KET8 above gripped
                    goal = [-0.5665, 0.4725, 0.09]
                    orientation = [0, 0, -90]
                    init = pos_KET8_gripping
                    pos_KET8_above_gripped = calculate_ik(model, data, goal, orientation, 
                                                        init)[robotjoints]
                    print("Position KET8 above gripped calculated.")
                    
                    
                    ### SEARCH STRATEGY
                    
                    # Starting position for search strategy
                    # Depends also on startingpos_shift
                    site_name = "KET8_searchAngled"
                    data.site(site_name).xpos += np.array([startingpos_shift, 0, 0])
                    model.site(site_name).pos = data.site(site_name).xpos
                    startpos = data.site(site_name).xpos.copy()

                    
                    # KET8 angled            
                    orientation = [5, 0, -90]
                    init = pos_KET8_above_gripped 
                    pos_KET8_angled = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                    print("Position KET8 angled calculated.")
                    
                    
                    # KET8 touch Z
                    startpos += [0, 0, -0.0039]
                    orientation = [5, 0, -90]
                    init = pos_KET8_angled 
                    pos_KET8_touchZ = calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                    print("Position KET8 touchZ calculated.")

                    
                    # KET8 touch back
                    startpos += [0, -0.002, 0]
                    orientation = [5, 0, -90]
                    init = pos_KET8_touchZ 
                    pos_KET8_touchback = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("Position KET8 touchback calculated.")
                    
                    # KET8 touch front
                    startpos += np.array([0, 0.0015, -0.0005])
                    orientation = [5, 0, -90]
                    init = pos_KET8_touchback 
                    pos_KET8_touchfront = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("Position KET8 touchfront calculated.")
                    
                    
                    # KET8 touch side
                    startpos += np.array([-0.006, 0, 0])
                    orientation = [5, 0, -90]
                    init = pos_KET8_touchfront 
                    pos_KET8_touchside = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("Position KET8 touchside calculated.")
                    
                    
                    # KET8 assembly
                    startpos += np.array([0, 0, -0.028]) 
                    orientation = [5, 0, -90]
                    init = pos_KET8_touchside
                    pos_KET8_assembly = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("KET8 assembly position calculated.")
            
                    # KET8 assembled
                    startpos += [0.0055 - startingpos_shift, 0, 0.005]
                    orientation = [0, 0, -90]
                    init = pos_KET8_assembly        
                    pos_KET8_assembled = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("KET8 assembled position calculated.")

                    
                    ### SIMULATION 
                    
                    # timestep
                    options.timestep = 5e-5 

                    # start simulation in home position
                    data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                    print("Home position reached.")

                      # KET8 Above
                    data.ctrl[:6] = pos_KET8_above
                    gripper_state()
                    if not safe_simulate_tol(target_site_name="KET8_above"):
                        simulation_continue = False
                        break
                    print("Position KET8 Above reached.")



                    # KET8 gripping 
                    data.ctrl[:6] = pos_KET8_gripping
                    gripper_state()
                    if not safe_simulate_tol(target_site_name="KET8_gripping"):
                        simulation_continue = False
                        break
                    print("Position KET8 gripping reached.")
                    
                    # close gripper
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position KET8 gripping reached and gripper closed.")
                    
                    
                    # KET8 Above gripped
                    data.ctrl[:6] = pos_KET8_above_gripped
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_tol(target_site_name="KET8_above"):
                        simulation_continue = False
                        break
                    print("Position KET8 above gripped reached.")
                    
                    
                    ### SEARCH STRATEGY
                    
                    # Angled
                    data.ctrl[:6] = pos_KET8_angled
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_tol(target_site_name="KET8_searchAngled", tol=1e-4):
                        simulation_continue = False
                        break
                    print("Position KET8 angled reached.")
                    
                    
                    # Touch Z
                    data.ctrl[:6] = pos_KET8_touchZ
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("Position KET8 touchZ reached.")


                    
                    # Touch Back
                    data.ctrl[:6] = pos_KET8_touchback
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position KET8 touch back reached.")
                    

                    # Touch Front
                    data.ctrl[:6] = pos_KET8_touchfront
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position KET8 touch front reached.")


                    # Touch Side
                    data.ctrl[:6] = pos_KET8_touchside
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("Position KET8 touch side reached.")

                    gripper_axis_right_id = mujoco.mj_name2id(model, 
                                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                                            'gripper_axis_right_joint')
                    dof_right_id = model.jnt_dofadr[gripper_axis_right_id]
                    model.dof_armature[dof_right_id] = 1e7    
                    
                    
                    gripper_axis_left_id = mujoco.mj_name2id(model, 
                                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                                            'gripper_axis_left_joint')
                    dof_left_id = model.jnt_dofadr[gripper_axis_left_id]
                    model.dof_armature[dof_left_id] = 1e7                    # Assembly
                    data.ctrl[:6] = pos_KET8_assembly
                    gripper_state(state="closed", position=0.0023)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("KET8 assembled.")
                    
                    model.dof_armature[dof_right_id] = 0    
                    model.dof_armature[dof_left_id] = 0     


                    ### Assembled
                    data.ctrl[:6] = pos_KET8_assembled
                    gripper_state()
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("KET8 assembly complete.")


                    # time.sleep(2)
                    
                elif assembly_task == "KET12":
                    ### Camerapos.
                    viewer.cam.azimuth = 0
                    viewer.cam.elevation = -10
                    viewer.cam.distance = 0.1
                    viewer.cam.lookat =np.array([-0.3414, 0.397, 0.05])

                    
                    
                    ### Calculating joint positions
                    
                    # start in home position
                    data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                    
                    # KET12 above
                    goal = [-0.3414, 0.3971, 0.09]
                    orientation = [0, 0, 0]
                    init = pos_home[:6]        
                    pos_KET12_above = calculate_ik(model, data, goal, orientation, 
                                                init)[robotjoints]
                    print("Position KET12 above calculated.")
                    
                    
                    # KET12 gripping
                    goal = [-0.3414, 0.3971, 0.0465]
                    orientation = [0, 0, 0]
                    init = pos_KET12_above       
                    pos_KET12_gripping = calculate_ik(model, data, goal, orientation, 
                                                    init)[robotjoints]
                    print("Position KET12 gripping calculated.")
                    
                    # KET12 above gripped
                    goal = [-0.3414, 0.3971, 0.09]
                    orientation = [0, 0, 0]
                    init = pos_KET12_gripping
                    pos_KET12_above_gripped = calculate_ik(model, data, goal, orientation, 
                                                        init)[robotjoints]
                    print("Position KET12 above gripped calculated.")
                    

                    ### SEARCH STRATEGY
                    
                    # Starting position for search strategy
                    # Depends also on startingpos_shift
                    site_name = "KET12_searchAngled"
                    data.site(site_name).xpos += np.array([0, startingpos_shift, 0])
                    model.site(site_name).pos = data.site(site_name).xpos
                    startpos = data.site(site_name).xpos.copy()


                    orientation = [5, 0, 0]
                    init = pos_KET12_above_gripped 
                    pos_KET12_angled= calculate_ik(model, data, startpos, orientation, 
                                                init)[robotjoints]
                    print("Position KET12 angled calculated.")
                    
                    
                    # KET12 touch Z
                    startpos += [0, 0, -0.0041]
                    orientation = [5, 0, 0]
                    init = pos_KET12_angled 
                    pos_KET12_touchZ = calculate_ik(model, data, startpos, orientation, 
                                                    init)[robotjoints]
                    print("Position KET12 touchZ calculated.")

                    # KET12 touch back
                    startpos += [0.002, 0, 0]
                    orientation = [5, 0, 0]
                    init = pos_KET12_touchZ 
                    pos_KET12_touchback = calculate_ik(model, data, startpos, orientation,
                                                    init)[robotjoints]
                    print("Position KET12 touchback calculated.")
                    
                    
                    # KET12 touch front
                    startpos += [-0.0015, 0, -0.001]
                    orientation = [5, 0, 0]
                    init = pos_KET12_touchback 
                    pos_KET12_touchfront = calculate_ik(model, data, startpos, orientation, 
                                                        init)[robotjoints]
                    print("Position KET12 touchfront calculated.")
                    
                    
                    # KET12 touch side
                    startpos += [0, -0.007, 0]
                    orientation = [5, 0, 0]
                    init = pos_KET12_touchfront 
                    pos_KET12_touchside = calculate_ik(model, data, startpos, orientation, 
                                                        init)[robotjoints]
                    print("Position KET12 touchside calculated.")
                    
                    
                    # KET12 assembly
                    startpos += [0, 0, -0.028]
                    orientation = [5, 0, 0]
                    init = pos_KET12_touchside
                    pos_KET12_assembly = calculate_ik(model, data, startpos, orientation, 
                                                        init)[robotjoints]
                    print("KET12 assembly position calculated.")
                    
                    # KET12 assembled
                    startpos += [0, 0.006 - startingpos_shift, 0.005]
                    orientation = [0, 0, 0]
                    init = pos_KET12_assembly        
                    pos_KET12_assembled = calculate_ik(model, data, startpos, orientation, 
                                                        init)[robotjoints]
                    print("KET12 assembled position calculated.")


                    ### SIMULATION 
                    
                    # timestep
                    options.timestep = 5e-5 

                    # start simulation in home position
                    data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home
                    print("Home position reached.")

                      # KET12 Above
                    data.ctrl[:6] = pos_KET12_above
                    gripper_state()
                    if not safe_simulate_tol(target_site_name="KET12_above"):
                        simulation_continue = False
                        break
                    print("Position KET12 Above reached.")



                    # KET12 gripping 
                    data.ctrl[:6] = pos_KET12_gripping
                    gripper_state()
                    if not safe_simulate_tol(target_site_name="KET12_gripping", tol=1e-4):
                        simulation_continue = False
                        break
                    print("Position KET12 gripping reached.")
                    
                    # close gripper
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position KET12 gripping reached and gripper closed.")
                    
                    
                    # KET12 Above gripped
                    data.ctrl[:6] = pos_KET12_above_gripped
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_tol(target_site_name="KET12_above"):
                        simulation_continue = False
                        break
                    print("Position KET12 above gripped reached.")
                    
                    
                    ### SEARCH STRATEGY
                    
                    # Angled
                    data.ctrl[:6] = pos_KET12_angled
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_tol(target_site_name="KET12_searchAngled", tol=1e-4):
                        simulation_continue = False
                        break
                    print("Position KET12 angled reached.")
                    
                    
                    # Touch Z
                    data.ctrl[:6] = pos_KET12_touchZ
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("Position KET12 touchZ reached.")


                    
                    # Touch Back
                    data.ctrl[:6] = pos_KET12_touchback
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position KET12 touch back reached.")
                    

                    # Touch Front
                    data.ctrl[:6] = pos_KET12_touchfront
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_duration(5):
                        simulation_continue = False
                        break
                    print("Position KET12 touch front reached.")


                    # Touch Side
                    data.ctrl[:6] = pos_KET12_touchside
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("Position KET12 touch side reached.")

                    gripper_axis_right_id = mujoco.mj_name2id(model, 
                                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                                            'gripper_axis_right_joint')
                    dof_right_id = model.jnt_dofadr[gripper_axis_right_id]
                    model.dof_armature[dof_right_id] = 1e7    
                    
                    gripper_axis_left_id = mujoco.mj_name2id(model, 
                                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                                            'gripper_axis_left_joint')
                    dof_left_id = model.jnt_dofadr[gripper_axis_left_id]
                    model.dof_armature[dof_left_id] = 1e7                    # Assembly
                    data.ctrl[:6] = pos_KET12_assembly
                    gripper_state(state="closed", position=0.0047)
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("KET12 assembled.")

                    model.dof_armature[dof_right_id] = 0    
                    model.dof_armature[dof_left_id] = 0     


                    ### Assembled
                    data.ctrl[:6] = pos_KET12_assembled
                    gripper_state()
                    if not safe_simulate_duration(10):
                        simulation_continue = False
                        break
                    print("KET12 assembly complete.")

                    # time.sleep(2)
          # 每次模拟结束后关闭查看器
        viewer.close()
        print(f"=== 第 {run_num}/{repeat_count} 次模拟完成 ===")
        
        # 如果查看器被用户关闭或模拟被中断，退出所有重复
        if not simulation_continue:
            print(f"=== 模拟被用户取消，已完成 {run_num} 次模拟 ===")
            break
    
    # 所有模拟完成后的总结
    if simulation_continue:
        print(f"\n=== 所有 {repeat_count} 次模拟已完成 ===")
    else:
        print(f"\n=== 模拟被用户取消 ===") 

def main():
    # 创建并显示参数窗口
    param_window = ParameterWindow()
    param_window.show()

if __name__ == "__main__":
    main() 