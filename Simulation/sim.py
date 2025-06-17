import mujoco
import numpy as np
import time
from mujoco.viewer import launch
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import simpledialog






"""
                                    GLOBALS
"""

PI = np.pi


"""
                        CHOOSING THE ASSEMBLY TASK
                    
Opens a window with four Options: RJ45, USB, KET8 and KET12. 
By selecting one of the options and pressing ok, the corresponding XML file and
robot movement is loaded. 
"""

def choose_assembly_task():
    # Create a new window
    root = tk.Tk()
    root.geometry("250x175")
    root.title("Choose assembly task:")

    # Text
    instruction_label = tk.Label(root, 
                        text="Please choose one of the following options:")
    instruction_label.pack(pady=10)

    # Variable for the selected option
    selected_option = tk.StringVar(value="RJ45")

    # Choosable options
    options = ["RJ45", "USB", "KET8", "KET12"]

    # Create a radiobutton for every option
    for option in options:
        tk.Radiobutton(root, text=option, variable=selected_option, 
                       value=option).pack(anchor="w")

    # OK-Button, to confirm choice
    def on_ok():
        root.quit()  # Quits the mainloop
        root.destroy()  # Closes the window

    tk.Button(root, text="OK", command=on_ok).pack()

    root.mainloop()

    # returns chosen option
    return selected_option.get()


chosen_option = choose_assembly_task()


# Sets the scene and assembly task based on chosen option
xml = ""

if chosen_option == "RJ45":
    xml = "scene_chamf.xml"
 
elif chosen_option == "USB":
    xml = "scene_15.xml"
    assembly_task = "USB"

elif chosen_option == "KET8":
    xml = "scene_15.xml"
    assembly_task = "KET8"

elif chosen_option == "KET12":
    xml = "scene_15.xml"
    assembly_task = "KET12" 



"""
                        LOADING THE MODEL
                        
Loading the model and initializing options for the camera and the simulation.
"""

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()                        
option = mujoco.MjvOption()     
options = model.opt
# options.enableflags |= (1 << 4)
# options.timestep = 1e-5
# options.noslip_iterations = 10
# options.impratio = 3
# options.iterations = 100
# options.solver = 0 
# options.integrator = 2




"""
                    CHANGING THE STIFFNESS OF THE GRIPPER
                    
Opens six windows to set the stiffness of the gripper. The first three entries
are for the directions in y- (compliance), x- (assembly) and z-direction. The
last three entries are for the x-, y- and z-rotations.
The values are from the data type float and are entered with a dot as decimal 
separator.

Reference values:
compliance dir.:    1-4
assembly dir.:      20-80
z dir.:             100
x- and y-rotation:  10
z-rotation:         0.5-2


When all entries are made, the stiffness of the gripper in the simulation is
set accordingly. The values are set for both grippers.
"""

def open_input_window_for_stiffness():
    # Create a new window
    root = tk.Tk()
    root.withdraw()  # Hide main window, only show input window

    # Entry prompt for the parameters
    param1 = simpledialog.askfloat("Parameter 1", 
        "Please enter the stiffness value in compliance direction (N/mm):")
    param2 = simpledialog.askfloat("Parameter 2", 
        "Please enter the stiffness value in assembly direction (N/mm):")
    param3 = simpledialog.askfloat("Parameter 3", 
        "Please enter the stiffness value in z direction (N/mm):")
    param4 = simpledialog.askfloat("Parameter 4", 
        "Please enter the stiffness value in x rotation (Nm/rad):")
    param5 = simpledialog.askfloat("Parameter 5", 
        "Please enter the stiffness value in y rotation (Nm/rad):")
    param6 = simpledialog.askfloat("Parameter 6", 
        "Please enter the stiffness value in z rotation (Nm/rad):")

    # Closes main window after prompt is made
    root.destroy()

    # Returns entered values
    return param1, param2, param3, param4, param5, param6


param1, param2, param3, param4, param5, param6 = \
    open_input_window_for_stiffness()


# Get ids of the joints

# left
left_compliance_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'compliance_dir_left')
left_assembly_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'assembly_dir_left')
left_z_id           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'z_dir_left')
left_rotx_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'rot_x_left')
left_roty_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'rot_y_left')
left_rotz_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'rot_z_left')

# right
right_compliance_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'compliance_dir_right')
right_assembly_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'assembly_dir_right')
right_z_id          = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'z_dir_right')
right_rotx_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'rot_x_right')
right_roty_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'rot_y_right')
right_rotz_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        'rot_z_right')


# set joint stiffness
# stiffness for slide joints multiplied by 1000 to convert units to N/m

# left
model.jnt_stiffness[left_compliance_id]     = param1 * 1000
model.jnt_stiffness[left_assembly_id]       = param2 * 1000
model.jnt_stiffness[left_z_id]              = param3 * 1000
model.jnt_stiffness[left_rotx_id]           = param4
model.jnt_stiffness[left_roty_id]           = param5
model.jnt_stiffness[left_rotz_id]           = param6

# right
model.jnt_stiffness[right_compliance_id]    = param1 * 1000
model.jnt_stiffness[right_assembly_id]      = param2 * 1000
model.jnt_stiffness[right_z_id]             = param3 * 1000
model.jnt_stiffness[right_rotx_id]          = param4
model.jnt_stiffness[right_roty_id]          = param5
model.jnt_stiffness[right_rotz_id]          = param6


# Checking the values
current_stiffness = model.jnt_stiffness[left_compliance_id]
print(f"""The current stiffness of the joint 'left_compliance_id' is: 
      {current_stiffness}""")

current_stiffness = model.jnt_stiffness[left_assembly_id]
print(f"""The current stiffness of the joint 'assembly_dir_left' is:
      {current_stiffness}""")

current_stiffness = model.jnt_stiffness[left_z_id]
print(f"""The current stiffness of the joint 'left_z_id' is:
      {current_stiffness}""")

current_stiffness = model.jnt_stiffness[left_rotx_id]
print(f"""The current stiffness of the joint 'left_rotx_id' is: 
      {current_stiffness}""")

current_stiffness = model.jnt_stiffness[left_roty_id]
print(f"""The current stiffness of the joint 'left_roty_id' is:
      {current_stiffness}""")

current_stiffness = model.jnt_stiffness[left_rotz_id]
print(f"""The current stiffness of the joint 'left_rotz_id' is:
      {current_stiffness}""")



"""
        CHANGING THE STARTING POSITION OF THE SEARCH STRATEGY
"""

def open_input_window_for_startingpos():
    # Create a new window
    root = tk.Tk()
    root.withdraw()  # Hide main window, only show input window

    # Entry prompt for the parameters
    param = simpledialog.askfloat("Parameter 7", 
        """Please enter the shift in starting position for the search strategy 
        (mm):""")

    # Closes main window after prompt is made
    root.destroy()

    # Returns entered values
    return param

# Shift is converted to m
startingpos_shift = open_input_window_for_startingpos() / 1000



"""
                        CALCULATE INVERSE KINEMATICS
                        
"""

# Calculation for the quaternion error with conjugation
def quaternion_error(target, current):
    """Compute the quaternion error."""
    return np.array([target[0]*current[1] - target[1]*current[0] -
                     target[2]*current[3] + target[3]*current[2],
                     target[0]*current[2] + target[1]*current[3] -
                     target[2]*current[0] - target[3]*current[1],
                     target[0]*current[3] - target[1]*current[2] +
                     target[2]*current[1] - target[3]*current[0]])


# Normalizing the gradient
def normalize_gradient(grad):
    """Normalize the gradient to ensure stable steps."""
    norm = np.linalg.norm(grad)
    return grad / norm if norm > 0 else grad


# State of the gripper (open, closed)
# If statements have no difference but are kept for clarity in function call
def gripper_state(position=0.01, state="open"):
    # Check the gripper state (open, closed)
    if state == "open":
        data.ctrl[6] = -position
        data.ctrl[7] =  position
    if state == "closed":
        data.ctrl[6] = -position
        data.ctrl[7] =  position




# Calculate the desired joint angles for the goal using gradient descent, 
# considering both position and orientation.

def calculate_ik(model, data, goal_pos, goal_euler, init_q, 
                 body_id=model.body("TCP").id, step_size=1e-4, tol=1e-4, 
                 alpha=0.1, max_iters=1e5):
    
    # Set current position to inital
    data.qpos[28:34] = init_q
    goal_quat = R.from_euler('xyz', goal_euler,
                             degrees=True).as_quat(scalar_first=True)
    # Calculate model
    mujoco.mj_forward(model, data)

    # Get current position and orientation of body_id
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



"""
                        SIMULATE FUNCTION 1: TIME BASED
                            
This functions simulates the movement of the robot based on the given duration.

Inputs:
- duration:         The duration in which the movement is simulated
- frames_rendered:  How many frames are being skipped and not rendered. 1: 
                    every frame is rendered. 5: only every fifth frame is 
                    rendered. This accelerates the simulation without 
                    influencing the calculations.
"""


def simulate_movement_duration(duration=15, frames_rendered=50):
 
    frames_rendered = frames_rendered   
    frame_count = 0
   
    time_start = time.monotonic()
 
    while (time.monotonic() <= time_start + duration):
        mujoco.mj_step(model, data)
        if frame_count % frames_rendered == 0:
            viewer.sync()
        frame_count += 1



"""
                    SIMULATE FUNCTION 2: TOLERANCE BASED 
                            
This functions simulates the movement of the robot based on the given tolerance.

Inputs:
- duration:         The duration in which the movement is simulated
- tol:              The tolerance between current position and target position
                    in meter.
- frames_rendered:  How many frames are being skipped and not rendered. 1: 
                    every frame is rendered. 5: only every fifth frame is 
                    rendered. This accelerates the simulation without 
                    influencing he calculations.
"""




def simulate_movement_tol(target_site_name="", tol=1e-3, frames_rendered=50):
    
    frames_rendered = frames_rendered   
    frame_count = 0
  
    mujoco.mj_fwdPosition(model, data)

    end_effector = data.body("TCP").xpos
    target_site = data.site(target_site_name).xpos
 
    while any(np.abs(end_effector - target_site) >= tol):
        mujoco.mj_step(model, data)
        if frame_count % frames_rendered == 0:
            viewer.sync()
        frame_count += 1
 


"""
                            POSITIONS OF THE ROBOT
"""

# Home: robot is in the middle of tray and NIST at a safe hight
pos_home = np.array([3*PI/2, -PI/2, -PI/2, 3*PI/2, PI/2, 0, -0.01, 0.01])

# setting the home position initially (gripper open)
data.qpos[[28, 29, 30, 31, 32, 33, 34, 41]] = pos_home





"""
                            LAUNCH THE VIEWER
                        AND START THE SIMULATION
                            
Open the viewer. Here, options for the viewer can also be made, for example 
camera position and showing contacts.

From the viewer.is_running() onwards the simulation is started. Depending on
the selected assembly task the corresponding section is loaded. Here, the 
robot movement is simulated according to the set robot positions. When
the movement is finished, the viewer is closed.
"""


# Viewer-Setup
viewer = mujoco.viewer.launch_passive(model, data)

# Show contact forces, contact points and force labels
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True  
viewer.opt.label = viewer.opt.label | mujoco.mjtLabel.mjLABEL_CONTACTFORCE  



 
robotjoints = [28, 29, 30, 31, 32, 33]


if viewer.is_running():

           
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
        simulate_movement_tol(target_site_name="RJ45_above")
        print("Position RJ45 above reached.")
        
        
        ### RJ45 gripping
        data.ctrl[:6] = pos_RJ45_gripping
        gripper_state()
        simulate_movement_tol(target_site_name="RJ45_gripping")
        print("Position RJ45 gripping reached.")
        
        # close gripper
        gripper_state(state="closed", position=0.003)
        simulate_movement_duration(5)
        print("Position RJ45 gripped reached.")
        
        
        ### RJ45 Above gripped
        data.ctrl[:6] = pos_RJ45_above_gripped
        gripper_state(state="closed", position=0.003)
        simulate_movement_tol(target_site_name="RJ45_above_gripped")
        print("Position RJ45 above gripped reached.")
        
        
        ### SEARCH STRATEGY
        
        ### RJ45 angled
        data.ctrl[:6] = pos_RJ45_angled
        gripper_state(state="closed", position=0.003)
        simulate_movement_tol(target_site_name="RJ45_searchAngled", tol=1e-4)
        print("Position RJ45 angled reached.")
        
        
        # RJ45 Touch Z
        data.ctrl[:6] = pos_RJ45_touchZ
        gripper_state(state="closed", position=0.003)
        simulate_movement_duration(10)
        print("Position RJ45 touchZ reached.")
        
        
        ### Touch Back
        data.ctrl[:6] = pos_RJ45_touchback
        gripper_state(state="closed", position=0.003)
        simulate_movement_duration(5)
        print("Position RJ45 touch back reached.")
        

        ### Touch Front
        data.ctrl[:6] = pos_RJ45_touchfront
        gripper_state(state="closed", position=0.003)
        simulate_movement_duration(5)
        print("Position RJ45 touch front reached.")


        ### Touch Side
        data.ctrl[:6] = pos_RJ45_touchside
        gripper_state(state="closed", position=0.003)
        simulate_movement_duration(10)
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
        model.dof_armature[dof_left_id] = 1e7     


        ### Assembly
        data.ctrl[:6] = pos_RJ45_assembly
        gripper_state(state="closed", position=0.003)
        simulate_movement_duration(10)
        print("RJ45 assembled.")

        model.dof_armature[dof_right_id] = 0    
        model.dof_armature[dof_left_id] = 0     
        

        ### Assembled
        data.ctrl[:6] = pos_RJ45_assembled
        gripper_state()
        simulate_movement_duration(10)
        print("RJ45 assembly complete.")
        

        
     
    if xml == "scene_15.xml" and assembly_task == "KET8":
        
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
        simulate_movement_tol(target_site_name="KET8_above")
        print("Position KET8 Above reached.")



        # KET8 gripping 
        data.ctrl[:6] = pos_KET8_gripping
        gripper_state()
        simulate_movement_tol(target_site_name="KET8_gripping")
        print("Position KET8 gripping reached.")
        
        # close gripper
        gripper_state(state="closed", position=0.0023)
        simulate_movement_duration(5)
        print("Position KET8 gripping reached and gripper closed.")
        
        
        # KET8 Above gripped
        data.ctrl[:6] = pos_KET8_above_gripped
        gripper_state(state="closed", position=0.0023)
        simulate_movement_tol(target_site_name="KET8_above")
        print("Position KET8 above gripped reached.")
        
        
        ### SEARCH STRATEGY
        
        # Angled
        data.ctrl[:6] = pos_KET8_angled
        gripper_state(state="closed", position=0.0023)
        simulate_movement_tol(target_site_name="KET8_searchAngled", tol=1e-4)
        print("Position KET8 angled reached.")
        
        
        # Touch Z
        data.ctrl[:6] = pos_KET8_touchZ
        gripper_state(state="closed", position=0.0023)
        simulate_movement_duration(10)
        print("Position KET8 touchZ reached.")


        
        # Touch Back
        data.ctrl[:6] = pos_KET8_touchback
        gripper_state(state="closed", position=0.0023)
        simulate_movement_duration(5)
        print("Position KET8 touch back reached.")
        

        # Touch Front
        data.ctrl[:6] = pos_KET8_touchfront
        gripper_state(state="closed", position=0.0023)
        simulate_movement_duration(5)
        print("Position KET8 touch front reached.")


        # Touch Side
        data.ctrl[:6] = pos_KET8_touchside
        gripper_state(state="closed", position=0.0023)
        simulate_movement_duration(10)
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
        model.dof_armature[dof_left_id] = 1e7     

        # Assembly
        data.ctrl[:6] = pos_KET8_assembly
        gripper_state(state="closed", position=0.0023)
        simulate_movement_duration(10)
        print("KET8 assembled.")
        
        model.dof_armature[dof_right_id] = 0    
        model.dof_armature[dof_left_id] = 0     


        ### Assembled
        data.ctrl[:6] = pos_KET8_assembled
        gripper_state()
        simulate_movement_duration(10)
        print("KET8 assembly complete.")


        time.sleep(5)



    
    if xml == "scene_15.xml" and assembly_task == "KET12":
        
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
        simulate_movement_tol(target_site_name="KET12_above")
        print("Position KET12 Above reached.")



        # KET12 gripping 
        data.ctrl[:6] = pos_KET12_gripping
        gripper_state()
        simulate_movement_tol(target_site_name="KET12_gripping",  tol=1e-4)
        print("Position KET12 gripping reached.")
        
        # close gripper
        gripper_state(state="closed", position=0.0047)
        simulate_movement_duration(5)
        print("Position KET12 gripping reached and gripper closed.")
        
        
        # KET12 Above gripped
        data.ctrl[:6] = pos_KET12_above_gripped
        gripper_state(state="closed", position=0.0047)
        simulate_movement_tol(target_site_name="KET12_above")
        print("Position KET12 above gripped reached.")
        
        
        ### SEARCH STRATEGY
        
        # Angled
        data.ctrl[:6] = pos_KET12_angled
        gripper_state(state="closed", position=0.0047)
        simulate_movement_tol(target_site_name="KET12_searchAngled", tol=1e-4)
        print("Position KET12 angled reached.")
        
        
        # Touch Z
        data.ctrl[:6] = pos_KET12_touchZ
        gripper_state(state="closed", position=0.0047)
        simulate_movement_duration(10)
        print("Position KET12 touchZ reached.")


        
        # Touch Back
        data.ctrl[:6] = pos_KET12_touchback
        gripper_state(state="closed", position=0.0047)
        simulate_movement_duration(5)
        print("Position KET12 touch back reached.")
        

        # Touch Front
        data.ctrl[:6] = pos_KET12_touchfront
        gripper_state(state="closed", position=0.0047)
        simulate_movement_duration(5)
        print("Position KET12 touch front reached.")


        # Touch Side
        data.ctrl[:6] = pos_KET12_touchside
        gripper_state(state="closed", position=0.0047)
        simulate_movement_duration(10)
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
        model.dof_armature[dof_left_id] = 1e7     


        # Assembly
        data.ctrl[:6] = pos_KET12_assembly
        gripper_state(state="closed", position=0.0047)
        simulate_movement_duration(10)
        print("KET12 assembled.")

        model.dof_armature[dof_right_id] = 0    
        model.dof_armature[dof_left_id] = 0     


        ### Assembled
        data.ctrl[:6] = pos_KET12_assembled
        gripper_state()
        simulate_movement_duration(10)
        print("KET12 assembly complete.")

        time.sleep(5)        
        
        
        

    if xml == "scene_15.xml" and assembly_task == "USB":
        
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
        simulate_movement_tol(target_site_name="USB_above")
        print("Position USB Above reached.")



        ### USB gripping 
        data.ctrl[:6] = pos_USB_gripping
        gripper_state()
        simulate_movement_tol(target_site_name="USB_gripping")
        print("Position USB gripping reached.")
        
        # close gripper
        gripper_state(state="closed", position=0.0048)
        simulate_movement_duration(5)
        print("Position USB gripping reached and gripper closed.")
        
        
        ### USB Above gripped
        data.ctrl[:6] = pos_USB_above_gripped
        gripper_state(state="closed", position=0.0048)
        simulate_movement_tol(target_site_name="USB_above_gripped")
        print("Position USB above gripped reached.")
        
        
        ### SEARCH STRATEGY
        
        ### Angled
        data.ctrl[:6] = pos_USB_angled
        gripper_state(state="closed", position=0.0048)
        simulate_movement_tol(target_site_name="USB_searchAngled", tol=1e-4)
        print("Position USB angled reached.")
        
        
        ### Touch Z
        data.ctrl[:6] = pos_USB_touchZ
        gripper_state(state="closed", position=0.0048)
        simulate_movement_duration(10)
        print("Position USB touchZ reached.")


        
        ### Touch Back
        data.ctrl[:6] = pos_USB_touchback
        gripper_state(state="closed", position=0.0048)
        simulate_movement_duration(5)
        print("Position USB touch back reached.")
        

        ### Touch Front
        data.ctrl[:6] = pos_USB_touchfront
        gripper_state(state="closed", position=0.0048)
        simulate_movement_duration(5)
        print("Position USB touch front reached.")


        ### Touch Side
        data.ctrl[:6] = pos_USB_touchside
        gripper_state(state="closed", position=0.0048)
        simulate_movement_duration(10)
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
        model.dof_armature[dof_left_id] = 1e7     


        ### Assembly
        data.ctrl[:6] = pos_USB_assembly
        gripper_state(state="closed", position=0.0048)
        simulate_movement_duration(10)
        print("USB assembled.")

        model.dof_armature[dof_right_id] = 0    
        model.dof_armature[dof_left_id] = 0     

        ### Assembled
        data.ctrl[:6] = pos_USB_assembled
        gripper_state()
        simulate_movement_duration(10)
        print("USB assembly complete.")
        time.sleep(5)

    
    viewer.close()

# Quit viewer
viewer.close()
















