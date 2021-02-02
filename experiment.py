import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import numpy as np
import sys
import copy
from numpy.linalg import norm, pinv
import data_loading as data_loading_module
from learn_lpv_ds import lpvds,mse,show_DS
from scipy.io import loadmat


# Class for the Kuka IIWA Experiment
class KukaLettersExperiment(object):
    """
    This class runs the KUKA IIWA Obstacle Avoidance experiment for Assignment 1
    and attempts to avoid deadlock.

    Attributes
    ----------

    dt : float
        Time step for the simulation
    clid : int
        PyBullet parameter for connecting to simulator.
    prevPose : list or list-like
        Previous position of end-effector in task space.
    prevRef : list or list-like
        Previous reference velocity (command) of end-effector in task space.
    hasPrevPose : bool
        If True, the simulation has been executed for previous timestep, and
        position has been updated
    robot : int
        PyBullet parameter to track bodyUniqueId
    robotNumJoints : int
        PyBullet parameter on number of joints on the robot
    robotEndEffectorIndex : int
        PyBullet parameter on the End-effector link's index value
    ll : list of float
        Lower limits of robot's joints in null space (size = robotNumJoints)
    ul : list of float
        Upper limits of robot's joints in null space (size = robotNumJoints)
    jr : list of float
        Joint ranges of robot's joints in null space (size = robotNumJoints)
    rp : list of float
        Rest poses of robot's joints in null space (size = robotNumJoints)
    jd : list of float
        Joint damping coefficients of robot's joints in null space
        (size = robotNumJoints)
    joint_ll: list of float
        Lower limits of robot's joint angles in joint space
    joint_ul: list of float
        Upper limits of robot's joint angles in joint space
    joint_vmax: list of float
        Upper limits of robot's joint velocities in joint space
    x : list or list-like
        Current robot's state in task space (End-effector position) in 3D space
    q : list or list-like
        Current robot's state in config space
        (Joint positions, size = robotNumJoints)
    J : numpy.ndarray
        Jacobian matrix based on the current joint positions
    trailDuration : int
        PyBullet parameter on number of steps to trail the debug lines
    x_start : list or list-like
        End-effector start position in task space
    x_target : list or list-like
        End-effector target position in task space
    t : float
        Current time in simulation since simulation started
    data_pos: np.ndarray
        Position values for
    data_vel: np.ndarray

    """


    # Set timestep in simulation
    dt = 1/100.

    def __init__(self, shape = "V"):
        """KukaRobotExperiment constructor. Initializes and runs the experiment

        Parameters
        ----------
        shape : string
            Letter shape to be drawn by the manipulator. One of "L", "V" or "S".

        Returns
        -------
        None

        """

        # Connect to PyBullet simulator
        self.clid = p.connect(p.SHARED_MEMORY)
        if self.clid < 0:
            p.connect(p.GUI)

        # Set PyBullet installed Data path for URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.shape_scale = 0.2
        # Load objects
        self.loadObjects()

        # load a dataset

        if shape == "L":
            self.data_pos, self.data_vel = data_loading_module.load('2D_Lshape_with_noise_level_0.4')
        elif shape == "V":
            self.data_pos, self.data_vel = data_loading_module.load('2D_Ashape_with_noise_level_0.4')
            self.data_pos[:,1] *= -3
            self.data_vel[:,1] *= -3
        elif shape == "S":
            self.data_pos, self.data_vel = data_loading_module.load('2D_Sshape')


        # Mirror the dataset (explanation in documentation)
        self.data_pos[:,0] *= -1
        self.data_vel[:,0] *= -1

        # Scale the dataset manually for Kuka robot
        self.data_pos *= self.shape_scale
        self.data_vel *= self.shape_scale * 0.5

        # Obtain hardcoded limits, ranges and coefficents
        self.setRobotLimitsRanges()

        # Initialize Robot to rest position
        self.setJointStates(self.rp)

        # Hardcoded value for rest position of Kuka
        self.prevPose = [0, 0, 0]
        self.prevRef = [0, 0, 0]
        self.hasPrevPose = False

        # Initialize states
        self.initializeParamsAndState()

        # Conduct experiment
        self.experiment()

        # Show experiment results
        self.experimentResults()

    def loadObjects(self):
        """ Loads the required models on the simulator and
        sets simulator paremeters

        Returns
        -------
        None

        """

        # Load floor plane at -2
        p.loadURDF("plane.urdf",[0,0,-2])

        # Load Robot
        self.robot = p.loadURDF("kuka_iiwa/model.urdf",[0,0,0])
        p.resetBasePositionAndOrientation(
            self.robot,
            [0, 0, 0],
            [0, 0, 0, 1]
        )

        # Joints and End effector Index
        self.robotNumJoints = p.getNumJoints(self.robot)
        self.robotEndEffectorIndex = 6
        assert self.robotNumJoints == 7, "Model incorrect"

        # Camera adjustment
        p.resetDebugVisualizerCamera(
            cameraDistance = 3,
            cameraYaw = 230,
            cameraPitch = -22,
            cameraTargetPosition = [0,0,0]
        )

        # Gravity setting
        p.setGravity(0, 0, 0)

        # Set timestep
        p.setTimeStep(self.dt)

        # Is Simulation Real Time?
        p.setRealTimeSimulation(0)

    def setRobotLimitsRanges(self):
        """Sets the Lower limits, upper limits, joint ranges and
        rest poses for the null space of the robot. Hardcoded values here.
        Also obtains the Joint angle limits and velocity limits from URDF model.

        Returns
        -------
        None

        """

        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]

        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]

        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]

        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]

        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.joint_ll = []
        self.joint_ul = []
        self.joint_vmax = []

        # For each joint in the manipulator
        for robotJoint in range(self.robotNumJoints):
            all_limits = p.getJointInfo(
                            bodyUniqueId = self.robot,
                            jointIndex = robotJoint
                        )

            # Extract the joint limits
            jointLowerLimit, jointUpperLimit, _, jointMaxVelocity = all_limits[8:12]
            self.joint_ll.append(jointLowerLimit)
            self.joint_ul.append(jointUpperLimit)
            self.joint_vmax.append(jointMaxVelocity)



    def updateState(self, updateJ = False):
        """Retrieves/Measures the end-effector position
        and derives the configuration position and Jacobian matrix from it.
        (Ideally, configuration position can also be measured directly)


        Parameters
        ----------
        updateJ : bool
            If True, Jacobian matrix is updated based on measurements
            (Default: False)

        Returns
        -------
        None

        """

        # Get link state
        linkState = p.getLinkState(
                        self.robot,
                        self.robotEndEffectorIndex,
                        computeLinkVelocity = 1,
                        computeForwardKinematics = 1
        )

        # Save x value and find q
        self.x = linkState[4]
        self.q = self.ik(self.x)

        # Calculate Jacobian
        if updateJ:
            J, _ = p.calculateJacobian(
                        bodyUniqueId = self.robot,
                        linkIndex = self.robotEndEffectorIndex,
                        localPosition = list(linkState[2]),
                        objPositions = list(self.q),
                        objVelocities = [0.] * len(list(self.q)),
                        objAccelerations = [0.] * len(list(self.q))
            )

            self.J = np.array(J)


    def setJointStates(self, q):
        """Hard set joint values. This is used to reset simulations or
        perform manipulations outside an experiment.

        Parameters
        ----------
        q : list or list-like of floats (size = number of joints of robot)
            Configuration states to reset the joints to.

        Returns
        -------
        None

        """

        # Set each joint's states
        for jointNumber in range(self.robotNumJoints):
            p.resetJointState(self.robot, jointNumber, float(q[jointNumber]))

    def ik(self, x):
        """Performs Inverse Kinematics to obtain the joint positions
        from the end-effector position (Config space from Task space)

        Parameters
        ----------
        x : list or list-like of 3 floats
            Task space state (End-effector position) in 3D space

        Returns
        -------
        numpy.array
            Configuration space state (Joint positions)
            Array of size = number of joints on robot

        """

        q = p.calculateInverseKinematics(
                        bodyUniqueId = self.robot,
                        endEffectorLinkIndex = self.robotEndEffectorIndex,
                        targetPosition = list(x),
                        lowerLimits = self.ll,
                        upperLimits = self.ul,
                        jointRanges = self.jr,
                        restPoses = self.rp
                        )

        return np.array(q)


    def initializeParamsAndState(self):
        """Initializes parameters such as start and target states.
        Also initilizes simulation parameters

        Returns
        -------
        None

        """

        # Trail debug line delay
        self.trailDuration = 150

        # Start states
        self.x_start = np.concatenate([[-.4], self.data_pos[0,:]])

        # Set current state to start state
        self.x = copy.deepcopy(self.x_start)
        self.q = self.ik(self.x_start)

        # Update states on robot
        self.setJointStates(self.q)

        # Target states
        self.x_target = np.concatenate([[-.4], self.data_pos[-1,:]])

        # Initialize time
        self.t = 0


    def setRobotTaskReference(self, x_dot_ref):
        """Specifies motor command given the reference velocity in
        Task space (end effector reference velocity).
        It convert task space reference to joint space reference using
        the pseudo-inverse of Jacobian matrix.
        It also performs redundancy resolution

        Parameters
        ----------
        x_dot_ref : list or list-like
            Reference velocity in task space. Must be of size 3.

        Returns
        -------
        None

        """

        # REDUNDANCY RESOLUTION

        # pseudo-inverse of jacobian
        pinv_J = pinv(self.J)

        # Null space projection
        projection = np.eye(7) - pinv_J @ self.J

        # Null space vel
        bqdot_0 = self.ik(self.x_start) - self.q

        # Task to Joint Reference
        q_dot_ref =  pinv_J @ x_dot_ref + projection @ bqdot_0


        # Set joint reference for each joint
        for robotJoint in range(self.robotNumJoints):

            #print(float("{:.2f}".format(q_dot_ref[robotJoint])), "qdr", robotJoint, end=" ")
            p.setJointMotorControl2(
                bodyIndex = self.robot,
                jointIndex = robotJoint,
                controlMode = p.VELOCITY_CONTROL,
                targetVelocity = float(q_dot_ref[robotJoint])
            )
        #print(" ")

    def checkKinematicLimits(self):
        """
        Checks if the robot is within its Kinematic
        (Joint angle and velocity) limits. Returns False if not within limits.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if robot is within joint angle and joint velocity limits.
            False otherwise.
        """

        within_limits = True

        # Obtain limit values from URDF
        for robotJoint in range(self.robotNumJoints):

            # Joint state and velocity
            jointState, jointVelocity, _, _ = p.getJointState(
                                                    bodyUniqueId = self.robot,
                                                    jointIndex = robotJoint
                                                )

            # Boolean to check state and velocity limits
            checkStateLimit = (jointState >= self.joint_ll[robotJoint]) and (jointState <= self.joint_ul[robotJoint])
            checkVelocityLimit = abs(jointVelocity) <= self.joint_vmax[robotJoint]

            # True only when previous joints are within limits and current joint
            # is within limits as well.
            within_limits = within_limits and checkStateLimit and checkVelocityLimit

        return within_limits


    def limitReference(self, x_dot):
        """
        Checks if the Robot is within joint velocity limits. This is calculated by
        converting to end-effector max velocity, and comparing with the end-effector
        reference velocity calculated by the planner.
        If not within limits, it will revert to reference velocity
        from previous iteration.

        Parameters
        ----------
        x_dot : list or list-like
            Reference velocity in task space. Must be of size 3.

        Returns
        -------
        list of float
            Adjusted Reference velocity.
        """

        # Find x_dot limits
        x_dot_ul = np.abs(self.J @ np.array(self.joint_vmax))

        # Check if previous reference velocity is within limits
        # If not, set new velocity to max velocity
        new_velocity = self.prevRef if np.any(self.prevRef <= x_dot_ul) else x_dot_ul

        # Limit to max and min values
        x_dot = new_velocity if np.any(np.abs(x_dot) > x_dot_ul) else x_dot

        return x_dot.tolist()


    def experiment(self):
        """Performs the experiment after all the initializations.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        # learning nonlinear dynamical systems
        tol = 0.0001
        self.model_ds = lpvds()
        self.model_ds.fit_ds(self.data_pos,self.data_vel,tol)

        # show_DS(self.model_ds, self.data_pos)

        # Continue motion until target is reached OR if time exceeds 90s
        while norm(self.x[1:] - self.x_target[1:]) > 1e-2 and self.t < 90 / self.dt:

            # Update timestep
            self.t += self.dt


            # Perform simulation in this step
            p.stepSimulation()

            # Obtain robot state by updating the parameters from measurements.
            self.updateState(updateJ = True)

            # Velocity
            x_mod = np.array(self.x[1:])[np.newaxis,:]

            # Predict reference velocity (Planned command)
            x_dot_ref = self.model_ds.predict(x_mod)

            # No velocity along x-axis
            x_dot_ref = np.concatenate([[0.], np.squeeze(x_dot_ref)])

            # Limit the value of x_dot_ref only after first
            if self.hasPrevPose:
                x_dot_ref = self.limitReference(x_dot_ref)

            # Check limits
            if not self.checkKinematicLimits():
                print("Robot is not within its Kinematics limits")

            # Move the robot joints based on given reference velocity.
            self.setRobotTaskReference(x_dot_ref)

            # Draw trail line of the end-effector path while debugging
            if self.hasPrevPose:
                p.addUserDebugLine(self.prevPose, self.x, [1, 0, 0], 1, self.trailDuration)

            # Keep track of previous iteration's position.
            self.prevPose = self.x
            self.prevRef = x_dot_ref
            self.hasPrevPose = True

    def experimentResults(self):
        """Generates and displays any experiment results that are shown
        after the simulation.

        Returns
        -------
        None

        """

        print("Reached target in: ",self.t," seconds")
        time.sleep(10)
        p.disconnect()

# Runs the following code when run from command line
if __name__ == "__main__":

    # Run experiment

    if len(sys.argv)>1:
        # This is the line to be modified to get different results
        KukaLettersExperiment(shape = sys.argv[1])
    else:
        KukaLettersExperiment()
