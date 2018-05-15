import numpy as np
from physics_sim import PhysicsSim

class Task():
    
    """Task (environment) that defines the goal and provides feedback to the agent."""
    # the goal is to lift off the ground and reach a target height 10
    def __init__(self, init_pose=np.array([0., 0., 2.,0.,0.,0]), init_velocities=np.array([0., 0., 0.]), 
        init_angle_velocities=np.array([0., 0., 0.]), runtime=5., target_pos=np.array([0., 0., 10.])):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = zero for matching target z. Farther you go, the larger the negative reward gets, up to -2,
        reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)/10.0 
        
        # reward the vertical velocity so agent will be rewarded to fly vertical upwards
        reward += min(self.sim.v[2], 500.0)/100.0
        
        # agent has crossed the target height
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 2.0  # bonus reward
            done = True
            # penalize crash
        if self.sim.pose[2] <= 0 and self.sim.time < self.sim.runtime: 
            reward -= 10
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    
#        def step(self, state, action):
#         """Uses action to obtain next state, reward, done."""
#         reward = 0
#         pose_all = []
#         actionX = self.action_space[action]
#         speed_each_rotor = np.array([actionX, actionX, actionX, actionX])
        
#         for i in range(self.action_repeat):
#             done = self.sim.next_timestep(speed_each_rotor)
#             reward = reward + self.get_reward()
#             pose_all.append(self.sim.pose)
            
#         next_state = np.concatenate(pose_all)
        
#         return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state