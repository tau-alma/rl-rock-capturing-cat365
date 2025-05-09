# TODO List

## Features in Progress

- [x] **Increase control penalty weight** (`experiment/increaseControlPenaltyWeight`)

- [x] **Test `squash_output` in PPO policy** (`feature/squash_output`)  
  _Added `squash_output=True` to the `policy_kwargs` and `use_sde=True` in PPO to constrain actions within [-1, 1]._

  **use_sde**: State Dependent Exploration  
  **Purpose**: To improve exploration by making the noise in the policy dependent on the current state, rather than just being random (like standard Gaussian noise).

  **squash_output**: Squash action using `tanh`  
  **Purpose**: To ensure that actions lie within the desired bounds when using gSDE (`use_sde=True`).

  _From Stable-Baselines3 Docs:_  
  https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example  

  > For A2C and PPO, continuous actions are clipped during training and testing (to avoid out of bound error).  
  > SAC, DDPG and TD3 squash the action, using a `tanh()` transformation, which handles bounds more correctly.

  _Also, in our current implementation, actions are clipped manually before being sent to the excavator._

- [ ] **Test PPO implementation from `skrl` library**

- [ ] **Include previous joint speed command in the observation space**

- [ ] **Design a reward-independent performance index**  
  _Helps compare different methods and reward functions objectively._

- [ ] **Revise reward function and termination conditions**  
  _Increase the positive terminal reward and allow episode termination when the task is complete, rather than using only max episode length._

- [ ] _(Optional)_ **Change domain randomization settings**  
  _E.g., change mass distribution._

- [ ] _(Optional)_ **Experiment with different network architectures**

- [ ] _(Optional)_ **Try alternative RL algorithms**

## 🧪 Experiments to Run

- [ ] **Evaluate generalization of trained agent**  
  _Test performance when rock or soil differs from training setup._

## 🔖 Notes

- Use branch naming conventions: `feature/` for new capabilities and `experiment/` for testing variants.




# TODO List 

## Features in Progress
- [x] Increase control penalty weight (branch: `experiment/increaseControlPenaltyWeight`)
- [x] Test squash_output in PPO policy (branch: `feature/squash_output`) [Adding the tanh activation function to the output layer of the policy network to generate the control input in the range [-1,1] (in the current implementation, we apply the clip function to the output of the policy network, and then send it to the excavator)]
- [ ] Test PPO implementation from `skrl` library
- [ ] Add the previous joint speed command to the observation space
- [ ] Desig an index independent of the reward function and the observation space to compare the performance of different methods and reward functions
- [ ] Change the reward function (increasing the terminal reward) and terminate the episode after reaching the desired conditions (in the current implementation, the episode terminates after reaching the maximum episode length)
- [ ] (optional) Change the randomization settings, for instance, changing the mass distribution, etc.
- [ ] (optional) Test different structures for policy and value networks
- [ ] (optional) Test different RL algorithms

## Experiments to Run
- [ ] Evaluation of a trained agent when the rock or soil is different from the training setup


## Notes
- Use `feature/` or `experiment/` prefixes for new branches.
- Always push branches for visibility even if not ready to merge.