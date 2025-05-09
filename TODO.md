# TODO List

## Features in Progress

- [ ] **Contact between the rock and bucket**  
  _Clarify whether explicit contact definition is needed. Current simulation includes:_  
  - `agxTerrain:Terrain::Particle <-> agxTerrain:Terrain::Particle`  
  - `agxTerrain:Terrain::Particle <-> BucketMaterial`  
  - `BucketMaterial <-> terrain`  
  - `agxTerrain:Terrain::Particle <-> Rocks`  
  - `Rocks <-> terrain`

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

- [ ] **Fix bug regarding the initial position of the rock**  
  _When the rock falls from a 0.5-meter height, it sometimes tilts and moves in the y-direction, causing it to go out of the working area of the arm, stick, and bucket (x-z plane)._

- [ ] **Test PPO implementation from `skrl` library**

- [ ] **Include previous joint speed command in the observation space**

- [ ] **Design a performance index independent of the reward function and observation space**  
  _Helps objectively compare different methods, observation spaces, and reward functions._

- [ ] **Revise reward function and termination conditions**  
  _Increase the positive terminal reward and allow episode termination when the task is complete, rather than using only max episode length._

- [ ] _(Optional)_ **Change domain randomization settings**  
  _E.g., change mass distribution._

- [ ] _(Optional)_ **Experiment with different network architectures**

- [ ] _(Optional)_ **Try alternative RL algorithms**

## Experiments to Run

- [ ] **Evaluate generalization of trained agent**  
  _Test performance when rock or soil differs from training setup._

## Notes

- Use branch naming conventions: `feature/` for new capabilities and `experiment/` for testing variants.
