import os
import time
from typing import Optional, Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agxPythonModules.utils.environment import application, simulation
from agxPythonModules.utils.callbacks import KeyboardCallback as kc

import agx
import agxRender


try:
    from PySide2 import QtWidgets, QtGui
except ImportError:
    print("Could not find PySide2. Continuing without displaying images")
    QtWidgets = None
    QtGui = None


class ExitException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class DisplayVirtualCameraGUI():
    '''
    PySide2 gui for displaying a list of virtual camera sensors
    '''

    def __init__(self, virtual_cameras, width=None, height=None):
        self._virtual_cameras = virtual_cameras
        self._width = width
        self._height = height

        self._qt_app = QtWidgets.QApplication.instance()
        if self._qt_app is None:
            self._qt_app = QtWidgets.QApplication([])
        self._qt_app.setQuitOnLastWindowClosed(True)
        self._qt_window = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout2 = QtWidgets.QHBoxLayout()
        layout.addLayout(layout2)

        self._qt_labels = []
        for _ in virtual_cameras:
            label = QtWidgets.QLabel("No camera feed")
            layout2.addWidget(label)
            self._qt_labels.append(label)

        self._qt_window.setLayout(layout)
        self._qt_window.show()

    def update(self):
        for c, l in zip(self._virtual_cameras, self._qt_labels):
            q_image = self.get_qt_image(c.image)
            l.setPixmap(QtGui.QPixmap(q_image))
        self._qt_app.processEvents()

    def get_qt_image(self, img):
        if img.dtype != np.dtype('uint8'):
            # assume float image between 0, 1. Then scale to 0 255
            img = 255 * img
            # and cast to uint8
            img.astype(np.uint8)
        if img.shape[2] == 1:
            # stack the only channel to three channels
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] > 3:
            # only take the three first channels
            img = img[:, :, :3]
        img = np.require(img, np.uint8, 'C')
        qt_img = QtGui.QImage(img,
                              img.shape[1],
                              img.shape[0],
                              QtGui.QImage.Format_RGB888)
        if self._width is not None and self._height is not None:
            qt_img = qt_img.scaled(self._width, self._height)
        return qt_img


class EnvironmentSceneDecorator():
    def __init__(self, app):
        self.app = app
        self.step = 0
        self.cumulative_reward = 0

    def update(self, step, reward, observation=None):
        if step < self.step:
            self.cumulative_reward = 0
            self.step = step

        self.step += 1
        self.cumulative_reward += reward

        color = agxRender.Color.Black()

        row = 1
        self.app.getSceneDecorator().setText(row, f"Step: {self.step}", color)
        row += 1
        self.app.getSceneDecorator().setText(row, f"Reward: {reward:.2f}", color)
        row += 1
        self.app.getSceneDecorator().setText(row, f"Cumulative reward: {self.cumulative_reward:.2f}", color)
        row += 1

        if observation is not None:
            s = "Observation: "
            for o in observation:
                s += f"{o:.2f}, "
            self.app.getSceneDecorator().setText(row, s, color)
            row += 1


class KeyBoardListenerWrapper(gym.Wrapper):
    '''
    Adds a GuiEventListener to existing ExampleApplication instance in
    the environment that listens to these keyboard events:

    e: Pauses the stepping. Do not step the simuation. But continue to render.

    n: Resets the environment now.
    '''
    def __init__(self, env: gym.Env):
        super().__init__(env)
        kc.bind(name='toggle stepping',
                key="e",
                mode=kc.Mode.NATIVE,
                callback=self._on_toggle_stepping)
        kc.bind(name='RESET',
                key="n",
                mode=kc.Mode.NATIVE,
                callback=self._on_reset)

        self.environment_stepping_enabled = True
        self.prev_step_result = None
        # obs, reward, done, info
        self.prev_obs = None
        self.prev_reward = None
        self.prev_done = None
        self.prev_truncated = None
        self.prev_info = None
        self.prev_action = None

        self.should_reset = False

        # We need a timer to determine when the graphics should be rendered to
        # comply to the setTargetFPS
        self.m_timer = agx.HighAccuracyTimer(True)

        self.init_wall_clock_time = time.perf_counter()

    def _on_toggle_stepping(self, data):
        if data.down:
            self.environment_stepping_enabled = not self.environment_stepping_enabled
            self.prev_reward = 0

    def _on_reset(self, data):
        if data.down:
            self.should_reset = True

    def step(self, action):
        current_time = time.perf_counter()
        if simulation().getTimeStamp() > current_time - self.init_wall_clock_time:
            time.sleep(simulation().getTimeStamp() - (current_time - self.init_wall_clock_time))
        self.prev_action = action
        if self.environment_stepping_enabled:
            self.prev_obs, self.prev_reward, self.prev_done, self.prev_truncated, self.prev_info = super().step(action)
        else:
            if application() is not None:
                if application().done():
                    raise ExitException("*** User has closed the application.")
            else:
                raise ExitException("*** ExampleApplication must exist for KeyBoardListenerWrapper exists.")
            application().executeOneStepWithGraphics(self.m_timer)
            self.prev_reward = 0

        return self.prev_obs, self.prev_reward, self.prev_done or self.should_reset, self.prev_truncated or self.should_reset, self.prev_info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        r = super().reset(seed=seed, options=options)
        self.init_wall_clock_time = time.perf_counter()
        self.should_reset = False
        return r


class NormalizeObsSpace(gym.ObservationWrapper):
    ''' Normalize a Box action space to [-1,1]^n '''

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        self.observation_space = spaces.Box(
            low=-np.ones_like(env.observation_space.low),
            high=np.ones_like(env.observation_space.low),
            dtype=np.float64
        )
        self.log_observations = False
        self.unscaled_observations = [[] for _ in range(self.observation_space.shape[0])]
        self.scaled_observations = [[] for _ in range(self.observation_space.shape[0])]

    def observation(self, obs):
        n_obs = obs.copy()

        if self.log_observations:
            for o, l in zip(n_obs, self.unscaled_observations):
                l.append(o)

        n_obs -= self.env.observation_space.low
        n_obs /= (self.env.observation_space.high - self.env.observation_space.low) / 2
        n_obs = n_obs - 1

        if self.log_observations:
            for o, l in zip(n_obs, self.scaled_observations):
                l.append(o)

        return n_obs


def get_name_from_hp(path, **kwargs):
    p = os.path.join(path, "hp_")
    for k, v in kwargs.items():
        p += f"{k}_{v}-"
    return p
