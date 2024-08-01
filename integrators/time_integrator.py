from body import Body

import numpy as np
from numpy.typing import NDArray
from typing import List, Callable
from abc import ABC, abstractmethod


class TimeIntegrator(ABC):
    time_step: float
    run_time: float
    time: float
    diff_eq: Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
    y_vec_arr: List[NDArray[np.float64]]
    
    @abstractmethod
    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError
    
    def set_y_vec_arr(self, y_vec_array: List[NDArray[np.float64]]):
        self.y_vec_arr = y_vec_array

    def set_diff_eq(self, func: Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]):
        self.diff_eq = func

    def update_time_step(self) -> bool:
        self.time += self.time_step
        return self.time < self.run_time


class ForwardEuler(TimeIntegrator):

    def __init__(self, time_step: float, run_time: float) -> None:
        self.time_step = time_step
        self.run_time = run_time
        self.time = 0

    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        return y_vec + self.time_step * self.diff_eq(self.time, y_vec, masses)


class RungeKutta4(TimeIntegrator):

    def __init__(self, time_step: float, run_time: float):
        self.time_step = time_step
        self.run_time = run_time
        self.time = 0

    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        k1: NDArray[np.float64] = self.time_step * self.diff_eq(self.time, y_vec, masses)
        k2: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + 0.5 * self.time_step, y_vec + 0.5 * k1, masses)
        k3: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + 0.5 * self.time_step, y_vec + 0.5 * k2, masses)
        k4: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + self.time_step, y_vec + k3, masses)

        return y_vec + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    

class VectorVerlet(TimeIntegrator):

    def __init__(self, time_step: float, run_time: float):
        self.time_step = time_step
        self.run_time = run_time
        self.time = 0

    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        # time = t
        dt1 : NDArray[np.float64] = self.diff_eq(self.time, y_vec, masses)
        for i in range(0, y_vec.shape[0], 6): # x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            y_vec[i] += self.time_step * dt1[i] + (0.5 * self.time_step ** 2) * dt1[i+3]
            y_vec[i + 1] += self.time_step * dt1[i + 1] + (0.5 * self.time_step ** 2) * dt1[i+4]
            y_vec[i + 2] += self.time_step * dt1[i + 2] + (0.5 * self.time_step ** 2) * dt1[i + 5]

        # time = t + dt
        dt2 : NDArray[np.float64] = self.diff_eq(self.time + self.time_step, y_vec, masses)
        for i in range(0, y_vec.shape[0], 6): # v(t + dt) = v(t) + 0.5*a(t)*a(t+dt)*dt
            y_vec[i+3] += 0.5 * self.time_step * (dt1[i+3] + dt2[i+3])
            y_vec[i+4] += 0.5 * self.time_step * (dt1[i+4] + dt2[i+4])
            y_vec[i+5] += 0.5 * self.time_step * (dt1[i+5] + dt2[i+5])

        return y_vec


class RungeKutta3dash8(TimeIntegrator):

    def __init__(self, time_step: float, run_time: float):
        self.time_step = time_step
        self.run_time = run_time
        self.time = 0

    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        k1: NDArray[np.float64] = self.time_step * self.diff_eq(self.time, y_vec, masses)
        k2: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + (1/3) * self.time_step, y_vec + (1/3) * k1, masses)
        k3: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + (2/3) * self.time_step, y_vec - (1/3) * k1 + k2, masses)
        k4: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + self.time_step, y_vec + k1 - k2 + k3, masses)

        return y_vec + (k1 + 3 * k2 + 3 * k3 + k4) / 8


class RungeKuttaFehlberg(TimeIntegrator):

    def __init__(self, time_step: float, run_time: float):
        self.time_step = time_step
        self.run_time = run_time
        self.time = 0

    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        k1: NDArray[np.float64] = self.time_step * self.diff_eq(self.time, y_vec, masses)
        k2: NDArray[np.float64] = self.time_step * self.diff_eq(self.time * (1/4) * self.time_step, y_vec + (1/4) * k1, masses)
        k3: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + (3/8) * self.time_step, y_vec + (3/32) * k1 + (9/32) * k2, masses)
        k4: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + (12/13) * self.time_step, y_vec + (1932/2197) * k1 + (-7200/2197) * k2 + (7296/2197) * k3, masses)
        k5: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + self.time_step, y_vec + (439/216) * k1 + (-8) * k2 + (3680/513) * k3 + (-845/4104) * k4, masses)
        k6: NDArray[np.float64] = self.time_step * self.diff_eq(self.time + 0.5 * self.time_step, y_vec + (-8/27) * k1 + (2) * k2 + (-3544/2565) * k3 + (1859/4104) * k4 + (-11/40) * k5, masses)

        return y_vec + ((16/135) * k1 + (6656/12825) * k3 + (28561/56430) * k4 + (-9/50) * k5 + (2/55) * k6)


class TwoStepAdamsBashforth(TimeIntegrator):

    def __init__(self, time_step: float, run_time: float):
        self.time_step = time_step
        self.run_time = run_time
        self.time = 0

    def simulate_step(self, y_vec: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(self.y_vec_arr) == 1:
            return y_vec + self.time_step * self.diff_eq(self.time, y_vec, masses)
        
        dt1: NDArray[np.float64] = (3/2) * self.time_step * self.diff_eq(self.time, y_vec, masses)
        dt0: NDArray[np.float64] = (-1/2) * self.time_step * self.diff_eq(self.time - self.time_step, self.y_vec_arr[-1], masses)

        return y_vec + dt1 + dt0
