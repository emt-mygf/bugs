import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.80  # Gravity
        self.viscosity = 0.07  # viscosity
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-3
    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    @ti.func
    def pressure_force(self, p_i, p_j, r):
        # Compute the pressure force contribution, Symmetric Formula
        res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
              + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
              * self.cubic_kernel_derivative(r)
        return res

    @ti.func
    def solid_fluid_force(self, p_i, p_j, r):
        # p_i 是液体粒子，p_j 是固体粒子，计算液体受到固体的惩罚力
        h = self.ps.support_radius
        k1 = 2e5
        k2 = self.ps.density[p_j] / (self.ps.density[p_i] + self.ps.density[p_j])
        k = k1 * k2

        r_norm = r.norm()
        q = r_norm / h
        k /= r_norm ** 2

        res = ti.Vector([0.0 for _ in range(self.ps.dim)])

        if q <= 1.0:
            if q <= 1/3:
                res = 2/3 * k * r
            elif q <= 1/2:
                q2 = q * q
                res = (4.0 * q - 6.0 * q2) * k * r
            else:
                res = 2.0 * ti.pow(1 - q, 2.0) * k * r
        return res

    def substep(self):
        pass

    def step(self):
        self.ps.initialize_particle_system()
        self.substep()
