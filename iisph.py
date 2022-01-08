from os import X_OK
from numpy import pi
import taichi as ti
from sph_base import SPHBase

class IISPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # self.exponent = 7.0
        # self.stiffness = 50.0

        # self.method = 1

        self.density_avg = ti.field(dtype=float, shape=())

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)

        # d_ii
        self.eye = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node.place(self.eye)

        # sum(d_ij * p_j) for i
        self.count = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node.place(self.count)

        # a_ii
        self.a = ti.field(dtype=float)
        particle_node.place(self.a)

        self.density_adv = ti.field(dtype=float)
        particle_node.place(self.density_adv)

        self.myPressure = ti.field(dtype=float)
        particle_node.place(self.myPressure)

        self.fluid_particle_num = ti.field(dtype=int, shape=())
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.fluid_particle_num[None] += 1


    @ti.kernel
    def compute_densities(self):
        # 计算密度和d_ii
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            d_eye = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_eye -= self.mass *self.cubic_kernel_derivative(x_i - x_j)
                self.ps.density[p_i] += self.mass * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            d_eye *= (self.dt[None] ** 2) / (self.ps.density[p_i] ** 2)
            self.eye[p_i] = d_eye

    @ti.kernel
    def solve_pressure(self):
        # 迭代求解压强

        # if self.method == 0:
        #     for p_i in range(self.ps.particle_num[None]):
        #         if self.ps.material[p_i] != self.ps.material_fluid:
        #             continue
        #         self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        # else:
        step = 0
        eps = 100
        flag = True
        while (flag and step < 10) or step < 2:
            self.density_avg[None] = 0.0
            for p_i in range(self.ps.particle_num[None]):
                if self.ps.material[p_i] != self.ps.material_fluid:
                    continue
                x_i = self.ps.x[p_i]
                d_c = ti.Vector([0.0 for _ in range(self.ps.dim)])
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    d_ij = -(self.dt[None] ** 2) * self.mass\
                        * self.cubic_kernel_derivative(x_i - x_j) / (self.ps.density[p_j] ** 2)
                    d_c += self.ps.pressure[p_j] * d_ij
                self.count[p_i] = d_c
            for p_i in range(self.ps.particle_num[None]):
                if self.ps.material[p_i] != self.ps.material_fluid:
                    continue
                x_i = self.ps.x[p_i]
                density3 = 0.0
                d_density = 0.0
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    d_ji = -(self.dt[None] ** 2) * self.mass * self.cubic_kernel_derivative(x_j - x_i) / (self.ps.density[p_i] ** 2)
                    density3 += self.mass * (self.count[p_i] + d_ji * self.ps.pressure[p_i] - self.eye[p_j] * self.ps.pressure[p_j] - self.count[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
                d_density = density3 + self.a[p_i] * self.ps.pressure[p_i]
                self.myPressure[p_i] = 0.5 * self.ps.pressure[p_i] + 0.5 * (self.density_0 - self.density_adv[p_i] - density3) / self.a[p_i]                    
                densityNew = d_density + self.density_adv[p_i]
                if densityNew < self.density_0:
                    self.density_avg[None] += self.density_0
                    self.myPressure[p_i] = 0
                else:
                    assert self.myPressure[p_i] > 0.0
                    self.density_avg[None] += densityNew

            self.density_avg[None] /= self.fluid_particle_num[None]

            for p_i in range(self.ps.particle_num[None]):
                if self.ps.material[p_i] != self.ps.material_fluid:
                    continue
                self.ps.pressure[p_i] = self.myPressure[p_i]
            flag = self.density_avg[None] - self.density_0 >= 0 and self.density_avg[None] - self.density_0 < eps
            flag = ~flag
            step += 1

            # for p_i in range(self.ps.particle_num[None]):
            #     if self.ps.material[p_i] == self.ps.material_fluid:
            #         print(self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0), self.ps.pressure[p_i])
                
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                # Compute Pressure force contribution
                d_v += self.pressure_force(p_i, p_j, x_i-x_j)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            # Add body force
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[self.ps.dim-1] = self.g
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
                if self.ps.material[p_j] == self.ps.material_boundary:
                    d_v += self.solid_fluid_force(p_i, p_j, x_i - x_j)                    
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def advect(self):
        # 更新v和x
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]
    
    @ti.kernel
    def preAdvect(self):
        # 计算v_adv和density_adv, 初始化压强, 计算a_ii
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.ps.pressure[p_i] *= 0.5
            self.a[p_i] = 0.0
            d_density_adv = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_density_adv += self.mass * (self.ps.v[p_i] - self.ps.v[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
                d_ji = -(self.dt[None] ** 2) * self.mass * self.cubic_kernel_derivative(x_j - x_i) / (self.ps.density[p_i] ** 2)
                self.a[p_i] += self.mass * (self.eye[p_i] - d_ji).dot(self.cubic_kernel_derivative(x_i - x_j))
            if abs(self.a[p_i]) < 0.000001:
                if self.a[p_i] < 0:
                    self.a[p_i] = -0.000001
                else:
                    self.a[p_i] = 0.000001
            self.density_adv[p_i] = self.ps.density[p_i] + self.dt[None] * d_density_adv
            self.density_adv[p_i] = ti.max(self.density_adv[p_i], self.density_0)

    def subsubstep(self):
        self.solve_pressure()
        self.compute_pressure_forces()
        self.advect()

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.preAdvect()
        self.subsubstep()
