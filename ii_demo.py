import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from iisph import IISPHSolver

# ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
ti.init(arch=ti.gpu, device_memory_GB=3, packed=True, debug=True)


if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    ps.add_cube(lower_corner=[6, 2],
                cube_size=[3.0, 5.0],
                velocity=[-5.0, -10.0],
                density=1000.0,
                color=0x00FFFF,
                material=1)

    ps.add_cube(lower_corner=[4, 1],
                cube_size=[1.5, 6.0],
                velocity=[0.0, -20.0],
                density=1000.0,
                color=0x00FFFF,
                material=1)

    ps.add_cube(lower_corner=[1.5, 0.5],
                cube_size=[1.5, 0.5],
                color=0x1E90FF,
                material=0)

    ps.add_cube(lower_corner=[2.5, 2.0],
                cube_size=[1.0, 0.5],
                color=0x1E90FF,
                material=0)

    ps.add_cube(lower_corner=[1.0, 2.0],
                cube_size=[1.0, 0.5],
                color=0x1E90FF,
                material=0)

    ps.add_boundary(boundaryColor=0x1E90FF)

    # ps.add_boundary(boundaryColor=0xFFFFFF)

    iisph_solver = IISPHSolver(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        for i in range(5):
            iisph_solver.step()
        particle_info = ps.dump()
        gui.rect(np.array([1.5-ps.support_radius/2, 0.5-ps.support_radius/2]) * ps.screen_to_world_ratio / 512, 
                np.array([3.0+ps.support_radius/2, 1.0+ps.support_radius/2]) * ps.screen_to_world_ratio / 512, 
                radius=2, color=0x1C1C1C)
        gui.rect(np.array([2.5-ps.support_radius/2, 2.0-ps.support_radius/2]) * ps.screen_to_world_ratio / 512, 
                np.array([3.5+ps.support_radius/2, 2.5+ps.support_radius/2]) * ps.screen_to_world_ratio / 512, 
                radius=2, color=0x1C1C1C)
        gui.rect(np.array([1.0-ps.support_radius/2, 2.0-ps.support_radius/2]) * ps.screen_to_world_ratio / 512, 
                np.array([2.0+ps.support_radius/2, 2.5+ps.support_radius/2]) * ps.screen_to_world_ratio / 512, 
                radius=2, color=0x1C1C1C)
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=particle_info['color'])
        gui.show()