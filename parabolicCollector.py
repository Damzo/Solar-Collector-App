import numpy as np
import sympy as spy
from decimal import *
from typing import Union

getcontext().prec = 80
x, y, z = spy.symbols('x y z')


class parabolicCollector:
    """
    This class contains all methods required to compute focus profile of a sunlight collector
    The collector has a parabolic shape of equation type Z = a*X² + b*Y²
    The collector is delimited by its diameter
    with an additional freedom degree of rotation by angle 'khoi' respect to axe (OX)
    The center of the coordinate system is the vertex of the parabola
    the sun position is defined by point S[xs, ys, zs]
    """

    # initialization of each instance of the class
    def __init__(self, surface: tuple, sun_pos: tuple, khoi=0.0, z_0=0.0):
        """
        :return: none
        :type z_0: Z up step of the parabola (for x=0, y=0)
        :type khoi: float value to define the rotation angle of the parabola respect to (OX) axis
        :type sun_pos: a tuple of 3 values (xs, ys, zs) to define the position of the sun compared to the parabola
        :type surface: a tuple of 3 values (fx, fy, h) to define the parabolic surface equation
        Z = (1/4*fx)*X² + (1/4*fy)*Y² + z_0, with:
        (fx, fy) be the focus points related to axes X and Y for a collimated incident beam
        h be the height of the parabola respect to horizontal plane
        """

        self.khoi = khoi
        self.sun_pos = np.array(sun_pos)
        self.surface = np.array(surface)
        self.z_0 = z_0
        # Tolerence in the diameter (considering that the edge of the parabola is critical to use)
        self.edge_tol = 1e-3
        # Parabola diameters
        self.diameter_x = 2 * np.sqrt(self.surface[2] * 4 * self.surface[0]) - self.edge_tol
        self.diameter_y = 2 * np.sqrt(self.surface[2] * 4 * self.surface[1]) - self.edge_tol

        # Rotation matrix
        self.rot_x = np.array([[1., 0., 0.], [0., np.cos(khoi), -np.sin(khoi)], [0., np.sin(khoi), np.cos(khoi)]])
        # Z coordinate of the higher point of the parabolic collector without rotation
        self.z_max = surface[2] - self.z_0  # + surface[1]*surface[3]**2
        # Parabola surface expression to use
        self.surf_implicit_equ = self.symbolic_parabola_equation()
        # Surface gradients equations
        self.grad_x, self.grad_y, self.grad_z = self.symbolic_gradients(self.surf_implicit_equ)
        self.grad_x_lambda = spy.lambdify((x, y, z), self.grad_x, modules='numpy')
        self.grad_y_lambda = spy.lambdify((x, y, z), self.grad_y, modules='numpy')
        self.grad_z_lambda = spy.lambdify((x, y, z), self.grad_z, modules='numpy')
        # Incident ray and surface intersection equation
        self.x_equ, self.y_equ = spy.symbols('xequ xequ')
        self.x_equ, self.y_equ = self.solarPointSource_ray_equation()
        self.inters_equ = self.symbolic_incident_ray_intersection(self.x_equ, self.y_equ, self.surf_implicit_equ)

    # Update the variables if you want
    def update_init(self, surface: tuple, sun_pos: tuple, khoi=0.0, z_0=0.0):
        """
        :return: none
        :type khoi: float value to define the rotation angle of the parabola respect to (OX) axis
        :type sun_pos: a tuple of 3 values (xs, ys, zs) to define the position of the sun compared to the parabola
        :type surface: a tuple of 3 values (a, b, D) to define the parabolic surface equation Z = a*X² + b*Y²,
        with D the aperture diameter of the parabola
        :type z_0:
        """
        import numpy as npy
        inputdatas = {'surface': npy.array(surface), 'sun_pos': npy.array(sun_pos), 'khoi': khoi, 'z_0': z_0}
        default_inputdatas = {'surface': self.surface, 'sun_pos': self.sun_pos, 'khoi': self.khoi, 'z_0': self.z_0}
        for name, val in inputdatas.items():
            # print('name is', name, 'and value is', val)
            if val is None:
                inputdatas[name] = default_inputdatas[name]

        self.__init__(**inputdatas)

    @staticmethod
    def compute_material_surface(r: Union[float, np.ndarray], h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        :return: area of the parabola, same type as inputs. r and h should have the same shape
        :type h: parabola height, float or numpy array
        :type r: parabola ray, float or numpy array
        """
        if isinstance(r, np.ndarray):
            if not np.array_equal(r.shape, h.shape):
                raise Exception("r and h should be numpy array of the same dimension or float")

        a = np.pi * r / (6 * h ** 2)
        b = (r ** 2 + 4 * h ** 2) ** (3 / 2)
        area = a * (b - r ** 3)

        return area

    def collection_area(self):
        # we should consider the angle of collection in future versions...
        a = np.pi * self.diameter_x / 2 * self.diameter_y / 2

        return a

    @staticmethod
    def compute_sectionLength(r: Union[float, np.ndarray], f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        :return: the section length of the parabola, same type as inputs
        :param f: parabola focal length, float or numpy array
        :type r: parabola ray, float or numpy array
        """
        if isinstance(r, np.ndarray):
            if not np.array_equal(r.shape, f.shape):
                raise Exception("r and f should be numpy array of the same dimension or float")

        q = np.sqrt(f ** 2 + r ** 2)
        a = r * q / f
        b = f * np.log((r + q) / f)

        section = a + b

        return section

    def symbolic_parabola_equation(self):
        x_m, y_m, z_m = spy.symbols('x y z')
        u = spy.Matrix([x_m, y_m, z_m])
        rot = spy.Matrix(self.rot_x)
        u_rot = rot * u
        # Equation Z - (1/(4*fx))*X² - (1/(4*fy))*Y² - Z_0 = 0
        a = (1. / (4. * self.surface[0]))
        b = (1. / (4. * self.surface[1]))
        func = spy.simplify(u_rot[2] - a * u_rot[0] ** 2 - b * u_rot[1] ** 2 - self.z_0)

        return func

    def solarPointSource_ray_equation(self):
        """
        method to calculate the incident ray equation depending on spherical coordinate (theta, phi)
        Approximation here: sun is a point source at the distance of "sun_pos" defined in the init
        :returns x = f(z), y = f(z)
        """
        z_m, theta, phi, x_equ, y_equ = spy.symbols('z theta phi x_equ y_equ')
        # unit vector of the incident rays
        u_inc = [spy.sin(theta) * spy.cos(phi), spy.sin(theta) * spy.sin(phi), -spy.cos(theta)]
        # Equations
        x_equ = (z_m - self.sun_pos[2]) / u_inc[2] * u_inc[0] + self.sun_pos[0]
        y_equ = (z_m - self.sun_pos[2]) / u_inc[2] * u_inc[1] + self.sun_pos[1]

        return x_equ, y_equ

    @staticmethod
    def symbolic_incident_ray_intersection(inc_raysX: spy.Function, inc_raysY: spy.Function,
                                           surface: spy.Function):
        """
        method to calculate the intersection equation
        :param inc_raysX incident rays symblic expression x=f(theta, phi, z)
        :param inc_raysY incident rays symblic expression y=f(theta, phi, z)
        :param surface: the surface equation of type Sympy function f(x,y,z) = 0
        :return: a second order polynomial f(z, theta, phi) = 0
        """
        x_m, xp, y_m, yp, z_m, theta, phi = spy.symbols('x xp y yp z theta phi')
        # Incident ray line equation
        xp, yp = inc_raysX, inc_raysY

        # Replace in parabola equation
        intersection = surface.subs([(x_m, xp), (y_m, yp)])

        return spy.simplify(intersection)

    @staticmethod
    def solve_incident_intersection(func: spy.Function, theta_val: float, phi_val: float):
        """
        Solve the intersection equation for each theta and phi angles value
        func is the equation to be solved ( output of symbolic_incident_ray_intersection)
        """
        z_m, theta, phi = spy.symbols('z theta phi')
        equ_eval = spy.Eq(func.subs([(theta, theta_val), (phi, phi_val)]), 0)
        ans = spy.solve(equ_eval, z_m)

        return np.asarray(ans, dtype=np.float64)

    @staticmethod
    def incident_unit_vec(theta, phi):
        x_m = np.sin(theta) * np.cos(phi)
        y_m = np.sin(theta) * np.sin(phi)
        z_m = -np.cos(theta)
        return np.array([x_m, y_m, z_m])

    @staticmethod
    def incident_ray(inc_raysX: spy.Function, inc_raysY: spy.Function, theta_val: Union[float, np.ndarray],
                     phi_val: Union[float, np.ndarray], z_val: Union[float, np.ndarray]):
        """
        Compute line equation of the incident ray
        func is the symbolic sympy function of the incident ray equation
        """
        # vec = self.incident_unit_vec(theta_val, phi_val)
        # x = (z_val-self.sun_pos[2])/vec[2] * vec[0] + self.sun_pos[0]
        # y = (z_val-self.sun_pos[2])/vec[2] * vec[1] + self.sun_pos[1]

        z_m, theta, phi = spy.symbols('z theta phi')
        # x = inc_raysX.subs({theta: theta_val, phi: phi_val, z: z_val})
        # y = inc_raysX.subs({theta: theta_val, phi: phi_val, z: z_val})
        x_m = spy.lambdify((theta, phi, z_m), inc_raysX, 'numpy')
        y_m = spy.lambdify((theta, phi, z_m), inc_raysY, 'numpy')

        return x_m(theta_val, phi_val, z_val), y_m(theta_val, phi_val, z_val), z_val

    def parabola_aperture_conic_section(self, phi):
        """
        This function defines the parabola aperture function (a conic section)
        :param phi: angle phi value
        :return: numpy vector [x,y,z]
        """
        a = np.array([self.diameter_x / 2 * np.cos(phi), self.diameter_y / 2 * np.sin(phi), self.z_max])
        vec = np.dot(self.rot_x, a)

        return vec

    def parabola_aperture_theta_limit(self, phi):

        O_m = np.dot(self.rot_x, np.array([0, 0, self.z_max]))
        M = self.parabola_aperture_conic_section(phi)
        S = self.sun_pos
        # OM = M - O_m
        # a = Decimal(OM[0] ** 2 + OM[1] ** 2 + OM[2] ** 2).sqrt()
        SO = O_m - S
        b = (Decimal(SO[0]) ** 2 + Decimal(SO[1]) ** 2 + Decimal(SO[2]) ** 2).sqrt()
        SM = M - S
        c = (Decimal(SM[0]) ** 2 + Decimal(SM[1]) ** 2 + Decimal(SM[2]) ** 2).sqrt()
        dot_product = Decimal(SO[0]) * Decimal(SM[0]) + Decimal(SO[1]) * Decimal(SM[1]) + Decimal(SO[2]) * Decimal(
            SM[2])
        cos_val = Decimal(dot_product) / Decimal(b * c)
        # theta_lim = np.arccos( float(cos_val) )
        theta_lim = Decimal(2 * (1 - cos_val)).sqrt()

        return theta_lim

    @staticmethod
    def symbolic_gradients(func: spy.Function):
        df_x = spy.diff(func, x)
        df_y = spy.diff(func, y)
        df_z = spy.diff(func, z)

        return df_x, df_y, df_z

    def surf_normal_unit_vec(self, xv, yv, zv):
        vec = np.zeros(3)
        # t1 = self.grad_x_lambda(xv,yv,zv)
        # t2 = self.grad_y_lambda(xv,yv,zv)
        # t3 = self.grad_z_lambda(xv,yv,zv)
        vec[:] = [self.grad_x_lambda(xv, yv, zv), self.grad_y_lambda(xv, yv, zv), self.grad_z_lambda(xv, yv, zv)]

        u_x = self.grad_x_lambda(xv, yv, zv) / np.linalg.norm(vec)
        u_y = self.grad_y_lambda(xv, yv, zv) / np.linalg.norm(vec)
        u_z = self.grad_z_lambda(xv, yv, zv) / np.linalg.norm(vec)

        return np.array([u_x, u_y, u_z])

    def reflected_unit_vec(self, theta_v: Union[float, np.ndarray], phi_v: Union[float, np.ndarray]):
        # theta, phi = spy.symbols('theta phi')
        # incident ray unit vector coordinates
        u_i = self.incident_unit_vec(theta_v, phi_v)
        # calculation of the intersection point coordinates
        z_i = self.solve_incident_intersection(self.inters_equ, theta_v, phi_v)
        x_inters, y_inters, z_inters = self.incident_ray(self.x_equ, self.y_equ, theta_v, phi_v, np.min(z_i))
        # calculation of the normal vector coordinates
        n_vec = self.surf_normal_unit_vec(x_inters, y_inters, z_inters)
        # calculation of the reflected ray unit vector coordinates
        u_r = u_i - 2 * np.dot(u_i, n_vec) * n_vec

        return u_r, x_inters, y_inters, z_inters

    def reflected_ray(self, theta_v: Union[float, np.ndarray], phi_v: Union[float, np.ndarray],
                      z_m: Union[float, np.ndarray]):
        u_r, x_inters, y_inters, z_inters = self.reflected_unit_vec(theta_v, phi_v)
        x_m = (z_m - z_inters) / u_r[2] * u_r[0] + x_inters
        y_m = (z_m - z_inters) / u_r[2] * u_r[1] + y_inters

        return np.array(x_m), np.array(y_m), np.array(z_m)
