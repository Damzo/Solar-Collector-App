import imp
from tabnanny import check
import numpy as np
import sympy as spy
from rotationVectU import rotationVectU as rotVU
from decimal import *
from typing import Any, Union, Optional
getcontext().prec = 80
x, y, z = spy.symbols('x y z')

class cylindricalCollector:
    """
    This class contains all methods required to compute focus profile of a sun light collector
    The collector has a cylinder shape of equation type |vect(OM) x vect(MU)|-R.|vect(OU)|=0 
    The collector is delimited along (oy) axis by it's length L and along (oz) axis by the half of it's height h
    and in the (ox) axis by it's thickness th, with an additional freedom degree of rotation by angle 'khoi' respect to axe (ox)
    The center of the coordinate system is the center of the cylinder
    The revolution axis of the cylinder is defined by a vector vect(OU) which coordinates are x0, y0 and z0
    the sun position is defined by point S[xs, ys, zs]
    """

    # initialization of each instance of the class
    def __init__(self, surface: tuple, revol_axis: tuple, sun_pos: tuple, khoi = 0.0, z_0 = 0.0):
        """
        :type z_0: Z up step of the cylinder (for x=0, y=0)
        :type khoi: float value to define the rotation angle of the parabola respect to (OX) axis
        :type sun_pos: a tuple of 3 values (xs, ys, zs) to define the position of the sun compared to the parabola
        :type surface: a tuple of 3 values (L, th, h) to define the parabolic surface equation
        :type rev_axis: a tuple of 3 values (x0, y0, z0) to define the revolution vector of the cylinder

        """

        self.khoi = khoi
        self.sun_pos = np.array(sun_pos)
        self.surface = np.array(surface)
        self.z_0 = z_0
        # Tolerence in the diameter (considering that the edge of the parabola is critical to use)
        self.edge_tol = 1e-3
        # Cylinder surface parameters
        self.revol_axis = np.array(revol_axis) / (revol_axis[0]**2 + revol_axis[1]**2 + revol_axis[2]**2)**0.5
        self.length = surface[0]
        self.thickness = surface[1] - self.edge_tol
        self.height = surface[2]

        # Rotation matrix
        self.rot_y = rotVU([0, 1, 0], khoi)
        # Z coordinate of the higher point of the cylindrical collector without rotation
        self.z_max = -self.z_0 # surface[2] - self.z_0
        # Parabola surface expression to use
        self.surf_implicit_equ = self.symbolic_cylinder_equation()
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
    def update_init(self, surface: tuple, sun_pos: tuple, revol_axis = (0, 1, 0), khoi = 0.0, z_0 = 0.0):
        """
        :type khoi: float value to define the rotation angle of the parabola respect to (OX) axis
        :type sun_pos: a tuple of 3 values (xs, ys, zs) to define the position of the sun compared to the parabola
        :type surface: a tuple of 3 values (a, b, D) to define the parabolic surface equation Z = a*X² + b*Y²,
        with D the aperture diameter of the parabola
        """
        import numpy as np
        inputdatas = {'surface': np.array(surface),'revol_axis': np.array(revol_axis), 'sun_pos': np.array(sun_pos), 'khoi': khoi, 'z_0': z_0}
        default_inputdatas = {'surface': self.surface, 'revol_axis': self.revol_axis, 'sun_pos': self.sun_pos, 'khoi': self.khoi, 'z_0': self.z_0}
        for name, val in inputdatas.items():
            # print('name is', name, 'and value is', val)
            if val is None:
                inputdatas[name] = default_inputdatas[name]

        self.__init__(**inputdatas)

    def compute_material_surface(self, minor_axis: Union[float, np.ndarray], major_axis: Union[float, np.ndarray],
                     length: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        :return: area of the lateral cylinder, same type as inputs. r and h should have the same shape.
        :type minor_axis: cylinder minor axis length (half of the minor vetex), float or numpy array
        :type major_axis: cylinder major axis length (half of the minor vetex), float or numpy array
        :type length: cylinder lateral length, float or numpy array
        """
        if isinstance(minor_axis, np.ndarray):
            if not (np.array_equal(minor_axis.shape, major_axis.shape) &
                    np.array_equal(minor_axis.shape, length.shape) &
                    np.array_equal(length.shape, major_axis.shape) ):
                raise Exception("Minor axis, major axis and length should be numpy array of the same dimension or float")

        a = np.pi * np.sqrt((minor_axis**2 + major_axis**2) / 2)
        area = a * length

        return area
    
    def collection_area(self):
        # we should consider the angle of collection in future versions...
        a = self.length * self.thickness
        
        return a


    def symbolic_cylinder_equation(self):
        x, y, z = spy.symbols('x y z')
        u = spy.Matrix([x, y, z - self.z_0])
        rot = spy.Matrix(self.rot_y)
        u_rot = rot*u
        MU_vec = u_rot - spy.Matrix(self.revol_axis)
        # cylinder elliptical section prameters (th is the major axis and h is the minor axis)
        # by default the calculations are made for a revolution axis of the cylinder is (0, 1, 0)
        th = self.thickness/2
        h = self.height
        temp = spy.sqrt(th**2-x**2)
        corde = spy.Matrix([x, 0, h/th*temp]) # default point coordinates on the elliptical section
        # compute rotation angles in case the revolution axis the user gives is not (0, 1, 0)
        theta = np.pi/2 - np.arccos(self.revol_axis[2])
        if (self.revol_axis[0]==0):
            phi = 0
        else :
            phi = np.arctan(self.revol_axis[1]/self.revol_axis[0])
        
        # rotate default corde according to the position of the asked revolution axis
        rot_phi = spy.Matrix(rotVU([0, 0, 1], phi)) # rotation of angle phi around (oz) axis
        rot_theta = spy.Matrix(rotVU([1, 0, 0], theta)) # rotation of angle theta around (ox) axis
        corde_r = rot_phi * corde
        corde_rr = corde # rot_theta * corde_r
        # distance from each point of the surfaceto the revolution axis is then
        # d_len = 1
        d_len = spy.simplify( spy.sqrt(corde_rr[0]**2 + corde_rr[1]**2 + corde_rr[2]**2) )
        # Equation 
        a = u_rot.cross(MU_vec)
        a_norm = spy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        b_norm = 1.0 # spy.sqrt(self.revol_axis[0]**2 + self.revol_axis[1]**2 + self.revol_axis[2]**2)
        
        func = spy.simplify( a_norm - d_len * b_norm )

        return func

    def solarPointSource_ray_equation(self):
        """
        method to calculate the incident ray equation depending on spherical coordinate (theta, phi)
        Approximation here: sun is a point source at the distance of "sun_pos" defined in the init
        :returns x = f(z), y = f(z)
        """
        z, theta, phi, x_equ, y_equ = spy.symbols('z theta phi x_equ y_equ')
        # unit vector of the incident rays
        u_inc = [spy.sin(theta) * spy.cos(phi), spy.sin(theta) * spy.sin(phi), -spy.cos(theta)]
        # Equations
        y_equ = (z - self.sun_pos[2]) / u_inc[2] * u_inc[0] + self.sun_pos[0]
        x_equ = (z - self.sun_pos[2]) / u_inc[2] * u_inc[1] + self.sun_pos[1]

        return x_equ, y_equ
    
    def arbitrary_ray_equation(unit_vec: np.ndarray):
        
        z, x_equ, y_equ = spy.symbols('z x y')
        y_equ = z / unit_vec[2] * unit_vec[0]
        x_equ = z / unit_vec[2] * unit_vec[1]
        
        return x_equ, y_equ

    def symbolic_incident_ray_intersection(self, inc_raysX:spy.Function, inc_raysY: spy.Function, surface: spy.Function):
        """
        method to calculate the intersection equation
        :param inc_raysX incident rays symblic expression x=f(theta, phi, z)
        :param inc_raysY incident rays symblic expression y=f(theta, phi, z)
        :param surface: the surface equation of type Sympy function f(x,y,z) = 0
        :return: a second order polynomial f(z, theta, phi) = 0
        """
        x, xp, y, yp, z, theta, phi = spy.symbols('x xp y yp z theta phi')
        # Incident ray line equation
        xp, yp = inc_raysX, inc_raysY

        # Replace in parabola equation
        intersection = surface.subs({x: xp, y:yp})

        return spy.simplify(intersection)

    def solve_incident_intersection(self, func: spy.Function, theta_val:float, phi_val:float):
        """
        Solve the intersection equation for each theta and phi angles value
        func is the equation to be solved ( output of symbolic_incident_ray_intersection)
        """
        z, theta, phi = spy.symbols('z theta phi')
        equ_eval = spy.Eq(func.subs({theta: theta_val, phi: phi_val}),0)
        ans = spy.solve(equ_eval, z, check=False)

        return np.asarray(ans, dtype=np.float)


    def incident_unit_vec(self, theta, phi):
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = -np.cos(theta)
        return np.array([x, y, z ])

    def incident_ray(self, inc_raysX:spy.Function, inc_raysY: spy.Function, theta_val: Union[float, np.ndarray],
                     phi_val: Union[float, np.ndarray], z_val:Union[float, np.ndarray]):
        """
        Compute line equation of the incident ray
        func is the symbolic sympy function of the incident ray equation
        """
        # vec = self.incident_unit_vec(theta_val, phi_val)
        # x = (z_val-self.sun_pos[2])/vec[2] * vec[0] + self.sun_pos[0]
        # y = (z_val-self.sun_pos[2])/vec[2] * vec[1] + self.sun_pos[1]

        z, theta, phi = spy.symbols('z theta phi')
        # x = inc_raysX.subs({theta: theta_val, phi: phi_val, z: z_val})
        # y = inc_raysX.subs({theta: theta_val, phi: phi_val, z: z_val})
        x = spy.lambdify((theta, phi, z), inc_raysX, 'numpy')
        y = spy.lambdify((theta, phi, z), inc_raysY, 'numpy')

        return x(theta_val, phi_val, z_val), y(theta_val, phi_val, z_val), z_val

    def cylinder_aperture_rectangular_section(self, phi: float):
        """
        This function defines the cylinder aperture function (a rectangular section)
        :param phi: angle phi value
        :return: numpy vector [x,y,z]
        """
        L = self.length
        d = self.thickness - 1e-3
        phi_1 = np.arctan(d/L)
        phi_2 = np.pi - phi_1
        phi_3 = np.pi + phi_1
        phi_4 = 2 * np.pi - phi_1
        vec = np.array([0, 0, 0])
        
        if (phi>=0.0) & (phi<=phi_1):
            vec = np.array([L/2 * np.tan(phi), L/2, self.z_max])
        elif (phi>phi_1) & (phi<=phi_2):
            vec = np.array([d/2, d / (2*np.tan(phi)), self.z_max])
        elif (phi>phi_2) & (phi<=phi_3):
            vec = np.array([-L/2 * np.tan(phi), -L/2, self.z_max])
        elif (phi>phi_3) & (phi<=phi_4):
            vec = np.array([-d/2, -d / (2*np.tan(phi)), self.z_max])
        elif phi>phi_4:
            vec = np.array([L/2 * np.tan(phi), L/2, self.z_max])
        
        return vec

    def cylinder_aperture_theta_limit(self, phi):

        O = np.dot(self.rot_y, np.array([0, 0, self.z_max]))
        M = self.cylinder_aperture_rectangular_section(phi)
        S = self.sun_pos
        OM = M-O
        a = Decimal(OM[0]**2 + OM[1]**2 + OM[2]**2).sqrt()
        SO = O-S
        b = (Decimal(SO[0])**2 + Decimal(SO[1])**2 + Decimal(SO[2])**2).sqrt()
        SM = M-S
        c = (Decimal(SM[0])**2 + Decimal(SM[1])**2 + Decimal(SM[2])**2).sqrt()
        dot_product = Decimal(SO[0])*Decimal(SM[0]) + Decimal(SO[1])*Decimal(SM[1]) + Decimal(SO[2])*Decimal(SM[2])
        cos_val = Decimal(dot_product) / Decimal(b*c)
        # theta_lim = np.arccos( float(cos_val) )
        theta_lim = Decimal(2*(1-cos_val)).sqrt()

        return theta_lim

    def symbolic_gradients(self, func: spy.Function):
        
        df_x = spy.diff(func, x)
        df_y = spy.diff(func, y)
        df_z = spy.diff(func, z)

        return df_x, df_y, df_z

    def surf_normal_unit_vec(self, xv, yv, zv):
        
        vec = np.zeros(3)
        # t1 = self.grad_x_lambda(xv,yv,zv)
        # t2 = self.grad_y_lambda(xv,yv,zv)
        # t3 = self.grad_z_lambda(xv,yv,zv)
        vec[:] = [self.grad_x_lambda(xv,yv,zv), self.grad_y_lambda(xv,yv,zv), self.grad_z_lambda(xv,yv,zv)]

        u_x = self.grad_x_lambda(xv, yv, zv) / np.linalg.norm(vec)
        u_y = self.grad_y_lambda(xv, yv, zv) / np.linalg.norm(vec)
        u_z = self.grad_z_lambda(xv, yv, zv) / np.linalg.norm(vec)

        return np.array([u_x, u_y, u_z])

    def reflected_unit_vec(self, theta_v: Union[float, np.ndarray], phi_v: Union[float, np.ndarray]):
        theta, phi = spy.symbols('theta phi')
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

    def reflected_ray(self, theta_v: Union[float, np.ndarray], phi_v: Union[float, np.ndarray], z: Union[float, np.ndarray] ):
        u_r, x_inters, y_inters, z_inters = self.reflected_unit_vec(theta_v, phi_v)
        x = (z - z_inters) / u_r[2] * u_r[0] + x_inters
        y = (z - z_inters) / u_r[2] * u_r[1] + y_inters

        return np.array(x), np.array(y), np.array(z)

