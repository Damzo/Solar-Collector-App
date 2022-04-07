from parabolicCollector import parabolicCollector as pmc

from decimal import *
getcontext().prec = 80

import numpy as np
import ipyvolume as ipv

class plot_reflected_rays:
    
    def __init__(self, n_phi, n_theta, inters, refl_rayClr_value, h_2D, focs, pt_source_pos, rx, yl_min,yl_max, 
                 parabola_rings, N, h, h_cyl,
                 center_instance: object, left_instance: object, parabolic: object, cylinder: object) -> None:

        self.n_phi = n_phi
        self.n_theta = n_theta
        self.inters = inters
        self.refl_rayClr_value = refl_rayClr_value
        self.h_2D = h_2D
        self.focs = focs
        self.pt_source_pos = pt_source_pos
        self.rx = rx
        self.yl_min = yl_min
        self.yl_max = yl_max
        self.parabola_rings = parabola_rings
        self.N = N
        self.h = h
        self.h_cyl = h_cyl
        self.center_instance = center_instance
        self.left_instance = left_instance
        self.parabolic = parabolic
        self.cylinder = cylinder
        

    def update_init(self, n_phi = None, n_theta = None, inters = None, refl_rayClr_value = None, h_2D = None, 
                    focs = None, pt_source_pos = None, rx = None, yl_min = None,yl_max = None, parabola_rings = None,
                    N = None, h = None, h_cyl = None, center_instance = None, left_instance = None, parabolic = None, cylinder = None):

        input_data = {'n_phi':n_phi, 'n_theta':n_theta, 'inters':inters, 'refl_rayClr_value':refl_rayClr_value, 'h_2D':h_2D, 'focs':focs,
                      'pt_source_pos':pt_source_pos,'rx':rx, 'yl_min':yl_min, 'yl_max':yl_max, 'parabola_rings':parabola_rings,
                      'N':N, 'h':h, 'h_cyl':h_cyl, 'center_instance':center_instance, 'left_instance':left_instance, 'parabolic':parabolic, 'cylinder':cylinder}
        
        default_input_data = {'n_phi':self.n_phi, 'n_theta':self.n_theta, 'inters':self.inters, 'refl_rayClr_value':self.refl_rayClr_value, 'h_2D':self.h_2D, 'focs':self.focs,
                      'pt_source_pos':self.pt_source_pos,'rx':self.rx, 'yl_min':self.yl_min, 'yl_max':self.yl_max, 'parabola_rings':self.parabola_rings,
                      'N':self.N, 'h':self.h, 'h_cyl':self.h_cyl, 'center_instance':self.center_instance, 'left_instance':self.left_instance, 'parabolic':self.parabolic, 'cylinder':self.cylinder}
        for name, val in input_data.items():
            if val is None:
                input_data[name] = default_input_data[name]

        self.__init__(**input_data)

    def plot_reflected_parabola(self):
        
        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        ipv.figure(self.center_instance.main_scene)
        self.center_instance.main_scene.scatters.clear()
        for ii in np.arange(self.n_phi):
            phi_v = phi_tab[ii]
            theta_limit = float(self.parabolic.parabola_aperture_theta_limit(phi_v))
            for theta_v in np.linspace(0, theta_limit*0.99, self.n_theta, endpoint=True):
                z_i = self.parabolic.solve_incident_intersection(self.inters, theta_v, phi_v)
                x_refl, y_refl, z_refl = self.parabolic.reflected_ray(theta_v, phi_v,
                                                            np.linspace(np.min(z_i), 3*self.h, 5))
                ipv.plot(x_refl, y_refl, z_refl, color=self.refl_rayClr_value)
        ipv.xlim(-self.rx, self.rx)
        ipv.ylim(self.yl_min, self.yl_max)
    
    def plot_reflected_ringArray(self):

        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        z_max = 2*self.h_2D[0, 1]
        internal_rings = []
        
        ipv.figure(self.center_instance.main_scene)
        self.center_instance.main_scene.scatters.clear()
        
        for i in np.arange(self.N):
            internal_rings.append(pmc((self.focs[i], self.focs[i], self.h_2D[i, 0]), self.pt_source_pos, 0.0, z_0=self.focs[0]-self.focs[i]))
            rx = self.parabola_rings[i].diameter_x/2
            ry = self.parabola_rings[i].diameter_y/2
            # Incident rays symbolic expression
            # xequ, yequ = self.parabola_rings[i].solarPointSource_ray_equation()
            # Compute intersection point equation
            inters = self.parabola_rings[i].inters_equ
            
            for ii in np.arange(self.n_phi):
                phi_v = phi_tab[ii]
                theta_limit_max = float(self.parabola_rings[i].parabola_aperture_theta_limit(phi_v))
                theta_limit_min = float(internal_rings[i].parabola_aperture_theta_limit(phi_v))
                theta_tab = np.linspace(theta_limit_min, theta_limit_max*0.99, self.n_theta, endpoint=True)
                for jj in np.arange(self.n_theta):
                    theta_v = theta_tab[jj]
                    z_i = self.parabola_rings[i].solve_incident_intersection(inters, theta_v, phi_v)
                    x_refl, y_refl, z_refl = self.parabola_rings[i].reflected_ray(theta_v, phi_v, np.linspace(-self.focs[i], np.min(z_i), 5))
                    ipv.plot(x_refl, y_refl, z_refl, color=self.refl_rayClr_value)
            ipv.xlim(-rx, rx)
            ipv.ylim(-ry, ry)
            ipv.zlim(0.0, z_max)
            
    
    def plot_reflected_cylinder(self):
        
        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        ipv.figure(self.center_instance.main_scene)
        self.center_instance.main_scene.scatters.clear()
        for ii in np.arange(self.n_phi):
            phi_v = phi_tab[ii]
            theta_limit = float(self.cylinder.cylinder_aperture_theta_limit(phi_v))
            for theta_v in np.linspace(0, theta_limit*0.99, self.n_theta, endpoint=True):
                z_i = self.cylinder.solve_incident_intersection(self.inters, theta_v, phi_v)
                x_refl, y_refl, z_refl = self.cylinder.reflected_ray(theta_v, phi_v,
                                                            np.linspace(np.min(z_i), 5*self.h_cyl, 5))
                ipv.plot(x_refl, y_refl, z_refl, color=self.refl_rayClr_value)
        ipv.xlim(-self.rx, self.rx)
        ipv.ylim(self.yl_min, self.yl_max)
        