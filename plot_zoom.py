from parabolicCollector import parabolicCollector as pmc

from decimal import *
getcontext().prec = 80

import numpy as np
import ipyvolume as ipv

class plot_zoom:
    
    def __init__(self, n_phi, n_theta, h_2D, focs, parabola_rings, N, pt_source_pos,
                 center_instance: object, right_instance: object, parabolic: object,
                 cylinder:object) -> None:
        self.n_phi = n_phi
        self.n_theta = n_theta
        self.h_2D = h_2D
        self.focs = focs
        self.parabola_rings = parabola_rings
        self.N = N
        self.pt_source_pos = pt_source_pos
        self.center_instance = center_instance
        self.right_instance = right_instance
        self.parabolic = parabolic
        self.cylinder = cylinder
        
    def update_init(self, n_phi, n_theta, h_2D, focs, parabola_rings, N, pt_source_pos,
                 center_instance: object, right_instance: object, parabolic: object, cylinder: object):
        
        input_data = {'n_phi':n_phi, 'n_theta':n_theta, 'h_2D':h_2D, 'focs':focs, 'pt_source_pos':pt_source_pos,
                      'parabola_rings':parabola_rings, 'N':N, 'center_instance':center_instance, 
                      'right_instance': right_instance, 'parabolic':parabolic, 'cylinder':cylinder}
        
        default_input_data = {'n_phi':self.n_phi, 'n_theta':self.n_theta, 'h_2D':self.h_2D, 'focs':self.focs, 'pt_source_pos':self.pt_source_pos,
                              'parabola_rings':self.parabola_rings, 'N':self.N, 'center_instance':self.center_instance,
                              'right_instance': self.right_instance, 'parabolic':self.parabolic, 'cylinder':self.cylinder}
        for name, val in input_data.items():
            if val is None:
                input_data[name] = default_input_data[name]

        self.__init__(**input_data)
        
    
    # Zoom on focus volume
    def plot_zoom_parabola(self):
        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        ipv.figure(self.right_instance.zoom_scene)
        self.right_instance.zoom_scene.scatters.clear()
        min_v = self.center_instance.focus_zoom.value[0]
        max_v = self.center_instance.focus_zoom.value[1]
        
        for ii in np.arange(self.n_phi):
            phi_v = phi_tab[ii]
            theta_limit = float(self.parabolic.parabola_aperture_theta_limit(phi_v))
            for theta_v in np.linspace(0, theta_limit, self.n_theta, endpoint=False):
                x_refl, y_refl, z_refl = self.parabolic.reflected_ray(theta_v, phi_v, np.linspace(min_v, max_v))
                ipv.scatter(x_refl, y_refl, z_refl, size=0.8)
                ipv.xlim(2*min(x_refl), 2*max(x_refl))
                ipv.ylim(2*min(y_refl), 2*max(y_refl))
        ipv.zlim(min_v, max_v)
        
    def plot_zoom_ringArray(self):
        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        min_v = self.center_instance.focus_zoom.value[0]
        max_v = self.center_instance.focus_zoom.value[1]
        internal_rings = []
        
        ipv.figure(self.right_instance.zoom_scene)
        self.right_instance.zoom_scene.scatters.clear()
        
        for i in np.arange(self.N):
            internal_rings.append(pmc((self.focs[i], self.focs[i], self.h_2D[i, 0]), 
                                                         self.pt_source_pos, 0.0, z_0=self.focs[0]-self.focs[i]))
            # rx = self.parabola_rings[i].diameter_x/2
            # ry = self.parabola_rings[i].diameter_y/2
            # Incident rays symbolic expression
            # xequ, yequ = self.parabola_rings[i].solarPointSource_ray_equation()
            # Compute intersection point equation
            # inters = self.parabola_rings[i].inters_equ
            
            for ii in np.arange(self.n_phi):
                phi_v = phi_tab[ii]
                theta_limit_max = float(self.parabola_rings[i].parabola_aperture_theta_limit(phi_v))
                theta_limit_min = float(internal_rings[i].parabola_aperture_theta_limit(phi_v))
                theta_tab = np.linspace(theta_limit_min, theta_limit_max, self.n_theta, endpoint=False)
                for jj in np.arange(self.n_theta):
                    theta_v = theta_tab[jj]
                    x_refl, y_refl, z_refl = self.parabola_rings[i].reflected_ray(theta_v, phi_v, np.linspace(min_v, max_v))
                    ipv.scatter(x_refl, y_refl, z_refl, size=0.8)
                    ipv.xlim(2*min(x_refl), 2*max(x_refl))
                    ipv.ylim(2*min(y_refl), 2*max(y_refl))
        ipv.zlim(min_v, max_v)
        
    
    def plot_zoom_cylinder(self):
        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        ipv.figure(self.right_instance.zoom_scene)
        self.right_instance.zoom_scene.scatters.clear()
        min_v = self.center_instance.focus_zoom.value[0]
        max_v = self.center_instance.focus_zoom.value[1]
        
        for ii in np.arange(self.n_phi):
            phi_v = phi_tab[ii]
            theta_limit = float(self.cylinder.cylinder_aperture_theta_limit(phi_v))
            for theta_v in np.linspace(0, theta_limit, self.n_theta, endpoint=False):
                x_refl, y_refl, z_refl = self.cylinder.reflected_ray(theta_v, phi_v, np.linspace(min_v, max_v))
                ipv.scatter(x_refl, y_refl, z_refl, size=0.8)
                ipv.xlim(2*min(x_refl), 2*max(x_refl))
                ipv.ylim(2*min(y_refl), 2*max(y_refl))
        ipv.zlim(min_v, max_v)
        