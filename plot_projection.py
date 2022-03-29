from parabolicCollector import parabolicCollector as pmc

from decimal import *
getcontext().prec = 80

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class plot_projection:
    
    def __init__(self, n_phi, n_theta, h_2D, focs, parabola_rings, N, pt_source_pos, inters,
                 center_instance: object, right_instance: object, parabolic: object, cylinder: object) -> None:
        
        self.n_phi = n_phi
        self.n_theta = n_theta
        self.N = N
        self.focs = focs
        self.h_2D = h_2D
        self.inters = inters
        self.parabola_rings = parabola_rings
        self.pt_source_pos = pt_source_pos
        self.center_instance = center_instance
        self.right_instance = right_instance
        self.parabolic = parabolic
        self.cylinder = cylinder
        
    def update_init(self, n_phi, n_theta, h_2D, focs, parabola_rings, N, pt_source_pos, inters,
                 center_instance: object, right_instance: object, parabolic: object, cylinder:object):
        
        input_data = {'n_phi':n_phi, 'n_theta':n_theta, 'h_2D':h_2D, 'focs':focs, 'pt_source_pos':pt_source_pos,
                      'parabola_rings':parabola_rings, 'N':N, 'center_instance':center_instance, 'inters':inters,
                      'right_instance': right_instance, 'parabolic':parabolic, 'cylinder':cylinder}
        
        default_input_data = {'n_phi':self.n_phi, 'n_theta':self.n_theta, 'h_2D':self.h_2D, 'focs':self.focs, 'pt_source_pos':self.pt_source_pos,
                              'parabola_rings':self.parabola_rings, 'N':self.N, 'center_instance':self.center_instance, 'inters':self.inters,
                              'right_instance': self.right_instance, 'parabolic':self.parabolic, 'cylinder':self.cylinder}
        for name, val in input_data.items():
            if val is None:
                input_data[name] = default_input_data[name]

        self.__init__(**input_data)
    
    def plot_proj_parabola(self):

        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        if plt.isinteractive:
            plt.ion()
        plt.figure(num=1)
        self.right_instance.proj_scene.clear(keep_observers=False)
        z0 = self.center_instance.focus_zText.value
        nbins = self.n_theta*self.n_phi
        xplan = np.zeros(self.n_phi*self.n_theta)
        yplan = np.zeros(self.n_phi*self.n_theta)
        jj=0
        for ii in np.arange(self.n_phi):
            phi_v = phi_tab[ii]
            theta_limit = float(self.parabolic.parabola_aperture_theta_limit(phi_v))
            for theta_v in np.linspace(0, theta_limit, self.n_theta):
                z_i = self.parabolic.solve_incident_intersection(self.inters, theta_v, phi_v)
                xplan[jj], yplan[jj], z_refl = self.parabolic.reflected_ray(theta_v, phi_v, z0)
                jj+=1

        xmin = np.min(xplan)
        xmax = np.max(xplan)
        ymin = np.min(yplan)
        ymax = np.max(yplan)
        xy = np.vstack((xplan, yplan))
        dxy = (xmax-xmin+1)/nbins * (ymax-ymin+1)/nbins

        xi, yi = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j]
        
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([xplan, yplan])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, xi.shape)
        Z = Z / np.max(Z)
        
        # plt.imshow(np.zeros(xi.shape), extent=[xmin, xmax, ymin, ymax])
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        # plt.plot(xplan, yplan, '.')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        
        return Z
        
    def plot_proj_ringArray(self):

        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        if plt.isinteractive:
            plt.ion()
        plt.figure(num=1)
        self.right_instance.proj_scene.clear(keep_observers=False)
        z0 = self.center_instance.focus_zText.value
        nbins = self.n_theta*self.n_phi
        xplan = np.zeros(self.N*self.n_phi*self.n_theta)
        yplan = np.zeros(self.N*self.n_phi*self.n_theta)
        
        internal_rings = []
        kk = 0
        for i in np.arange(self.N):
            internal_rings.append(pmc((self.focs[i], self.focs[i], self.h_2D[i, 0]), 
                                      self.pt_source_pos, 0.0, z_0=self.focs[0]-self.focs[i]))
            # Compute intersection point equation
            inters = self.parabola_rings[i].inters_equ
            
            for ii in np.arange(self.n_phi):
                phi_v = phi_tab[ii]
                theta_limit_max = float(self.parabola_rings[i].parabola_aperture_theta_limit(phi_v))
                theta_limit_min = float(internal_rings[i].parabola_aperture_theta_limit(phi_v))
                theta_tab = np.linspace(theta_limit_min, theta_limit_max, self.n_theta, endpoint=False)
                for jj in np.arange(self.n_theta):
                    theta_v = theta_tab[jj]
                    xplan[kk], yplan[kk], z_refl = self.parabola_rings[i].reflected_ray(theta_v, phi_v, z0)
                    kk+=1

        xmin = np.min(xplan)
        xmax = np.max(xplan)
        ymin = np.min(yplan)
        ymax = np.max(yplan)
        xy = np.vstack((xplan, yplan))
        dxy = (xmax-xmin+1)/nbins * (ymax-ymin+1)/nbins

        xi, yi = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j]
        
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([xplan, yplan])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, xi.shape)
        Z = Z / np.max(Z)
        
        # plt.imshow(np.zeros(xi.shape), extent=[xmin, xmax, ymin, ymax])
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        # plt.plot(xplan, yplan, '.')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        
        return Z
        
        
    def plot_proj_cylinder(self):

        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        if plt.isinteractive:
            plt.ion()
        plt.figure(num=1)
        self.right_instance.proj_scene.clear(keep_observers=False)
        z0 = self.center_instance.focus_zText.value
        nbins = self.n_theta*self.n_phi
        xplan = np.zeros(self.n_phi*self.n_theta)
        yplan = np.zeros(self.n_phi*self.n_theta)
        jj=0
        for ii in np.arange(self.n_phi):
            phi_v = phi_tab[ii]
            theta_limit = float(self.cylinder.cylinder_aperture_theta_limit(phi_v))
            for theta_v in np.linspace(0, theta_limit, self.n_theta):
                z_i = self.cylinder.solve_incident_intersection(self.inters, theta_v, phi_v)
                xplan[jj], yplan[jj], z_refl = self.cylinder.reflected_ray(theta_v, phi_v, z0)
                jj+=1

        xmin = np.min(xplan)
        xmax = np.max(xplan)
        ymin = np.min(yplan)
        ymax = np.max(yplan)
        xy = np.vstack((xplan, yplan))
        dxy = (xmax-xmin+1)/nbins * (ymax-ymin+1)/nbins

        xi, yi = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j]
        
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([xplan, yplan])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, xi.shape)
        Z = Z / np.max(Z)
        
        # plt.imshow(np.zeros(xi.shape), extent=[xmin, xmax, ymin, ymax])
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        # plt.plot(xplan, yplan, '.')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        
        return Z
