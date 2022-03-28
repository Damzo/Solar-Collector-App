from parabolicCollector import parabolicCollector as pmc

from decimal import *
getcontext().prec = 80

import numpy as np
import ipyvolume as ipv
import sympy as spy



class plot_geometry:
    
    def __init__(self, fx, fy, h, pt_source_pos, khoi, Rin_0, A_target, N, w, h_max, L, h_cyl, th,
                 n_theta, n_phi,
                 center_instance: object, left_instance: object, parabolic: object, ring:object, cylinder:object) \
            -> None:

        self.fx = fx
        self.fy = fy
        self.h = h
        self.pt_source_pos = pt_source_pos
        self.khoi = khoi
        self.Rin_0 = Rin_0 
        self.A_target = A_target 
        self.N = N 
        self.w = w 
        self.h_max = h_max
        self.L = L
        self.h_cyl = h_cyl
        self.th = th
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.center_instance = center_instance
        self.left_instance = left_instance
        self.parabolic = parabolic
        self.ring = ring
        self.cylinder = cylinder


    def update_init(self, fx = None, fy = None, h = None, pt_source_pos = None, khoi = None, Rin_0 = None, 
                    L = None, h_cyl = None, th = None, A_target = None, N = None, w = None, h_max = None,
                    n_theta = None, n_phi = None, center_instance = None,
                    left_instance = None, parabolic = None, ring = None, cylinder = None):

        input_data = {'fx':fx, 'fy':fy, 'h':h, 'khoi':khoi, 'L':L, 'h_cyl':h_cyl, 'th':th,
                      'Rin_0':Rin_0, 'A_target':A_target, 'N':N, 
                      'w':w, 'h_max':h_max, 'pt_source_pos':pt_source_pos, 
                      'n_theta':n_theta, 'n_phi':n_phi, 'center_instance':center_instance, 'left_instance':left_instance,
                      'parabolic':parabolic, 'ring':ring, 'cylinder':cylinder}
        
        default_input_data = {'fx':self.fx, 'fy':self.fy, 'h':self.h, 'L':self.L, 'h_cyl':self.h_cyl, 'th':self.th,
                              'khoi':self.khoi,'Rin_0':self.Rin_0, 'A_target':self.A_target, 
                              'N':self.N,'w':self.w, 'h_max':self.h_max, 
                              'pt_source_pos':self.pt_source_pos, 'n_theta':self.n_theta, 'n_phi':self.n_phi,
                              'center_instance':self.center_instance,'left_instance':self.left_instance,
                              'parabolic':self.parabolic, 'ring':self.ring, 'cylinder':self.cylinder}
        
        for name, val in input_data.items():
            if val is None:
                input_data[name] = default_input_data[name]

        self.__init__(**input_data)
    
    
    def plot_parabola(self):
        
        ipv.figure(self.center_instance.main_scene)
        self.center_instance.main_scene.meshes.clear()
        self.center_instance.main_scene.scatters.clear()
        # General parameters
        self.parabolic.update_init(surface=(self.fx, self.fy, self.h),sun_pos=self.pt_source_pos, khoi=self.khoi)
        phi_tab = np.linspace(0, 2*np.pi,self.n_phi, endpoint=False)
        z_tab = np.linspace(0, 2*self.h)
        rx = self.parabolic.diameter_x/2
        ry = self.parabolic.diameter_y/2

        # Compute the graphical extent after the rotation
        rot_lim_1 = np.dot(self.parabolic.rot_x, (0, -ry, 0))
        rot_lim_2 = np.dot(self.parabolic.rot_x, (0, ry, 0))
        rot_lim_3 = np.dot(self.parabolic.rot_x, (0, -ry, self.h))
        rot_lim_4 = np.dot(self.parabolic.rot_x, (0, ry, self.h))
        yl_min = np.min([-rot_lim_1[1], -rot_lim_2[1], -rot_lim_3[1], -rot_lim_4[1]])
        yl_max = np.max([-rot_lim_1[1], -rot_lim_2[1], -rot_lim_3[1], -rot_lim_4[1]])
        zl_min = np.min([rot_lim_1[2], rot_lim_2[2], rot_lim_3[2], rot_lim_4[2]])
        zl_max = np.max([rot_lim_1[2], rot_lim_2[2], rot_lim_3[2], rot_lim_4[2]])
        
        # Compute parabola surface equation
        x, y, z, theta, phi = spy.symbols('x y z theta phi')
        equ = self.parabolic.symbolic_parabola_equation()
        parabola_equ = spy.lambdify((x, y, z), equ, 'numpy')
        xx, yy, zz = np.meshgrid(np.linspace(-rx, rx), np.linspace(yl_min, yl_max), np.linspace(zl_min, zl_max), indexing='ij')
        # xx, yy, zz = np.meshgrid(np.linspace(-rx, rx), np.linspace(-ry, ry), np.linspace(0, h))
        
        values = parabola_equ(xx, yy, zz)
        z_rot = np.zeros(xx.shape)
        for ii in np.arange(50):
            for jj in np.arange(50):
                for kk in np.arange(50):
                    z_rot[ii,jj,kk] = np.dot(self.parabolic.rot_x, (xx[ii,jj,kk], yy[ii,jj,kk], zz[ii,jj,kk]))[2]

        values[z_rot>=self.h] = 0.1
        
        # Compute intersection point equation
        inters = self.parabolic.inters_equ
        inters_lambda = spy.lambdify((z, theta, phi), inters)

        surf = ipv.plot_isosurface(values, level=0, extent=[[-rx, rx], [yl_min, yl_max], [zl_min, zl_max]])
        # surf = ipv.plot_isosurface(values, level=0, extent=[[-rx, rx], [-ry, ry], [0, h]])
        #ipv.style.axes_off()
        ipv.xlim(-rx, rx)
        ipv.ylim(yl_min, yl_max)
        ipv.zlim(zl_min, 2*zl_max)
        
        return phi_tab, z_tab, rx, ry, inters, inters_lambda, yl_min, yl_max, zl_min, zl_max

    
    def plot_ringArray(self):
        
        parabola_rings=[]
        h_2D=np.array([[]])
        focs=np.array([])
    
        ipv.figure(self.center_instance.main_scene)
        self.center_instance.main_scene.meshes.clear()
        self.center_instance.main_scene.scatters.clear()
        # General parameters
        self.ring.update_init(Rin_0=self.Rin_0, A_target=self.A_target, N=self.N, w=self.w, h_max=self.h_max)
        phi_tab = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        # Make optimization
        sol = self.ring.optimize()
        G = sol['G']
        f0 = sol['f0']
        focs = sol['focal']
        N = int(sol['N'])
        R_2D = sol['Rays']
        h_2D = sol['Heights']
        rms = sol['RMS']
        
        if rms<10:
            z_0 = 2*h_2D[0, 1]
            for i in np.arange(N):
                parabola_rings.append(pmc((focs[i], focs[i], h_2D[i, 1]), self.pt_source_pos, 0.0, z_0=focs[0]-focs[i]))
                z_min = h_2D[i, 0]
                z_max = h_2D[i, 1]
                # rx = parabola_rings[i].diameter_x/2
                # ry = parabola_rings[i].diameter_y/2
                rx = R_2D[i, 1]
                ry = R_2D[i, 1]
                z_tab = np.linspace(z_min, 2*z_max)
                # Compute parabola surface equation
                x, y, z, theta, phi = spy.symbols('x y z theta phi')
                equ = parabola_rings[i].symbolic_parabola_equation()
                parabola_equ = spy.lambdify((x, y, z), equ, 'numpy')
                xx, yy, zz = np.meshgrid(np.linspace(-rx, rx), np.linspace(-ry, ry), np.linspace(z_min, z_max), indexing='ij')
                values = parabola_equ(xx, yy, zz)
                # Plot ring array
                ipv.xlim(-rx, rx)
                ipv.ylim(-ry, ry)
                ipv.plot_isosurface(values, level=0, extent=[[-rx, rx], [-ry, ry], [z_min, z_max]])
                
                
                self.left_instance.result_label.value = 'Succeed RMS value = \n' + str(rms)
            ipv.zlim(0, z_0)
        else:
            self.left_instance.result_label.value = 'Failed with a RMS value = \n' + str(rms) 
            
        return parabola_rings, h_2D, focs

    def plot_cylinder(self):
        
        ipv.figure(self.center_instance.main_scene)
        self.center_instance.main_scene.meshes.clear()
        self.center_instance.main_scene.scatters.clear()
        # General parameters
        self.cylinder.update_init(surface=(self.L, self.th, self.h_cyl), sun_pos=self.pt_source_pos, khoi=self.khoi)
        phi_tab = np.linspace(0, 2*np.pi,self.n_phi, endpoint=False)
        z_tab = np.linspace(-self.h_cyl, self.h_cyl)
        rx = self.th/2
        ry = self.L/2
        rz = self.h_cyl

        # Compute the graphical extent after the rotation (Not working very well !!!!)
        rot_lim_1 = np.dot(self.cylinder.rot_y, (-rx, -ry, 0))
        rot_lim_2 = np.dot(self.cylinder.rot_y, (-rx, ry, 0))
        rot_lim_3 = np.dot(self.cylinder.rot_y, (0, -ry, -rz))
        rot_lim_4 = np.dot(self.cylinder.rot_y, (0, ry, -rz))
        yl_min = np.min([-rot_lim_1[1], -rot_lim_2[1], -rot_lim_3[1], -rot_lim_4[1]])
        yl_max = np.max([-rot_lim_1[1], -rot_lim_2[1], -rot_lim_3[1], -rot_lim_4[1]])
        zl_min = np.min([rot_lim_1[2], rot_lim_2[2], rot_lim_3[2], rot_lim_4[2]])
        zl_max = np.max([rot_lim_1[2], rot_lim_2[2], rot_lim_3[2], rot_lim_4[2]])
        
        # Compute cylinder surface equation
        x, y, z, theta, phi = spy.symbols('x y z theta phi')
        equ = self.cylinder.symbolic_cylinder_equation()
        cylinder_equ = spy.lambdify((x, y, z), equ, 'numpy')
        # xx, yy, zz = np.meshgrid(np.linspace(-rx, rx), np.linspace(yl_min, yl_max), np.linspace(zl_min, zl_max), indexing='ij')
        xx, yy, zz = np.meshgrid(np.linspace(-rx, rx), np.linspace(-ry, ry), np.linspace(-rz, 0), indexing='ij')
        
        values = cylinder_equ(xx, yy, zz)
        z_rot = np.zeros(xx.shape)
        for ii in np.arange(50):
            for jj in np.arange(50):
                for kk in np.arange(50):
                    z_rot[ii,jj,kk] = np.dot(self.cylinder.rot_y, (xx[ii,jj,kk], yy[ii,jj,kk], zz[ii,jj,kk]))[2]

        # values[z_rot>=self.h_cyl] = 0.1
        
        # Compute intersection point equation
        inters = self.cylinder.inters_equ
        inters_lambda = spy.lambdify((z, theta, phi), inters)

        # surf = ipv.plot_isosurface(values, level=0, extent=[[-rx, rx], [yl_min, yl_max], [zl_min, zl_max]])
        surf = ipv.plot_isosurface(values, level=0, extent=[[-rx, rx], [-ry, ry], [-rz, 0]])
        #ipv.style.axes_off()
        # ipv.xlim(-rx, rx)
        # ipv.ylim(yl_min, yl_max)
        # ipv.zlim(zl_min, zl_max+rz)
        ipv.xlim(-rx, rx)
        ipv.ylim(-ry, ry)
        ipv.zlim(-rz, rz)
        
        return phi_tab, z_tab, rx, ry, inters, inters_lambda, yl_min, yl_max, zl_min, zl_max