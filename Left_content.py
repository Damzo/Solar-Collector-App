import ipywidgets as widgets
import numpy as np


class left_content:
    style = {'description_width': 'initial'}
    item_layout = widgets.Layout( width='auto')
    plot_surf_bt = widgets.Button(description='Plot Geometry', button_style='danger', style=style, icon='pencil-square-o',
                                      layout=item_layout)
    plot_inc_bt = widgets.Button(description='Plot Incident Rays', button_style='warning', style=style,
                                    layout=item_layout)
    plot_refl_bt = widgets.Button(description='Plot Reflected Rays', button_style='success', style=style,
                                    layout=item_layout)

    def __init__(self, center_instance:object, header_instance:object):
        
        self.c_object = center_instance
        self.h_object = header_instance
        self.focus_x = widgets.FloatText(
            value=0.5,
            description='Focal distance in X (fx):',
            disabled=False, style=self.style, layout=self.item_layout)
        self.focus_y = widgets.FloatText(
            value=0.5,
            description='Focal distance in Y (fy):',
            disabled=False, style=self.style, layout=self.item_layout)
        self.fy_eq_fx = widgets.Checkbox(
            value=False,
            description='fy = fx',
            disabled=False, layout=self.item_layout)

        self.height = widgets.FloatText(
            value=0.3,
            description='Height of the parabola',
            disabled=False, style=self.style, layout=self.item_layout)
        self.rot_angle = widgets.FloatText(
            value=0.0,
            description='Rotation angle in degrees',
            disabled=True, style=self.style, layout=self.item_layout)

        self.source_label = widgets.Label(value="Source position coordinates:")
        self.source_posX = widgets.FloatText(
            value=0.0,
            description='X ',
            disabled=False, style=self.style, layout=self.item_layout)
        self.source_posY = widgets.FloatText(
            value=0.0,
            description='Y ',
            disabled=False, style=self.style, layout=self.item_layout)
        self.source_posZ = widgets.FloatText(
            value=149597870.0,
            description='Z ',
            disabled=False, style=self.style, layout=self.item_layout)
        self.large_source_extend = widgets.FloatText(
            value=1e-1,
            description='Extend ',
            disabled=False, style=self.style, layout=self.item_layout)
        
        self.resolution = widgets.IntText(
            value=25,
            description='Number of rays to plot',
            disabled=False, style=self.style, layout=self.item_layout)

        self.Rin_0_widget = widgets.FloatText(
            value=0.1,
            description='Ring_0 internal ray:',
            disabled=False, style=self.style, layout=self.item_layout)
        self.N_widget = widgets.FloatText(
            value=2,
            description='Initial number of rings:',
            disabled=False, style=self.style, layout=self.item_layout)
        self.A_target_widget = widgets.FloatText(
            value=1.0,
            description='Targeted reflector area (m²):',
            disabled=False, style=self.style, layout=self.item_layout)
        self.w_widget = widgets.FloatText(
            value=0.1,
            description='Reflector material width (m):',
            disabled=False, style=self.style, layout=self.item_layout)
        self.h_max_widget = widgets.FloatText(
            value=1.0,
            description='Maximum height:',
            disabled=False, style=self.style, layout=self.item_layout)
        
        self.length = widgets.FloatText(
            value=1.0,
            description='Cylinder length:',
            disabled=False, style=self.style, layout=self.item_layout)
        self.thickness = widgets.FloatText(
            value=1.0,
            description='Cylinder thickness:',
            disabled=False, style=self.style, layout=self.item_layout)
        self.height_cyl = widgets.FloatText(
            value=1.0,
            description='Cylinder height:',
            disabled=False, style=self.style, layout=self.item_layout)
        
        # self.result_label = widgets.Label(value=r'\(\color{red} {' + 'Optimization\ results\ will\ be\ shown\ here'  + '}\)')
        self.result_label = widgets.HTML(value=" <h5 style='color:red;margin:auto; text-align:center; font-style'>Optimization results will be shown here</h5> ",)

        self.source = widgets.VBox([self.source_label, self.source_posX, self.source_posY, self.source_posZ])
        # self.source = widgets.VBox([self.source_label, self.source_posX, self.source_posY, self.source_posZ, self.large_source_extend])        
        
        # Variables to be output
            #General parameters
        self.pt_source_pos=(self.source_posX.value, self.source_posY.value, self.source_posZ.value)
        self.large_source_pos=(self.source_posX.value, self.source_posY.value, self.source_posZ.value, self.large_source_extend.value)
        self.n_theta = int(np.sqrt(self.resolution.value)) + 1
        self.n_phi = self.n_theta
        self.inc_rayClr_value = 'yellow'
        self.refl_rayClr_value = 'red'
            # Parabola
        self.fx = self.focus_x.value
        self.fy = self.focus_y.value
        self.h = self.height.value
        self.khoi = self.rot_angle.value/180 * np.pi
            # Ring array
        self.Rin_0 = self.Rin_0_widget.value
        self.A_target = self.A_target_widget.value
        self.N = int(self.N_widget.value)
        self.w = self.w_widget.value  
        self.h_max = self.h_max_widget.value
            # Cylinder
        self.L = self.length
        self.h_cyl = self.height_cyl
        self.th = self.thickness
        
    
    def update_variables(self):
        
        self.variables = {'pt_source_coordinate':self.pt_source_pos, 'n_theta':self.n_theta, 'n_phi':self.n_phi, 
                          'inc_rayClr_value':self.inc_rayClr_value, 'refl_rayClr_value':self.refl_rayClr_value, 
                          'parabola_focus_x':self.fx, 'parabola_focus_y':self.fy, 'parabola_height':self.h, 'parabola_rot_angle':self.khoi, 
                          'ringArray_internal_ray':self.Rin_0, 'ringArray_area':self.A_target, 'ringArray_N_rings':self.N, 
                          'ringArray_material_width':self.w, 'ringArray_h_max':self.h_max, 
                          'cylinder_length':self.L, 'cylinder_height':self.h_cyl, 'cylinder_thikness':self.th}
    

    def design_parabola(self):
        focus = widgets.VBox([self.focus_x, self.fy_eq_fx, self.focus_y])

        data2 = widgets.Box([self.source, self.resolution], layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px'
        ))
        data1_parabola = widgets.Box([focus, self.height, self.rot_angle], layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px'
        ))

        left_content = widgets.VBox([data1_parabola, data2, self.plot_surf_bt, self.plot_inc_bt, self.plot_refl_bt])
        
        self.__variables_parabola()
        self.__res()
        self.update_variables()
        
        # observes and links
        self.fy_eq_fx.observe(self.__fyEQfx, 'value')
        self.focus_x.observe(self.__variables_parabola,'value')
        self.focus_y.observe(self.__variables_parabola,'value')
        self.height.observe(self.__variables_parabola,'value')
        self.rot_angle.observe(self.__variables_parabola,'value')
        self.h_object.source_geometry.observe(self.__illumination_source,'value')
        self.resolution.observe(self.__res, 'value')
        self.c_object.inc_color.observe(self.__rays_color_def, 'value')
        self.c_object.refl_color.observe(self.__rays_color_def, 'value')
        
        return left_content

    
    def design_ring_array(self):
        
        data1_ringArray = widgets.Box([self.Rin_0_widget, self.A_target_widget, self.N_widget, self.w_widget, 
                                       self.h_max_widget, self.result_label],
                                      layout=widgets.Layout(
                                          display='flex',
                                          flex_flow='column',
                                          border='solid 2px'))
        data2 = widgets.Box([self.source, self.resolution], layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px'
            ))
        
        left_content_2 = widgets.VBox([data1_ringArray, data2, self.plot_surf_bt, self.plot_inc_bt, self.plot_refl_bt])
        
        self.__variables_ringarray()
        self.__res()
        self.update_variables()
        
        # observes and links
        self.Rin_0_widget.observe(self.__variables_ringarray,'value')
        self.A_target_widget.observe(self.__variables_ringarray,'value')
        self.N_widget.observe(self.__variables_ringarray,'value')
        self.w_widget.observe(self.__variables_ringarray,'value')
        self.h_max_widget.observe(self.__variables_ringarray,'value')
        self.rot_angle.observe(self.__variables_ringarray,'value')
        self.resolution.observe(self.__res, 'value')
        self.h_object.source_geometry.observe(self.__illumination_source,'value')
        self.c_object.inc_color.observe(self.__rays_color_def, 'value')
        self.c_object.refl_color.observe(self.__rays_color_def, 'value')

        return left_content_2
    
    
    def design_cylindrical(self):
        
        focus = widgets.VBox([self.length, self.height_cyl, self.thickness])
        data1_cylindrical = widgets.Box([focus, self.rot_angle], layout=widgets.Layout(
                    display='flex',
                    flex_flow='column',
                    border='solid 2px'
                ))

        data2 = widgets.Box([self.source, self.resolution], layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px'
        ))
        
        left_content_3 = widgets.VBox([data1_cylindrical, data2, self.plot_surf_bt, self.plot_inc_bt, self.plot_refl_bt])

        self.__variables_cylinder()
        self.__res()
        self.update_variables()
        
        # observes and links
        self.length.observe(self.__variables_cylinder, 'value')
        self.height_cyl.observe(self.__variables_cylinder,'value')
        self.thickness.observe(self.__variables_cylinder,'value')
        self.rot_angle.observe(self.__variables_cylinder,'value')
        self.h_object.source_geometry.observe(self.__illumination_source,'value')
        self.resolution.observe(self.__res, 'value')
        self.c_object.inc_color.observe(self.__rays_color_def, 'value')
        self.c_object.refl_color.observe(self.__rays_color_def, 'value')
        
        return left_content_3


    def __fyEQfx(self, *args):
        if self.fy_eq_fx.value:
            widgets.jsdlink((self.focus_x, 'value'), (self.focus_y, 'value'))
            self.focus_y.disabled = True
        else:
            self.focus_y.disabled = False
            
    
    def __rays_color_def(self, *arg):
        self.inc_rayClr_value = self.c_object.inc_color.value
        self.refl_rayClr_value = self.c_object.refl_color.value
        
        self.update_variables()
        
            
    def __illumination_source(self, *args):
        if self.h_object.source_geometry.value==1:
            self.source = widgets.VBox([self.source_label, self.source_posX, self.source_posY, self.source_posZ])
            self.pt_source_pos=(self.source_posX.value, self.source_posY.value, self.source_posZ.value)
            
        if self.h_object.source_geometry.value==2:
            self.source = widgets.VBox([self.source_label, self.source_posX, self.source_posY, self.source_posZ, self.large_source_extend])
            self.large_source_pos=(self.source_posX.value, self.source_posY.value, self.source_posZ.value)
            
        self.update_variables()
        
    
    def __res(self, *arg):
        temp = int(np.sqrt(self.resolution.value))
        self.resolution.value = temp**2
        self.n_theta = temp + 1
        self.n_phi = self.n_theta
        self.update_variables()
            
    def __variables_parabola(self, *args):
        self.fx = self.focus_x.value
        self.fy = self.focus_y.value
        self.h = self.height.value
        self.khoi = self.rot_angle.value/180 * np.pi
        
        self.c_object.focus_zoom.min = 0
        self.c_object.focus_zoom.max = self.focus_x.value*3/2
        self.c_object.focus_zoom.min = self.focus_x.value/2
        self.c_object.focus_zoom.value = [self.focus_x.value/2, self.focus_x.value*3/2]
        self.c_object.focus_zSlider.min = 0
        self.c_object.focus_zSlider.max = self.focus_x.value*3/2
        self.c_object.focus_zSlider.min = self.focus_x.value/2
        self.c_object.focus_zSlider.value = 0
        
        self.update_variables()
        
    
    def __variables_ringarray(self, *args):
        self.Rin_0 = self.Rin_0_widget.value
        self.A_target = self.A_target_widget.value
        self.N = int(self.N_widget.value)
        self.w = self.w_widget.value  
        self.h_max = self.h_max_widget.value
        
        self.c_object.focus_zoom.min = 0
        self.c_object.focus_zoom.max = 0.25
        self.c_object.focus_zoom.min = -0.25
        self.c_object.focus_zoom.value = [-0.25, 0.25]
        self.c_object.focus_zSlider.min = 0
        self.c_object.focus_zSlider.max = 0.25
        self.c_object.focus_zSlider.min = -0.25
        self.c_object.focus_zSlider.value =-0
        
        self.update_variables()
        
    
    def __variables_cylinder(self, *args):
        self.L = self.length.value
        self.th = self.thickness.value
        self.h_cyl = self.height_cyl.value
        self.khoi = self.rot_angle.value/180 * np.pi
        
        self.c_object.focus_zoom.min = 0
        self.c_object.focus_zoom.max = 1/2
        self.c_object.focus_zoom.min = -1/2
        self.c_object.focus_zoom.value = [-1/2, 1/2]
        self.c_object.focus_zSlider.min = 0
        self.c_object.focus_zSlider.max = 1/2
        self.c_object.focus_zSlider.min = -1/2
        self.c_object.focus_zSlider.value = 0
        
        self.update_variables()

        