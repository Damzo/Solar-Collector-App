import ipyvolume as ipv
import ipywidgets as widgets


class center_content:
    style = {'description_width': 'initial'}
    item_layout = widgets.Layout(justify_content='center', margin_left='100px', width='auto')
    analysis_bt = widgets.ToggleButton(value=False, description='Show analysis section', button_style='info', 
                                       tooltip='show/hide analysis section',
                                       layout=widgets.Layout(justify_content='center', margin_left='100px', width='75%'))

    def __init__(self):

        self.focus_zoom = widgets.FloatRangeSlider(
            value=[-0.25, 0.75],
            min=-1,
            max=1,
            step=0.05,
            description='Zoom',
            disabled=False,
            orientation='vertical',
            readout=True,
            height='auto',
            readout_format='.2f', style=self.style, layout=self.item_layout)
        self.focus_zSlider = widgets.FloatSlider(
            value=0,
            min=-1,
            max=1,
            description='Axial cut',
            orientation='vertical',
            readout=False, style=self.style, layout=self.item_layout)
        self.focus_zText = widgets.FloatText(
            value=0.0,
            step=0.01,
            description='',
            disabled=False, layout=widgets.Layout(width='auto'))
        self.inc_color_lbl = widgets.Label(value="Incident rays color", style=self.style)
        self.refl_color_lbl = widgets.Label(value="Reflected rays color", style=self.style)
        self.inc_color = widgets.ColorPicker(
            concise=True,
            desciption='Incident rays color',
            value='yellow',
            disabled=False, style=self.style, layout=self.item_layout)
        self.refl_color = widgets.ColorPicker(
            concise=True,
            desciption='Reflected rays color',
            value='Red',
            disabled=False, style=self.style, layout=self.item_layout)
        self.plot_zoom_bt = widgets.Button(description='Plot Zoom', button_style='success', layout=self.item_layout, tooltip='Plot Zoom')
        self.plot_xy_bt = widgets.Button(description='Plot XY', button_style='success', layout=self.item_layout, tooltip='Plot XY')

        self.main_scene = ipv.figure(width=400, height=400)
        

    def design(self):
        
        focus_xy = widgets.VBox([self.focus_zSlider, self.focus_zText, self.plot_xy_bt])
        zoom = widgets.VBox([self.focus_zoom, self.plot_zoom_bt])
        rays_color = widgets.HBox([self.inc_color_lbl, self.inc_color, self.refl_color_lbl, self.refl_color])
        rays_color.layout.justify_content='center'
        part_1 = widgets.VBox([rays_color, widgets.HBox([self.main_scene, widgets.VBox([zoom, focus_xy])])])
        center_cont = widgets.VBox([part_1, self.analysis_bt])
        center_cont.layout.align_items = 'center'  # 'flex-end'
        center_cont.layout.justify_content = 'center'
        
        # Observes and links
        widgets.jslink((self.focus_zSlider, 'value'), (self.focus_zText, 'value'))

        return center_cont
