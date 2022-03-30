from ipywidgets import widgets, Layout
import matplotlib.pyplot as plt
from ipyleaflet import Map, SearchControl, Marker, AwesomeIcon
import requests
import json

class footer_content:
    style = {'description_width': 'initial'}
    item_layout = widgets.Layout( width='auto')
    
    
    def __init__(self) -> None:
        self.fig_size = (2.5, 2)
        # for raw 2D analysis
        self.xy_sun = plt.figure(num=2, figsize=self.fig_size)
        self.xy_temperature = plt.figure(num=3, figsize=self.fig_size)
        self.xy_radiation = plt.figure(num=4, figsize=self.fig_size)
        self.raw_radiation = widgets.FloatText(
            value=4.5,
            description='Solar radiation (kW-hr/m^2)',
            disabled=False, style=self.style, layout=self.item_layout)
        
        # Map box for selecting a GPS position
        univ_kara = (9.68803261638838, 1.1285112820273637)
        self.m = Map(zoom=5, center=univ_kara, layout=Layout(width='250px', height='250px'))
        self.marker = Marker(location=univ_kara, icon=AwesomeIcon(name="check", marker_color='green', icon_color='darkgreen'))
        self.m.add_control(SearchControl(
        position="topleft",
        url='https://nominatim.openstreetmap.org/search?format=json&q={s}',
        zoom=5,
        marker=self.marker
        ))
        self.m.add_layer(self.marker)
        self.location = univ_kara
        self.marker.on_move(self.__handle_move)
        self.close_but = widgets.Button(icon='window-close', layout=Layout(width='40px', height='40px'))
        self.close_but.on_click(self.__close_map)
        self.map_box = widgets.Box([self.m, self.close_but], layout=Layout(width='260px', height='260px'))
        
        self.gps_bt = widgets.Button(description='Select location', button_style='info', style=self.style, icon='map-marker',
                                      layout=self.item_layout)
        self.gps_bt.on_click(self.__show_map)
        self.out_location = widgets.Output(layout=Layout(width='150px', height='100px'))
        self.start_date = widgets.DatePicker(description = 'Start Date')
        self.start_date.layout.width = 'auto'
        self.end_date = widgets.DatePicker(description = 'End Date')
        self.end_date.layout.width = 'auto'
        
        # for Solar thermal analysis
        self.time_radiation = plt.figure(num=5, figsize=self.fig_size)
        self.time_temperature = plt.figure(num=6, figsize=self.fig_size)
                
        # for Solar photovoltaic analysis
        
        # design_footer
        tab_titles = ['Raw 2D analysis', 'Solar Thermal', 'Solar Photovoltaic']
        info_box = widgets.VBox([self.gps_bt, self.out_location, self.start_date, self.end_date])
        info_box.layout.display = 'flex'
        self.tab = widgets.Tab()
        for i in range(len(tab_titles)):
            self.tab.set_title(i, tab_titles[i])
        self.tab1_content = widgets.HBox([info_box , self.map_box, self.xy_radiation.canvas, 
                                     self.xy_sun.canvas, self.xy_temperature.canvas])
        self.tab1_content.layout.display='flex'
        self.tab2_content = widgets.HBox([info_box, self.map_box, self.time_radiation.canvas, 
                                     self.time_temperature.canvas])
        self.tab3_content = widgets.HBox([info_box, self.map_box, self.time_radiation.canvas, 
                                     self.time_temperature.canvas])
        
        self.tab.children = [self.tab1_content, self.tab2_content, self.tab3_content]
    
    def api_request(self):
        end_point = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        fmt = "JSON"
        community="RE"
        param = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,WS2M,QV2M,RH2M"
        start_year=2020
        end_year=2020
        latd = self.location[0]
        lgtd = self.location[1]

        load = {'parameters': param, 'community': community, 'longitude': lgtd, 'latitude': latd, 'format': fmt, 'start': start_year, 'end': end_year}

        r = requests.get(end_point, params=load)

        data = json.loads(r.text)
        
        return data
    
    def plot_radiation_2D(self):
        
        pass
    
    def __handle_move(self, *args, **kwargs):
            self.location = self.marker.location
            with self.out_location:
                self.out_location.clear_output()
                print('Longitude: ')
                print(self.location[0])
                print('')
                print('Latitude: ')
                print(self.location[1])
            
    def __close_map(self, *args, **kwargs):
        self.map_box.layout = Layout(width='10px', height='10px')
        
    def __show_map(self, *args, **kwargs):
        self.map_box.layout = Layout(width='260px', height='260px')
