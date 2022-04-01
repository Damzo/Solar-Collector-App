from ipywidgets import widgets, Layout
import matplotlib.pyplot as plt
from ipyleaflet import Map, SearchControl, Marker, AwesomeIcon
import requests
import json
import numpy as np

class footer_content:
    style = {'description_width': 'initial'}
    item_layout = widgets.Layout( width='auto')
    
    
    def __init__(self) -> None:
        self.fig_size = (2.5, 2)
        # for raw 2D analysis
        self.xy_sun = plt.figure(num=2, figsize=self.fig_size)
        self.xy_temperature = plt.figure(num=3, figsize=self.fig_size)
        self.xy_radiation = plt.figure(num=4, figsize=self.fig_size)
        self.radiation_label = widgets.Label('Solar irradiation (W/m^2)')
        self.raw_radiation = widgets.FloatText(
            value=4.5,
            disabled=False, style=self.style, layout=self.item_layout)
        self.plot_Analysis = widgets.Button(description='Plot datas', button_style='success', layout=self.item_layout)
        
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
        info_box = widgets.VBox([self.gps_bt, self.out_location, self.start_date, self.end_date, self.plot_Analysis])
        info_box.layout.display = 'flex'
        self.tab = widgets.Tab()
        for i in range(len(tab_titles)):
            self.tab.set_title(i, tab_titles[i])
        self.tab1_content = widgets.HBox([widgets.VBox([self.radiation_label,self.raw_radiation, self.out_location, self.plot_Analysis]), 
                                          self.xy_radiation.canvas, self.xy_sun.canvas, self.xy_temperature.canvas])
        self.tab1_content.layout.display='flex'
        self.tab2_content = widgets.HBox([info_box, self.map_box, self.time_radiation.canvas, 
                                     self.time_temperature.canvas])
        self.tab3_content = widgets.HBox([info_box, self.map_box, self.time_radiation.canvas, 
                                     self.time_temperature.canvas])
        
        self.tab.children = [self.tab1_content, self.tab2_content, self.tab3_content]
        self.tab_idx = 0
        self.tab.observe( self.__selected_tab, names='selected_index')
    
    def api_request(self, start_year=2020, end_year=2020):
        end_point = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        fmt = "JSON"
        community="RE"
        param = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,WS2M,QV2M,RH2M"
        latd = self.location[0]
        lgtd = self.location[1]

        load = {'parameters': param, 'community': community, 'longitude': lgtd, 'latitude': latd, 
                'format': fmt, 'start': start_year, 'end': end_year}
        r = requests.get(end_point, params=load)
        data = json.loads(r.text)
        
        # Load parameters
        ALLSKY_SFC_SW_DWN = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        CLRSKY_SFC_SW_DWN = data['properties']['parameter']['CLRSKY_SFC_SW_DWN']
        RH2M = data['properties']['parameter']['RH2M']
        QV2M = data['properties']['parameter']['QV2M']
        WS2M = data['properties']['parameter']['WS2M']
        units = {'ALLSKY_SFC_SW_DWN':data['parameters']['ALLSKY_SFC_SW_DWN']['units'],
                'CLRSKY_SFC_SW_DWN':data['parameters']['CLRSKY_SFC_SW_DWN']['units'],
                'RH2M':data['parameters']['RH2M']['units'],
                'QV2M':data['parameters']['QV2M']['units'],
                'WS2M':data['parameters']['WS2M']['units']
                }

        ALLSKY_SFC_SW_DWN = np.array([np.array([k, v]) for k, v in ALLSKY_SFC_SW_DWN.items()])
        CLRSKY_SFC_SW_DWN = np.array([np.array([k, v]) for k, v in CLRSKY_SFC_SW_DWN.items()])
        RH2M = np.array([np.array([k, v]) for k, v in RH2M.items()])
        QV2M = np.array([np.array([k, v]) for k, v in QV2M.items()])
        WS2M = np.array([np.array([k, v]) for k, v in WS2M.items()])

        # slice years from months
        All_Sky_Irradiance = np.zeros((ALLSKY_SFC_SW_DWN.shape[0], ALLSKY_SFC_SW_DWN.shape[1]+1))
        Clear_Sky_Irradiance = np.zeros((CLRSKY_SFC_SW_DWN.shape[0], CLRSKY_SFC_SW_DWN.shape[1]+1))
        Relative_Humidity = np.zeros((RH2M.shape[0], RH2M.shape[1]+1))
        Specific_Humidity = np.zeros((QV2M.shape[0], QV2M.shape[1]+1))
        Wind_Speed = np.zeros((WS2M.shape[0], WS2M.shape[1]+1))

        All_Sky_Irradiance[:,0] = [i[0:4] for i in ALLSKY_SFC_SW_DWN[:,0]]
        All_Sky_Irradiance[:,1] = [i[4:6] for i in ALLSKY_SFC_SW_DWN[:,0]]
        All_Sky_Irradiance[:,2] = ALLSKY_SFC_SW_DWN[:,1]
        Clear_Sky_Irradiance[:,0] = [i[0:4] for i in CLRSKY_SFC_SW_DWN[:,0]]
        Clear_Sky_Irradiance[:,1] = [i[4:6] for i in CLRSKY_SFC_SW_DWN[:,0]]
        Clear_Sky_Irradiance[:,2] = CLRSKY_SFC_SW_DWN[:,1]
        Relative_Humidity[:,0] = [i[0:4] for i in RH2M[:,0]]
        Relative_Humidity[:,1] = [i[4:6] for i in RH2M[:,0]]
        Relative_Humidity[:,2] = RH2M[:,1]
        Specific_Humidity[:,0] = [i[0:4] for i in QV2M[:,0]]
        Specific_Humidity[:,1] = [i[4:6] for i in QV2M[:,0]]
        Specific_Humidity[:,2] = QV2M[:,1]
        Wind_Speed[:,0] = [i[0:4] for i in WS2M[:,0]]
        Wind_Speed[:,1] = [i[4:6] for i in WS2M[:,0]]
        Wind_Speed[:,2] = WS2M[:,1]
        
        return All_Sky_Irradiance, Clear_Sky_Irradiance, Relative_Humidity, Specific_Humidity, Wind_Speed, units
    
    def plot_raw2D_datas(self, data_2D, area):
        if not(plt.isinteractive):
            plt.ion()
        plt.figure(num=4)
        plt.imshow(np.rot90(data_2D))
    
    def plot_thermal_datas(self):
        start = str(self.start_date.value)
        end = str(self.end_date.value)
        All_Sky_Irradiance, Clear_Sky_Irradiance, Relative_Humidity, Specific_Humidity, Wind_Speed, units = self.api_request(start[0:5], end[0:5])
        plt.figure(num=4)
        x = All_Sky_Irradiance[:,1]
        ticks = []
        for i in range(x.size):
            a = str(int(x[i])) + '/' + str(int(All_Sky_Irradiance[i,0]))
            ticks.append(a)
            # print(ticks[i])

        y = All_Sky_Irradiance[:,2]
        plt.scatter(ticks, y)
        plt.xticks(ticks, ticks, rotation=90)
        plt.show()
    
    def plot_photovoltaic_datas(self):
        
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
        
    def __selected_tab(self, widget):
        self.tab_idx = widget['new']
