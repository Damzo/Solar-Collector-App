from ipywidgets import widgets, Layout
# import ipyvolume as ipv
import matplotlib.pyplot as plt
# from matplotlib import cm
from ipyleaflet import Map, SearchControl, Marker, AwesomeIcon
import requests
import json
import numpy as np
import datetime


class footer_content:
    style = {'description_width': 'initial'}
    item_layout = widgets.Layout(width='auto')

    def __init__(self) -> None:
        self.fig_size = (2, 2)
        # for raw 2D analysis
        self.xy_radiation = plt.figure(num='Radiation on receptor', figsize=self.fig_size)
        self.xy_radiation.canvas.header_visible = False
        # self.xy_radiation.label('Radiation on receptor')
        self.xy_conc_ratio = plt.figure(num='Concentration ratio', figsize=self.fig_size)
        self.xy_conc_ratio.canvas.header_visible = False
        # self.xy_conc_ratio.suptitle('Concentration ratio')
        # self.xy_temperature = plt.figure(num='Theoretical temperature', figsize=self.fig_size)
        # self.xy_temperature.canvas.header_visible = False
        # self.xy_temperature.suptitle('Theoretical temperature')
        self.radiation_label = widgets.Label('Solar irradiation (W.hr/m²)', style=self.style)
        self.raw_radiation = widgets.FloatText(
            value=4.5,
            disabled=False, style=self.style, layout=self.item_layout)
        self.plot_Analysis = widgets.Button(description='Plot datas', button_style='success', layout=self.item_layout)

        # Map box for selecting a GPS position  
        univ_kara = (9, 1)
        self.m = Map(center=univ_kara, zoom=5, layout=Layout(width='90%', height='250px'))
        self.marker = Marker(location=univ_kara,
                             icon=AwesomeIcon(name="check", marker_color='green', icon_color='darkgreen'))
        search_instance = SearchControl(
            position="topright",
            url='https://nominatim.openstreetmap.org/search?format=json&q={s}',
            zoom=5,
            marker=self.marker
        )
        self.m.add_control(search_instance)
        self.m.add_layer(self.marker)
        self.location = univ_kara
        self.marker.on_move(self.__handle_move)
        search_instance.on_location_found(self.__handle_move)
        self.close_but = widgets.Button(icon='window-close', layout=Layout(width='40px', height='40px'))
        self.close_but.on_click(self.__close_map)
        self.map_box = widgets.Box([self.m, self.close_but], layout=Layout(width='250px', height='500px'))

        self.gps_bt = widgets.Button(description='Select location', button_style='info', style=self.style,
                                     icon='map-marker',
                                     layout=self.item_layout)
        self.gps_bt.on_click(self.__show_map)
        self.out_location = widgets.Output()

        actual_date = datetime.datetime.utcnow()
        self.start_date = widgets.Dropdown(description='Start date',
                                           options=[i for i in range(1981, actual_date.year)],
                                           value=1981,
                                           disabled=False)
        self.start_date.layout.width = 'auto'
        self.end_date = widgets.Dropdown(description='End date',
                                         options=[i for i in range(1981, actual_date.year)],
                                         value=1981,
                                         disabled=False)
        self.end_date.layout.width = 'auto'

        # for meteorological data ploting
        self.meteo_figures = plt.figure(num='Data from Nasa Power Data', figsize=(4, 3), constrained_layout=True)  #
        self.meteo_figures.canvas.header_visible = False

        self.meteo_data_dropdown = widgets.Dropdown(
            options={'All Sky Irradiance': 1,
                     'Clear & Diffuse_Irradiance': 2,
                     'Midday Insolation': 3,
                     'All Sky Albedo': 4,
                     'Average Solar Declination': 5,
                     'Average Hourly Solar Angles': 6,
                     'Humidity at 2m': 7,
                     'Wind_Speed at 2m': 8,
                     'Temperature at 2m': 9},
            # options={'Sky Irradiance':1,
            #          'Humidity at 2m':2,
            #          'Wind_Speed':3,
            #          'Temperature':4},
            value=1,
            description='',
            disabled=False
            # button_style='danger' # 'success', 'info', 'warning', 'danger' or ''
        )
        self.meteo_data_dropdown.layout.width = 'auto'

        self.meteo_data_dropdown.observe(self.__meteo_data_plot_change, names='value')

        self.time_radiation = plt.figure(num='Sky Radiation', figsize=self.fig_size)
        self.time_radiation.canvas.header_visible = False
        self.time_humidity = plt.figure(num='Humidity', figsize=self.fig_size)
        self.time_humidity.canvas.header_visible = False
        self.time_wind = plt.figure(num='Wind Speed', figsize=self.fig_size)
        self.time_wind.canvas.header_visible = False

        # for Solar thermal analysis
        self.time_tmp1 = plt.figure(num=5, figsize=self.fig_size)
        self.time_tmp2 = plt.figure(num=6, figsize=self.fig_size)

        # for Solar photovoltaic analysis
        self.time_tmp1 = plt.figure(num=5, figsize=self.fig_size)
        self.time_tmp2 = plt.figure(num=6, figsize=self.fig_size)

        # design_footer
        raw_box = widgets.VBox([self.radiation_label, self.raw_radiation, self.out_location, self.plot_Analysis])
        info_box = widgets.VBox([self.gps_bt, self.out_location, self.start_date, self.end_date, self.plot_Analysis])

        self.tab1_content = widgets.HBox([raw_box, self.xy_radiation.canvas, self.xy_conc_ratio.canvas])
        self.tab1_content.layout.display = 'flex'
        # self.tab2_content = widgets.HBox([info_box, self.map_box, self.time_radiation.canvas, 
        #                              self.time_humidity.canvas, self.time_wind.canvas])
        self.tab2_content = widgets.HBox([widgets.VBox([info_box, self.meteo_data_dropdown]), self.map_box,
                                          self.meteo_figures.canvas])
        self.tab2_content.layout.display = 'flex'
        self.tab3_content = widgets.HBox([info_box, self.map_box, self.time_tmp1.canvas,
                                          self.time_tmp2.canvas])
        self.tab3_content.layout.display = 'flex'
        self.tab4_content = widgets.HBox([info_box, self.map_box, self.time_tmp1.canvas,
                                          self.time_tmp2.canvas])
        self.tab4_content.layout.display = 'flex'

        self.tab = widgets.Tab()
        tab_titles = ['Raw 2D analysis', 'Meteorological data', 'Solar Thermal', 'Solar Photovoltaic']
        self.tab.children = [self.tab1_content, self.tab2_content, self.tab3_content, self.tab4_content]
        for i in range(len(tab_titles)):
            self.tab.set_title(i, tab_titles[i])
        self.tab_idx = 0
        self.tab.observe(self.__selected_tab, names='selected_index')

        self.api_response = []
        self.All_Sky_Irradiance = []
        self.Clear_Sky_Irradiance = []
        self.All_Sky_Diffused_Irradiance = []
        self.Midday_Insolation = []
        self.All_Sky_Albedo = []
        self.Average_Declination = []
        self.Solar_Angle = []
        self.Relative_Humidity = []
        self.Specific_Humidity = []
        self.Wind_Speed = []
        self.Temperature = []
        self.ticks = []
        self.units = {}

    # method to request data from the Nasa database and put them on the right format for other methods
    def api_request(self, start_year=2020, end_year=2020):
        """Method to make request to the api endpoint
        live use case: https://power.larc.nasa.gov/data-access-viewer/
        parameter dictionary: https://power.larc.nasa.gov/#resources
        api docs : https://power.larc.nasa.gov/docs/services/api/ 
        
        Args:
            start_year (int, optional): _description_. Defaults to 2020.
            end_year (int, optional): _description_. Defaults to 2020.

        Returns:
            _type_: _description_
        """
        end_point = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        fmt = "JSON"
        community = "RE"
        param = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF," \
                "MIDDAY_INSOL,ALLSKY_SRF_ALB,SG_DEC,SG_HRZ_HR," \
                "WS2M,QV2M,RH2M,T2M"
        latd = self.location[0]
        lgtd = self.location[1]

        load = {'parameters': param, 'community': community, 'longitude': lgtd, 'latitude': latd,
                'format': fmt, 'start': start_year, 'end': end_year}
        r = requests.get(end_point, params=load)
        data = json.loads(r.text)

        if r.status_code == 200:
            with self.out_location:
                self.out_location.clear_output()
                print('API response code: ', r.status_code)
                print('datas uploaded')
            # Load parameters
            # ALLSKY_SFC_SW_DWN = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
            # CLRSKY_SFC_SW_DWN = data['properties']['parameter']['CLRSKY_SFC_SW_DWN']
            # RH2M = data['properties']['parameter']['RH2M']
            # QV2M = data['properties']['parameter']['QV2M']
            # WS2M = data['properties']['parameter']['WS2M']
            # T2M = data['properties']['parameter']['T2M']
            # units = {'All_Sky_Irradiance':data['parameters']['ALLSKY_SFC_SW_DWN']['units'],
            #         'Clear_Sky_Irradiance':data['parameters']['CLRSKY_SFC_SW_DWN']['units'],
            #         'Relative_Humidity':data['parameters']['RH2M']['units'],
            #         'Specific_Humidity':data['parameters']['QV2M']['units'],
            #         'Wind_Speed':data['parameters']['WS2M']['units'],
            #         'Temperature':data['parameters']['T2M']['units']
            #         }
            #
            # ALLSKY_SFC_SW_DWN = np.array([np.array([k, v]) for k, v in ALLSKY_SFC_SW_DWN.items()])
            # CLRSKY_SFC_SW_DWN = np.array([np.array([k, v]) for k, v in CLRSKY_SFC_SW_DWN.items()])
            # RH2M = np.array([np.array([k, v]) for k, v in RH2M.items()])
            # QV2M = np.array([np.array([k, v]) for k, v in QV2M.items()])
            # WS2M = np.array([np.array([k, v]) for k, v in WS2M.items()])
            # T2M = np.array([np.array([k, v]) for k, v in T2M.items()])
            #
            # # slice years from months
            # All_Sky_Irradiance = np.zeros((ALLSKY_SFC_SW_DWN.shape[0], ALLSKY_SFC_SW_DWN.shape[1]+1))
            # Clear_Sky_Irradiance = np.zeros((CLRSKY_SFC_SW_DWN.shape[0], CLRSKY_SFC_SW_DWN.shape[1]+1))
            # Relative_Humidity = np.zeros((RH2M.shape[0], RH2M.shape[1]+1))
            # Specific_Humidity = np.zeros((QV2M.shape[0], QV2M.shape[1]+1))
            # Wind_Speed = np.zeros((WS2M.shape[0], WS2M.shape[1]+1))
            # Temperature = np.zeros((T2M.shape[0], T2M.shape[1]+1))
            #
            # All_Sky_Irradiance[:,0] = [i[0:4] for i in ALLSKY_SFC_SW_DWN[:,0]]
            # All_Sky_Irradiance[:,1] = [i[4:6] for i in ALLSKY_SFC_SW_DWN[:,0]]
            # All_Sky_Irradiance[:,2] = ALLSKY_SFC_SW_DWN[:,1]
            # Clear_Sky_Irradiance[:,0] = [i[0:4] for i in CLRSKY_SFC_SW_DWN[:,0]]
            # Clear_Sky_Irradiance[:,1] = [i[4:6] for i in CLRSKY_SFC_SW_DWN[:,0]]
            # Clear_Sky_Irradiance[:,2] = CLRSKY_SFC_SW_DWN[:,1]
            # Relative_Humidity[:,0] = [i[0:4] for i in RH2M[:,0]]
            # Relative_Humidity[:,1] = [i[4:6] for i in RH2M[:,0]]
            # Relative_Humidity[:,2] = RH2M[:,1]
            # Specific_Humidity[:,0] = [i[0:4] for i in QV2M[:,0]]
            # Specific_Humidity[:,1] = [i[4:6] for i in QV2M[:,0]]
            # Specific_Humidity[:,2] = QV2M[:,1]
            # Wind_Speed[:,0] = [i[0:4] for i in WS2M[:,0]]
            # Wind_Speed[:,1] = [i[4:6] for i in WS2M[:,0]]
            # Wind_Speed[:,2] = WS2M[:,1]
            # Temperature[:,0] = [i[0:4] for i in T2M[:,0]]
            # Temperature[:,1] = [i[4:6] for i in T2M[:,0]]
            # Temperature[:,2] = T2M[:,1]
            ALLSKY_SFC_SW_DWN = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
            CLRSKY_SFC_SW_DWN = data['properties']['parameter']['CLRSKY_SFC_SW_DWN']
            ALLSKY_SFC_SW_DIFF = data['properties']['parameter']['ALLSKY_SFC_SW_DIFF']
            MIDDAY_INSOL = data['properties']['parameter']['MIDDAY_INSOL']
            ALLSKY_SRF_ALB = data['properties']['parameter']['ALLSKY_SRF_ALB']
            SG_DEC = data['properties']['parameter']['SG_DEC']
            list_l = [format(i, "02d") for i in range(24)]
            SG_HRZ_HR = [data['properties']['parameter']['SG_HRZ_%s' % i] for i in list_l]
            RH2M = data['properties']['parameter']['RH2M']
            QV2M = data['properties']['parameter']['QV2M']
            WS2M = data['properties']['parameter']['WS2M']
            T2M = data['properties']['parameter']['T2M']
            units = {'All_Sky_Irradiance': data['parameters']['ALLSKY_SFC_SW_DWN']['units'],
                     'Clear_Sky_Irradiance': data['parameters']['CLRSKY_SFC_SW_DWN']['units'],
                     'All_Sky_Diffused_Irradiance': data['parameters']['ALLSKY_SFC_SW_DIFF']['units'],
                     'Midday_Insolation': data['parameters']['MIDDAY_INSOL']['units'],
                     'All_Sky_Albedo': data['parameters']['ALLSKY_SRF_ALB']['units'],
                     'Average_Solar_Declination': data['parameters']['SG_DEC']['units'],
                     'Average_Hourly_Solar_Angles': data['parameters']['SG_HRZ_00']['units'],
                     'Relative_Humidity': data['parameters']['RH2M']['units'],
                     'Specific_Humidity': data['parameters']['QV2M']['units'],
                     'Wind_Speed': data['parameters']['WS2M']['units'],
                     'Temperature': data['parameters']['T2M']['units']
                     }

            ALLSKY_SFC_SW_DWN = np.array([np.array([k, v]) for k, v in ALLSKY_SFC_SW_DWN.items()])
            CLRSKY_SFC_SW_DWN = np.array([np.array([k, v]) for k, v in CLRSKY_SFC_SW_DWN.items()])
            ALLSKY_SFC_SW_DIFF = np.array([np.array([k, v]) for k, v in ALLSKY_SFC_SW_DIFF.items()])

            MIDDAY_INSOL = np.array([np.array([k, v]) for k, v in MIDDAY_INSOL.items()])
            ALLSKY_SRF_ALB = np.array([np.array([k, v]) for k, v in ALLSKY_SRF_ALB.items()])
            SG_DEC = np.array([np.array([k, v]) for k, v in SG_DEC.items()])
            SG_HRZ = []
            for w in range(24):
                SG_HRZ.append(np.array([np.array((k, v)) for k, v in SG_HRZ_HR[w].items()]))
            SG_HRZ = np.array(SG_HRZ)

            RH2M = np.array([np.array([k, v]) for k, v in RH2M.items()])
            QV2M = np.array([np.array([k, v]) for k, v in QV2M.items()])
            WS2M = np.array([np.array([k, v]) for k, v in WS2M.items()])
            T2M = np.array([np.array([k, v]) for k, v in T2M.items()])

            # slice years from months

            x_val = [j for j in ALLSKY_SFC_SW_DWN[:-1, 0]]

            All_Sky_Irradiance = [i for i in ALLSKY_SFC_SW_DWN[:-1, 1]]
            Clear_Sky_Irradiance = [i for i in CLRSKY_SFC_SW_DWN[:-1, 1]]
            All_Sky_Diffused_Irradiance = [i for i in ALLSKY_SFC_SW_DIFF[:-1, 1]]

            Midday_Insolation = [i for i in MIDDAY_INSOL[:-1, 1]]
            All_Sky_Albedo = [i for i in ALLSKY_SRF_ALB[:-1, 1]]
            Average_Declination = [i for i in SG_DEC[:-1, 1]]
            Solar_Angle = []
            for w in range(24):
                Solar_Angle.append(SG_HRZ[w, :-1, 1])

            Relative_Humidity = [i for i in RH2M[:-1, 1]]
            Specific_Humidity = [i for i in QV2M[:-1, 1]]
            Wind_Speed = [i for i in WS2M[:-1, 1]]
            Temperature = [i for i in T2M[:-1, 1]]

        else:
            with self.out_location:
                self.out_location.clear_output()
                print('API response code: ', r.status_code)
                if r.status_code == 422:
                    print('Reason: Validation Error')
                    print(json.loads(r.text)["messages"])
                elif r.status_code == 429:
                    print('Reason: Too Many Requests', )
                    print(json.loads(r.text)["messages"])
                elif r.status_code == 500:
                    print('Reason: Internal Server Error', )
                    print(json.loads(r.text)["messages"])
                elif r.status_code == 503:
                    print('Reason: Service Unreachable')
                    print(json.loads(r.text)["messages"])
                elif r.status_code == 404:
                    print('Internet connection problem')
                    print(json.loads(r.text)["messages"])
                else:
                    print('Failed:')
                    print(json.loads(r.text)["messages"])

            All_Sky_Irradiance = []
            Clear_Sky_Irradiance = []
            All_Sky_Diffused_Irradiance = []
            Midday_Insolation = []
            All_Sky_Albedo = []
            Average_Declination = []
            Solar_Angle = []
            Relative_Humidity = []
            Specific_Humidity = []
            Wind_Speed = []
            Temperature = []
            units = {}

        return r, All_Sky_Irradiance, Clear_Sky_Irradiance, All_Sky_Diffused_Irradiance, Midday_Insolation, \
            All_Sky_Albedo, Average_Declination, Solar_Angle, Relative_Humidity, Specific_Humidity, Wind_Speed, \
            Temperature, x_val, units

    # Method executed when user click on the button "Plot datas" under the tabs "Raw 2D analysis"
    def plot_raw2D_datas(self, data_2D, area, x, y):
        if plt.isinteractive:
            plt.ion()
        # dx = np.gradient(x)
        # dy = np.gradient(y)
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        input_rad = self.raw_radiation.value
        # focus area at 1/e of maximum 
        focus = x[data_2D >= np.amax(data_2D) / np.e]
        sigma1 = np.amax(focus)
        focus = y[data_2D >= np.amax(data_2D) / np.e]
        sigma2 = np.amax(focus)
        # Received energy "echantilloné"
        Er = input_rad * area

        # Plot of the radiation on the receptor
        Rd = (Er / (np.pi * sigma1 * sigma2)) * data_2D
        plt.figure(num='Radiation on receptor')
        plt.imshow(np.rot90(Rd), extent=[xmin, xmax, ymin, ymax], interpolation='bilinear', cmap='magma')
        # plt.xticks(np.round(x,2), rotation=45)
        # plt.yticks(np.round(y,2), rotation=30)
        plt.title('Radiation W.hr/m²')
        plt.colorbar()

        # Plot of the concentration ratio
        Cr = (area / (np.pi * sigma1 * sigma2)) * data_2D
        plt.figure(num='Concentration ratio')
        plt.imshow(np.rot90(Cr), extent=[xmin, xmax, ymin, ymax], interpolation='bilinear', cmap='magma')
        # plt.xticks(np.round(x,2), rotation=45)
        # plt.yticks(np.round(y,2), rotation=30)
        plt.title('Concentration ratio')
        plt.colorbar()

        # Plot of the Theoretical temperature
        # Tr = Cr
        # plt.figure(num='Theoretical temperature')
        # plt.imshow(np.rot90(Tr))
        # plt.xticks(x)
        # plt.yticks(y)

    # Method executed when user click on the button "Plot datas" under the tabs "Meteorological data"
    def plot_meteo_datas(self):

        self.api_response, self.All_Sky_Irradiance, self.Clear_Sky_Irradiance, self.All_Sky_Diffused_Irradiance, \
            self.Midday_Insolation, self.All_Sky_Albedo, self.Average_Declination, self.Solar_Angle, \
            self.Relative_Humidity, self.Specific_Humidity, self.Wind_Speed, self.Temperature, \
            self.ticks, self.units \
            = self.api_request(self.start_date.value, self.end_date.value)

        if self.api_response.status_code == 200:
            plt.ion()
            self.__meteo_data_plot_change()

    # Method executed when user click on the button "Plot datas" under the tabs "Solar Thermal"
    def plot_thermal_datas(self):

        pass

    # Method executed when user click on the button "Plot datas" under the tabs "Solar Photovoltaic"
    def plot_photovoltaic_datas(self):

        pass

    def __handle_move(self, *args, **kwargs):
        self.location = self.marker.location
        # self.m.center = self.location
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
        self.map_box.layout = Layout(width='250px', height='500px')

    def __selected_tab(self, widget):
        self.tab_idx = widget['new']

        if self.tab_idx == 1:
            with self.out_location:
                self.out_location.clear_output()
                print('Universite de Kara ')
                print('Longitude: ')
                print(self.location[0])
                print('Latitude: ')
                print(self.location[1])

    # Method that handles change of plots under the tab "Meteorological data"
    def __meteo_data_plot_change(self, *args, **kwargs):
        num = self.meteo_data_dropdown.value
        try:
            self.ax1
        except AttributeError:
            self.meteo_figures.clf()
        else:
            self.meteo_figures.clf()
            self.ax1.cla()

        try:
            self.ax2
        except AttributeError:
            self.meteo_figures.clf()
        else:
            self.meteo_figures.clf()
            self.ax2.cla()

        if num == 1:
            # 'All Sky Irradiance': 1
            self.__plotxy(y=self.All_Sky_Irradiance, label='All sky irradiance',
                          y_unit=self.units['All_Sky_Irradiance'],
                          fig_title='All sky irradiance')

        elif num == 2:
            # 'Clear & Diffused Irradiance': 2
            self.__plotxyy(y1=self.Clear_Sky_Irradiance, y2=self.All_Sky_Diffused_Irradiance,
                           label1='Clear sky', label2='Diffused',
                           y1_unit=self.units['Clear_Sky_Irradiance'],
                           y2_unit=self.units['All_Sky_Diffused_Irradiance'],
                           fig_title='Clear & Diffused Irradiance')

        elif num == 3:
            # 'Midday Insolation': 3
            self.__plotxy(y=self.Midday_Insolation, label='Midday Insolation', y_unit=self.units['Midday_Insolation'],
                          fig_title='Midday Insolation')

        elif num == 4:
            # 'All Sky Albedo': 4
            self.__plotxy(y=self.All_Sky_Albedo, label='Surface Albedo', y_unit=self.units['All_Sky_Albedo'],
                          fig_title='Surface Albedo')

        elif num == 5:
            # 'Average Solar Declination': 5
            self.__plotxy(y=self.Average_Declination, label='Average Solar Declination',
                          y_unit=self.units['Average_Solar_Declination'], fig_title='Average Solar Declination')

        elif num == 6:
            # 'Average Hourly Solar Angles': 6
            h_ticks = [format(i, "02d") for i in range(24)]
            z = np.zeros((24, 12))
            for w in range(24):
                z[w, :] = self.Solar_Angle[w]
            z[np.where(z == -999)] = np.nan
            self.ax1 = self.meteo_figures.add_subplot()
            rep = self.ax1.imshow(z.transpose(), cmap='magma')
            self.ax1.xaxis.label.set_color('blue')
            self.ax1.yaxis.label.set_color('blue')
            self.ax1.set_yticks(np.arange(12))
            self.ax1.set_yticklabels(self.ticks)
            self.ax1.tick_params(axis='x', color='blue', labelcolor='blue', rotation=60)
            self.ax1.set_xticks(np.arange(24))
            self.ax1.set_xticklabels(h_ticks)
            self.ax1.tick_params(axis='y', color='blue', labelcolor='blue')
            self.ax1.set_xlabel('Hour_GMT')
            self.ax1.set_title('Average hourly solar angles')
            self.meteo_figures.colorbar(rep, ax=self.ax1)
            self.meteo_figures.canvas.flush_events()
        #     h_ticks = [format(i, "02d") for i in range(24)]
        #     z = np.zeros((24, 12))
        #     for w in range(24):
        #         z[w, :] = self.Solar_Angle[w]
        #     z[np.where(z==-999)] = np.nan
        #     self.ax1 = self.meteo_figures.add_subplot(projection='3d')
        #     x2d, y2d = np.meshgrid(np.arange(12), np.arange(24))
        #     self.ax1.plot_surface(x2d, y2d, z,cmap='hot')
        #     # ax1.plot(x, y, color='blue', label=label)
        #     self.ax1.set_zlabel(self.units['Average_Hourly_Solar_Angles'])
        #     self.ax1.xaxis.label.set_color('blue')
        #     self.ax1.yaxis.label.set_color('blue')
        #     self.ax1.set_xticks(np.arange(12))
        #     self.ax1.set_xticklabels(self.ticks)
        #     self.ax1.tick_params(axis='x', color='blue', labelcolor='blue', rotation=60)
        #     self.ax1.set_yticks(np.arange(24))
        #     self.ax1.set_yticklabels(h_ticks)
        #     self.ax1.tick_params(axis='y', color='blue', labelcolor='blue', rotation=90)
        #     self.ax1.set_ylabel('Hour_GMT')
        #     self.ax1.set_title('Average hourly solar angles')
        #     self.meteo_figures.canvas.flush_events()

        elif num == 7:
            # 'Humidity at 2m': 7
            self.__plotxyy(y1=self.Relative_Humidity, y2=self.Specific_Humidity,
                           label1='Relative Humidity', label2='Specific Humidity',
                           y1_unit=self.units['Relative_Humidity'],
                           y2_unit=self.units['Specific_Humidity'],
                           fig_title='Wind_Speed at 2m')
        elif num == 8:
            # 'Wind_Speed at 2m': 8
            self.__plotxy(y=self.Wind_Speed, label='Wind speed',
                          y_unit=self.units['Wind_Speed'], fig_title='Wind_Speed at 2m')
        elif num == 9:
            # 'Temperature at 2m': 9
            self.__plotxy(y=self.Temperature, label='Temperature',
                          y_unit=self.units['Temperature'], fig_title='Temperature at 2m')

    def __plotxyy(self, y1, y2, label1='', label2='', y1_unit='', y2_unit='', fig_title=''):

        y1 = np.array(y1, dtype=np.float16)
        y1[np.where(y1 == -999)] = np.nan
        y2 = np.array(y2, dtype=np.float16)
        y2[np.where(y2 == -999)] = np.nan
        self.ax1 = self.meteo_figures.subplots()
        lns1 = self.ax1.plot(np.arange(12), y1, color='blue', label=label1)
        self.ax1.set_xticks(np.arange(12))
        self.ax1.set_xticklabels(self.ticks)
        self.ax1.set_ylabel(y1_unit)
        self.ax1.yaxis.label.set_color('blue')
        self.ax1.tick_params(axis='y', color='blue', labelcolor='blue')
        self.ax2 = self.ax1.twinx()
        lns2 = self.ax2.plot(np.arange(12), y2, color='red', label=label2)
        self.ax2.set_xticks(np.arange(12))
        self.ax2.set_xticklabels(self.ticks)
        self.ax2.set_ylabel(y2_unit)
        self.ax2.yaxis.label.set_color('red')
        self.ax2.tick_params(axis='y', color='red', labelcolor='red')
        self.ax1.tick_params(axis='x', rotation=60)
        self.ax1.set_title(fig_title)
        # added legends
        lns = lns1 + lns2
        labs = [lv.get_label() for lv in lns]
        self.ax1.legend(lns, labs, loc=0)
        self.meteo_figures.canvas.flush_events()

    def __plotxy(self, y, label='', y_unit='', fig_title=''):

        y = np.array(y, dtype=np.float16)
        y[np.where(y == -999)] = np.nan
        self.ax1 = self.meteo_figures.subplots()
        self.ax1.plot(np.arange(12), y, color='blue', label=label)
        self.ax1.set_xticks(np.arange(12))
        self.ax1.set_xticklabels(self.ticks)
        self.ax1.set_ylabel(y_unit)
        self.ax1.yaxis.label.set_color('blue')
        self.ax1.tick_params(axis='y', color='blue', labelcolor='blue')
        self.ax1.tick_params(axis='x', rotation=60)
        self.ax1.set_title(fig_title)
        # added legends
        self.ax1.legend()
        self.meteo_figures.canvas.flush_events()
