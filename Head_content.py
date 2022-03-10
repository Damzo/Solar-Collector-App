import ipywidgets as widgets



class head_content:
    style = {'description_width': 'initial'}
    item_layout = widgets.Layout(justify_content='center', width='auto')
    collector_type = widgets.Dropdown(
            options=[('Reflective parabola', 1), ('Ring arrays collector', 2), ('Cylindrical collector', 3)],
            value=1,
            desciption='Choose the type of collector',
            disabled=False)
    source_geometry = widgets.Dropdown(
            options=[('ideal point source', 1), ('large light source', 2)],
            value=1,
            desciption='Select sun geometry',
            disabled=False)

    def __init__(self):
        self.title = widgets.Label(value="Light Collector Simulator")


    def design(self):
        header_cont = widgets.VBox([
            self.title,
            widgets.HBox([self.collector_type, self.source_geometry], width='auto')],
            display='flex',
            flex_flow='column',
        )

        return header_cont
