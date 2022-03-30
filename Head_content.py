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
        # self.title = widgets.Label(value="r",height="250px", )
        self.title = widgets.HTML(
                    value=" <h1 style='color:blue;margin-:100px'; text-align:center><u><b>Solar collector simulator</b></u></h1> ",
                    # placeholder='Some HTML',
                    # description='Some HTML',

                    )

    def design(self):
        box_layout = widgets.Layout(
                # display='flex',
                flex_flow='column',
                margin='auto auto 50px auto',
                # align_items='stretch',
                width='auto',
                justify_content='center',
                align_items = 'center')

        header_cont = widgets.VBox([
            self.title,
            widgets.HBox([self.collector_type, self.source_geometry], width='50')],
            # display='flex',
            # flex_flow='column',
            layout = box_layout,
        )

        return header_cont
