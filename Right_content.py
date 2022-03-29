import ipyvolume as ipv
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


class right_content:
    def __init__(self):
        self.zoom_scene = ipv.figure(width=250, height=250)
        self.proj_scene = plt.figure(num=1, figsize=(2.5, 2.5))

    def design(self):
        self.proj_scene.canvas.header_visible = False
        # self.proj_scene.canvas.layout.width = '250px'
        # proj_scene.canvas.layout.width = '100px'

        right_content = widgets.VBox([self.zoom_scene, self.proj_scene.canvas])
        
        return right_content
