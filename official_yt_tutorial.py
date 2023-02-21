## https://www.youtube.com/watch?v=kkRXLrG5oMA

import dearpygui.dearpygui as dpg

dpg.create_context()

with dpg.window(label="Tutorial"):
	dpg.add_color_picker((255, 0, 255, 255), label="Texture",
                         no_side_preview=True, alpha_bar=True, width=200,
                         callback=lambda:print("callback"))


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()