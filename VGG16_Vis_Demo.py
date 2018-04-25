import sys
from PyQt5.QtWidgets import QApplication

from VGG16_Vis_Demo_View import VGG16_Vis_Demo_View
from VGG16_Vis_Demo_Model import  VGG16_Vis_Demo_Model
from VGG16_Vis_Demo_Ctl import VGG16_Vis_Demo_Ctl

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = VGG16_Vis_Demo_Model()
    ctl = VGG16_Vis_Demo_Ctl(model)
    ex = VGG16_Vis_Demo_View(model, ctl)
    sys.exit(app.exec_())