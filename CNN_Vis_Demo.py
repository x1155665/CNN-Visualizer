import sys
from PyQt5.QtWidgets import QApplication

from CNN_Vis_Demo_View import CNN_Vis_Demo_View
from CNN_Vis_Demo_Model import  CNN_Vis_Demo_Model
from CNN_Vis_Demo_Ctl import CNN_Vis_Demo_Ctl

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = CNN_Vis_Demo_Model()
    ctl = CNN_Vis_Demo_Ctl(model)
    ex = CNN_Vis_Demo_View(model, ctl)
    sys.exit(app.exec_())