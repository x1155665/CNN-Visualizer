import sys
from PyQt5.QtWidgets import QApplication

from NN_Vis_Demo_View import NN_Vis_Demo_View
from NN_Vis_Demo_Model import  NN_Vis_Demo_Model
from NN_Vis_Demo_Ctl import NN_Vis_Demo_Ctl

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = NN_Vis_Demo_Model()
    ctl = NN_Vis_Demo_Ctl(model)
    ex = NN_Vis_Demo_View(model, ctl)
    sys.exit(app.exec_())