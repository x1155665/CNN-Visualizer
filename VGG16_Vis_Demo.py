import sys
from PyQt5.QtWidgets import QApplication

from VGG16_Vis_Demo_View import VGG16_Vis_Demo_View

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VGG16_Vis_Demo_View()
    sys.exit(app.exec_())