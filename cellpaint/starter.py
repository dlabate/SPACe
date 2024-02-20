import os
import sys
from PyQt6.QtWidgets import *

from main2 import experiment_holder
# from my_functions import add

class filedialogdemo(QWidget):
    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)

        layout = QVBoxLayout()
        self.btn = QPushButton("Find Folder")
        self.btn.clicked.connect(self.getfile)

        layout.addWidget(self.btn)
        self.le = QLabel("Hello")

        layout.addWidget(self.le)
        self.btn1 = QPushButton("Send Files for Analysis")
        self.btn1.clicked.connect(self.getfiles)
        layout.addWidget(self.btn1)

        self.contents = QListWidget()
        self.contents.setSelectionMode(self.contents.SelectionMode.MultiSelection)
        layout.addWidget(self.contents)
        self.setLayout(layout)
        self.setWindowTitle("Choose Files to be Analyzed")

    def getfile(self):
        global folder
        global exp_names

        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        exp_names = [name for name in os.listdir(folder) if name[0:7].isnumeric()]

        self.le.setText(folder)
        x=0
        for i in exp_names:
            self.contents.insertItem(x,i)
            x=x+1

    def getfiles(self):
        selectedList = self.contents.selectedItems()
        textList = []
        textList.clear()
        for i in selectedList:
            # print(type(i.text()))
            textList.append(i.text())
        # print(textList)
        # print(selectedList.text())
        self.hide()
        experiment_holder(folder, textList)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = filedialogdemo()
    ex.show()
    sys.exit(app.exec())