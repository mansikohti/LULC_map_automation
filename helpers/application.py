import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QHBoxLayout, QComboBox, QMessageBox) 

from main_2 import *




class GET_UNCLASSIFIED_CLASS_APP(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Unclassified Class Automation")
        self.setGeometry(600, 200, 600, 400)

        layout = QVBoxLayout()
        ## file selection inputs
        self.input_folder_path = self.create_folder_input(layout, "Input folder path:")
        self.output_folder_path = self.create_folder_input(layout, "Output folder path:")


        ## Parameters
        self.buffer = self.create_line_edit(layout, "Buffer for Settlement (m):", "10")
        self.min_area_removed = self.create_line_edit(layout, "min settlement area to be removed (sq.m.)", "25")
        self.inset_buffer_for_AOI = self.create_line_edit(layout, "Inside Buffer for AOI", "0.2")

        
        ## Start processing Button
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.run_processing)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def create_file_input(self, layout, label_text):
        """Creates a file selection input."""
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.select_file(line_edit))
        hbox.addWidget(label)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        layout.addLayout(hbox)
        return line_edit
    
    def create_folder_input(self, layout, label_text):
        """Creates a folder selection input."""
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.select_folder(line_edit))
        hbox.addWidget(label)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        layout.addLayout(hbox)
        return line_edit
    
    def create_line_edit(self, layout, label_text, default_value):
        """Creates a labeled line edit with a default value."""
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit(default_value)
        hbox.addWidget(label)
        hbox.addWidget(line_edit)
        layout.addLayout(hbox)
        return line_edit
    

    def select_file(self, line_edit):
        """Opens a file dialog and sets the selected file path."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            line_edit.setText(file_path)

    def select_folder(self, line_edit):
        """Opens a folder dialog and sets the selected folder path."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            line_edit.setText(folder_path)

    def run_processing(self):
        try:
            input_folder_path = self.input_folder_path.text()
            output_folder_path = self.output_folder_path.text()

    
            buffer = float(self.buffer.text())
            min_area_removed = float(self.min_area_removed.text())
            inset_buffer = float(self.inset_buffer_for_AOI.text())
           
            os.makedirs(output_folder_path, exist_ok=True)

        
            processing( input_folder_path = input_folder_path, 
                        output_folder_path = output_folder_path,
                        buffer= buffer,
                        min_area_removed = min_area_removed,
                        inset_buffer_for_AOI= inset_buffer)

            QMessageBox.information(self, "Run Analysis", "Process complete successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GET_UNCLASSIFIED_CLASS_APP()
    window.show()
    sys.exit(app.exec())