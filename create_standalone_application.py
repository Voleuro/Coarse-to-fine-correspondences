import os

# Automatically creates a standalone executable out of the command line interface
os.system("pyinstaller app.py -n CoFiNet --add-data configs;configs --add-data weights;weights")