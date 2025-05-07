import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class RadarAltimeterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Радиовысотомер")
        
        # Создаем основной фрейм
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем фрейм для графиков
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # Создаем фигуру для графиков
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack()
        
        # Создаем тестовые данные
        t = np.linspace(0, 1, 1000)
        self.ax1.plot(t, np.sin(2*np.pi*t))
        self.ax1.set_title('Тестовый график 1')
        
        self.ax2.plot(t, np.cos(2*np.pi*t))
        self.ax2.set_title('Тестовый график 2')
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarAltimeterGUI(root)
    root.mainloop() 