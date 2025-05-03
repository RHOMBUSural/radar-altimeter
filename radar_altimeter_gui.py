import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType, ModulationType, CombinedSurface

class RadarAltimeterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Радиовысотомер")
        
        # Создаем экземпляр радиовысотомера
        self.altimeter = RadarAltimeter()
        
        # Устанавливаем начальные параметры
        self.altimeter.antenna_gain = 10  # Коэффициент усиления антенны в дБ
        self.altimeter.tx_power = 0.1    # Мощность передатчика в Вт
        
        # Флаг для отображения спектра
        self.show_spectrum = False
        
        # Создаем основной фрейм
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем фрейм для управления параметрами
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Параметры", padding="5")
        self.control_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Центральная частота
        ttk.Label(self.control_frame, text="Центральная частота (ГГц):").grid(row=0, column=0, padx=5, pady=5)
        self.f0_var = tk.DoubleVar(value=self.altimeter.f0 / 1e9)
        self.f0_entry = ttk.Entry(self.control_frame, textvariable=self.f0_var)
        self.f0_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Полоса частот
        ttk.Label(self.control_frame, text="Полоса частот (МГц):").grid(row=1, column=0, padx=5, pady=5)
        self.bandwidth_var = tk.DoubleVar(value=self.altimeter.B / 1e6)
        self.bandwidth_entry = ttk.Entry(self.control_frame, textvariable=self.bandwidth_var)
        self.bandwidth_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Тип модуляции
        ttk.Label(self.control_frame, text="Тип модуляции:").grid(row=2, column=0, padx=5, pady=5)
        self.modulation_var = tk.StringVar(value=ModulationType.LFM.value)
        self.modulation_combo = ttk.Combobox(self.control_frame, textvariable=self.modulation_var)
        self.modulation_combo['values'] = [mt.value for mt in ModulationType]
        self.modulation_combo.grid(row=2, column=1, padx=5, pady=5)
        self.modulation_combo.bind('<<ComboboxSelected>>', lambda e: self.update_modulation_controls())
        
        # Параметры импульсной модуляции
        self.pulse_frame = ttk.LabelFrame(self.control_frame, text="Параметры ИМ", padding="5")
        self.pulse_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Длительность импульса
        ttk.Label(self.pulse_frame, text="Длительность импульса (мкс):").grid(row=0, column=0, padx=5, pady=2)
        self.pulse_width_var = tk.DoubleVar(value=self.altimeter.pulse_width * 1e6)
        self.pulse_width_entry = ttk.Entry(self.pulse_frame, textvariable=self.pulse_width_var)
        self.pulse_width_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Период повторения
        ttk.Label(self.pulse_frame, text="Период повторения (мс):").grid(row=1, column=0, padx=5, pady=2)
        self.pulse_period_var = tk.DoubleVar(value=self.altimeter.pulse_period * 1e3)
        self.pulse_period_entry = ttk.Entry(self.pulse_frame, textvariable=self.pulse_period_var)
        self.pulse_period_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Коэффициент усиления антенны
        ttk.Label(self.control_frame, text="Усиление антенны (дБ):").grid(row=4, column=0, padx=5, pady=5)
        self.gain_var = tk.DoubleVar(value=self.altimeter.antenna_gain)
        self.gain_entry = ttk.Entry(self.control_frame, textvariable=self.gain_var)
        self.gain_entry.grid(row=4, column=1, padx=5, pady=5)
        
        # Мощность передатчика
        ttk.Label(self.control_frame, text="Мощность передатчика (Вт):").grid(row=5, column=0, padx=5, pady=5)
        self.power_var = tk.DoubleVar(value=self.altimeter.tx_power)
        self.power_entry = ttk.Entry(self.control_frame, textvariable=self.power_var)
        self.power_entry.grid(row=5, column=1, padx=5, pady=5)
        
        # Крен
        ttk.Label(self.control_frame, text="Крен (градусы):").grid(row=6, column=0, padx=5, pady=5)
        self.roll_var = tk.DoubleVar(value=self.altimeter.roll)
        self.roll_scale = ttk.Scale(self.control_frame, from_=-30, to=30, variable=self.roll_var, 
                                  orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
        self.roll_scale.grid(row=6, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.roll_label = ttk.Label(self.control_frame, text=f"{self.roll_var.get():.1f}°")
        self.roll_label.grid(row=6, column=2, padx=5, pady=5)
        
        # Тангаж
        ttk.Label(self.control_frame, text="Тангаж (градусы):").grid(row=7, column=0, padx=5, pady=5)
        self.pitch_var = tk.DoubleVar(value=self.altimeter.pitch)
        self.pitch_scale = ttk.Scale(self.control_frame, from_=-30, to=30, variable=self.pitch_var,
                                   orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
        self.pitch_scale.grid(row=7, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.pitch_label = ttk.Label(self.control_frame, text=f"{self.pitch_var.get():.1f}°")
        self.pitch_label.grid(row=7, column=2, padx=5, pady=5)
        
        # Высота
        ttk.Label(self.control_frame, text="Высота (м):").grid(row=8, column=0, padx=5, pady=5)
        self.height_var = tk.DoubleVar(value=10)  # Начальная высота 10 м
        self.height_entry = ttk.Entry(self.control_frame, textvariable=self.height_var)
        self.height_entry.grid(row=8, column=1, padx=5, pady=5)
        
        # Тип поверхности
        ttk.Label(self.control_frame, text="Тип поверхности:").grid(row=9, column=0, padx=5, pady=5)
        self.surface_var = tk.StringVar(value=SurfaceType.LAND.value)
        self.surface_combo = ttk.Combobox(self.control_frame, textvariable=self.surface_var)
        self.surface_combo['values'] = [st.value for st in SurfaceType]
        self.surface_combo.grid(row=9, column=1, padx=5, pady=5)
        
        # Переключатель спектр/сигнал
        self.spectrum_var = tk.BooleanVar(value=False)
        self.spectrum_check = ttk.Checkbutton(self.control_frame, text="Показать спектр", 
                                            variable=self.spectrum_var, command=self.toggle_spectrum)
        self.spectrum_check.grid(row=10, column=0, columnspan=2, pady=5)
        
        # Кнопка обновления
        self.update_button = ttk.Button(self.control_frame, text="Обновить", command=self.update_plots)
        self.update_button.grid(row=11, column=0, columnspan=2, pady=10)
        
        # Фрейм для комбинированного ландшафта
        self.landscape_frame = ttk.LabelFrame(self.control_frame, text="Комбинированный ландшафт", padding="5")
        self.landscape_frame.grid(row=12, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Параметры первого участка
        ttk.Label(self.landscape_frame, text="Участок 1:").grid(row=0, column=0, padx=5, pady=2)
        self.area1_type = ttk.Combobox(self.landscape_frame, values=[st.value for st in SurfaceType if st != SurfaceType.COMBINED])
        self.area1_type.grid(row=0, column=1, padx=5, pady=2)
        self.area1_type.set(SurfaceType.LAND.value)
        
        ttk.Label(self.landscape_frame, text="X1:").grid(row=0, column=2, padx=5, pady=2)
        self.area1_x1 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_x1.grid(row=0, column=3, padx=5, pady=2)
        self.area1_x1.insert(0, "0")
        
        ttk.Label(self.landscape_frame, text="X2:").grid(row=0, column=4, padx=5, pady=2)
        self.area1_x2 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_x2.grid(row=0, column=5, padx=5, pady=2)
        self.area1_x2.insert(0, "50")
        
        ttk.Label(self.landscape_frame, text="Y1:").grid(row=0, column=6, padx=5, pady=2)
        self.area1_y1 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_y1.grid(row=0, column=7, padx=5, pady=2)
        self.area1_y1.insert(0, "0")
        
        ttk.Label(self.landscape_frame, text="Y2:").grid(row=0, column=8, padx=5, pady=2)
        self.area1_y2 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_y2.grid(row=0, column=9, padx=5, pady=2)
        self.area1_y2.insert(0, "50")
        
        # Параметры второго участка
        ttk.Label(self.landscape_frame, text="Участок 2:").grid(row=1, column=0, padx=5, pady=2)
        self.area2_type = ttk.Combobox(self.landscape_frame, values=[st.value for st in SurfaceType if st != SurfaceType.COMBINED])
        self.area2_type.grid(row=1, column=1, padx=5, pady=2)
        self.area2_type.set(SurfaceType.SEA.value)
        
        ttk.Label(self.landscape_frame, text="X1:").grid(row=1, column=2, padx=5, pady=2)
        self.area2_x1 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_x1.grid(row=1, column=3, padx=5, pady=2)
        self.area2_x1.insert(0, "50")
        
        ttk.Label(self.landscape_frame, text="X2:").grid(row=1, column=4, padx=5, pady=2)
        self.area2_x2 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_x2.grid(row=1, column=5, padx=5, pady=2)
        self.area2_x2.insert(0, "100")
        
        ttk.Label(self.landscape_frame, text="Y1:").grid(row=1, column=6, padx=5, pady=2)
        self.area2_y1 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_y1.grid(row=1, column=7, padx=5, pady=2)
        self.area2_y1.insert(0, "0")
        
        ttk.Label(self.landscape_frame, text="Y2:").grid(row=1, column=8, padx=5, pady=2)
        self.area2_y2 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_y2.grid(row=1, column=9, padx=5, pady=2)
        self.area2_y2.insert(0, "50")
        
        # Кнопка 3D визуализации
        self.show_3d_button = ttk.Button(self.control_frame, text="Показать 3D ландшафт", command=self.show_3d_landscape)
        self.show_3d_button.grid(row=13, column=0, columnspan=3, pady=10)
        
        # Создаем фрейм для графиков
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # Создаем фигуру для графиков
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack()
        
        # Добавляем панель инструментов для масштабирования
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack()
        
        # Инициализируем графики
        self.update_plots()
        
    def toggle_spectrum(self):
        self.show_spectrum = self.spectrum_var.get()
        self.update_plots()
        
    def update_modulation_controls(self):
        # Обновляем видимость параметров импульсной модуляции
        is_pulse = self.modulation_var.get() == ModulationType.PULSE.value
        self.pulse_frame.grid_configure(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.pulse_frame.grid_remove() if not is_pulse else self.pulse_frame.grid()
        
        # Обновляем видимость полосы частот
        is_lfm = self.modulation_var.get() == ModulationType.LFM.value
        self.bandwidth_entry.grid_configure(row=1, column=1, padx=5, pady=5)
        self.bandwidth_entry.grid_remove() if not is_lfm else self.bandwidth_entry.grid()
        
        # Обновляем графики
        self.update_plots()

    def show_3d_landscape(self):
        # Создаем новое окно для 3D визуализации
        landscape_window = tk.Toplevel(self.root)
        landscape_window.title("3D визуализация ландшафта")
        
        # Создаем фигуру для 3D графика
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Создаем сетку для поверхности
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        
        # Создаем комбинированную поверхность
        combined_surface = CombinedSurface([
            (SurfaceType(self.area1_type.get()), 
             float(self.area1_x1.get()), float(self.area1_x2.get()),
             float(self.area1_y1.get()), float(self.area1_y2.get())),
            (SurfaceType(self.area2_type.get()),
             float(self.area2_x1.get()), float(self.area2_x2.get()),
             float(self.area2_y1.get()), float(self.area2_y2.get()))
        ])
        
        # Создаем Z-координаты в зависимости от типа поверхности
        Z = np.zeros_like(X)
        colors = np.zeros_like(X, dtype=object)
        
        # Цвета для разных типов поверхности
        color_map = {
            SurfaceType.SEA: 'blue',
            SurfaceType.LAND: 'green',
            SurfaceType.FOREST: 'darkgreen',
            SurfaceType.URBAN: 'gray',
            SurfaceType.ICE: 'lightblue',
            SurfaceType.DESERT: 'yellow'
        }
        
        # Высоты для разных типов поверхности
        height_map = {
            SurfaceType.SEA: 0,
            SurfaceType.ICE: 5,
            SurfaceType.LAND: 10,
            SurfaceType.DESERT: 15,
            SurfaceType.FOREST: 20,
            SurfaceType.URBAN: 30
        }
        
        for i in range(len(x)):
            for j in range(len(y)):
                surface_type = combined_surface.get_surface_type(x[i], y[j])
                Z[i,j] = height_map[surface_type]
                colors[i,j] = color_map[surface_type]
        
        # Отображаем поверхность с цветами
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)
        
        # Добавляем легенду
        legend_elements = []
        for surface_type, color in color_map.items():
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=f"{surface_type.value} ({height_map[surface_type]}м)"))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Настраиваем оси
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Высота (м)')
        
        # Устанавливаем начальный угол обзора
        ax.view_init(elev=30, azim=45)
        
        # Добавляем сетку
        ax.grid(True)
        
        # Добавляем подпись с текущими углами крена и тангажа
        ax.text2D(0.02, 0.95, f"Крен: {self.roll_var.get():.1f}°\nТангаж: {self.pitch_var.get():.1f}°", 
                 transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Создаем холст для отображения графика
        canvas = FigureCanvasTkAgg(fig, master=landscape_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Добавляем панель инструментов
        toolbar = NavigationToolbar2Tk(canvas, landscape_window)
        toolbar.update()
        toolbar.pack()
        
        # Добавляем кнопку для сохранения графика
        def save_plot():
            file_path = tk.filedialog.asksaveasfilename(defaultextension=".png",
                                                      filetypes=[("PNG files", "*.png"),
                                                                ("All files", "*.*")])
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
        
        save_button = ttk.Button(landscape_window, text="Сохранить график", command=save_plot)
        save_button.pack(pady=5)

    def update_plots(self):
        # Обновляем параметры радиовысотомера
        self.altimeter.f0 = self.f0_var.get() * 1e9
        self.altimeter.B = self.bandwidth_var.get() * 1e6
        self.altimeter.antenna_gain = self.gain_var.get()
        self.altimeter.tx_power = self.power_var.get()
        self.altimeter.roll = self.roll_var.get()
        self.altimeter.pitch = self.pitch_var.get()
        self.altimeter.modulation_type = ModulationType(self.modulation_var.get())
        
        # Обновляем метки углов
        self.roll_label.config(text=f"{self.roll_var.get():.1f}°")
        self.pitch_label.config(text=f"{self.pitch_var.get():.1f}°")
        
        # Обновляем параметры импульсной модуляции
        if self.altimeter.modulation_type == ModulationType.PULSE:
            self.altimeter.pulse_width = self.pulse_width_var.get() * 1e-6
            self.altimeter.pulse_period = self.pulse_period_var.get() * 1e-3
        
        # Создаем параметры поверхности
        if self.surface_var.get() == SurfaceType.COMBINED.value:
            # Создаем комбинированную поверхность
            combined_surface = CombinedSurface([
                (SurfaceType(self.area1_type.get()), 
                 float(self.area1_x1.get()), float(self.area1_x2.get()),
                 float(self.area1_y1.get()), float(self.area1_y2.get())),
                (SurfaceType(self.area2_type.get()),
                 float(self.area2_x1.get()), float(self.area2_x2.get()),
                 float(self.area2_y1.get()), float(self.area2_y2.get()))
            ])
            surface_params = SurfaceParameters(SurfaceType.COMBINED)
        else:
            surface_type = SurfaceType(self.surface_var.get())
            surface_params = SurfaceParameters(surface_type)
        
        # Получаем сигналы
        t, rx_signal = self.altimeter.simulate_reflection(self.height_var.get(), surface_params)
        _, tx_signal = self.altimeter.generate_chirp_signal(self.altimeter.T, self.altimeter.B)
        
        # Очищаем графики
        self.ax1.clear()
        self.ax2.clear()
        
        if self.show_spectrum:
            # Вычисляем спектр
            tx_spectrum = np.fft.fft(tx_signal)
            rx_spectrum = np.fft.fft(rx_signal)
            freq = np.fft.fftfreq(len(t), t[1] - t[0])
            
            # Отображаем спектр
            self.ax1.plot(freq / 1e6, np.abs(tx_spectrum))
            self.ax1.set_title('Спектр передаваемого сигнала')
            self.ax1.set_xlabel('Частота (МГц)')
            self.ax1.set_ylabel('Амплитуда')
            
            self.ax2.plot(freq / 1e6, np.abs(rx_spectrum))
            self.ax2.set_title('Спектр отраженного сигнала')
            self.ax2.set_xlabel('Частота (МГц)')
            self.ax2.set_ylabel('Амплитуда')
        else:
            # Отображаем сигналы во временной области
            self.ax1.plot(t * 1e6, tx_signal)
            self.ax1.set_title('Передаваемый сигнал')
            self.ax1.set_xlabel('Время (мкс)')
            self.ax1.set_ylabel('Амплитуда')
            
            # Отображаем отраженный сигнал
            self.ax2.plot(t * 1e6, rx_signal)
            self.ax2.set_title('Отраженный сигнал')
            self.ax2.set_xlabel('Время (мкс)')
            self.ax2.set_ylabel('Амплитуда')
            
            # Если используется комбинированный ландшафт, добавляем информацию о типах поверхности
            if self.surface_var.get() == SurfaceType.COMBINED.value:
                # Вычисляем время задержки для каждого участка
                delay1 = 2 * self.height_var.get() / self.altimeter.c * 1e6  # в микросекундах
                delay2 = 2 * self.height_var.get() / self.altimeter.c * 1e6  # в микросекундах
                
                # Добавляем вертикальные линии и подписи для каждого участка
                self.ax2.axvline(x=delay1, color='r', linestyle='--', alpha=0.5)
                self.ax2.axvline(x=delay2, color='b', linestyle='--', alpha=0.5)
                
                # Добавляем подписи с типами поверхности
                self.ax2.text(delay1, np.max(rx_signal) * 0.9, 
                            f"{self.area1_type.get()}\n({self.area1_x1.get()}-{self.area1_x2.get()}м)",
                            color='r', ha='center', bbox=dict(facecolor='white', alpha=0.8))
                self.ax2.text(delay2, np.max(rx_signal) * 0.8,
                            f"{self.area2_type.get()}\n({self.area2_x1.get()}-{self.area2_x2.get()}м)",
                            color='b', ha='center', bbox=dict(facecolor='white', alpha=0.8))
                
                # Добавляем информацию о высоте и углах
                self.ax2.text(0.02, 0.95, 
                            f"Высота: {self.height_var.get()}м\n"
                            f"Крен: {self.roll_var.get():.1f}°\n"
                            f"Тангаж: {self.pitch_var.get():.1f}°",
                            transform=self.ax2.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8))
        
        # Обновляем холст
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarAltimeterGUI(root)
    root.mainloop() 