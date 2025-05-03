import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType, ModulationType

class RadarAltimeterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Радиовысотомер")
        
        # Создаем экземпляр радиовысотомера
        self.altimeter = RadarAltimeter()
        
        # Добавляем новые параметры
        self.altimeter.antenna_gain = 30  # Коэффициент усиления антенны в дБ
        self.altimeter.tx_power = 1000    # Мощность передатчика в Вт
        
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
        
        # Тангаж
        ttk.Label(self.control_frame, text="Тангаж (градусы):").grid(row=7, column=0, padx=5, pady=5)
        self.pitch_var = tk.DoubleVar(value=self.altimeter.pitch)
        self.pitch_scale = ttk.Scale(self.control_frame, from_=-30, to=30, variable=self.pitch_var,
                                   orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
        self.pitch_scale.grid(row=7, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Высота
        ttk.Label(self.control_frame, text="Высота (м):").grid(row=8, column=0, padx=5, pady=5)
        self.height_var = tk.DoubleVar(value=1000)
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

    def update_plots(self):
        # Обновляем параметры радиовысотомера
        self.altimeter.f0 = self.f0_var.get() * 1e9
        self.altimeter.B = self.bandwidth_var.get() * 1e6
        self.altimeter.antenna_gain = self.gain_var.get()
        self.altimeter.tx_power = self.power_var.get()
        self.altimeter.roll = self.roll_var.get()
        self.altimeter.pitch = self.pitch_var.get()
        self.altimeter.modulation_type = ModulationType(self.modulation_var.get())
        
        # Обновляем параметры импульсной модуляции
        if self.altimeter.modulation_type == ModulationType.PULSE:
            self.altimeter.pulse_width = self.pulse_width_var.get() * 1e-6
            self.altimeter.pulse_period = self.pulse_period_var.get() * 1e-3
        
        # Создаем параметры поверхности
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
            
            self.ax2.plot(t * 1e6, rx_signal)
            self.ax2.set_title('Отраженный сигнал')
            self.ax2.set_xlabel('Время (мкс)')
            self.ax2.set_ylabel('Амплитуда')
        
        # Обновляем холст
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarAltimeterGUI(root)
    root.mainloop() 