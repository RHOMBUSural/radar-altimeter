import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType, ModulationType, CombinedSurface, SurfaceGradient, PolarizationType
import torch
from scipy import signal
import traceback
import logging

# Настраиваем логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='radar_altimeter.log'
)

class RadarAltimeterGUI:
    def __init__(self, root):
        try:
            logging.info("Инициализация GUI")
        self.root = root
        self.root.title("Радиовысотомер")
            
            # Устанавливаем минимальный размер окна
            self.root.minsize(1200, 800)
        
        # Создаем экземпляр радиовысотомера
            logging.info("Создание экземпляра радиовысотомера")
            self.altimeter = RadarAltimeter(use_gpu=torch.cuda.is_available())
        
            # Устанавливаем начальные параметры
            self.altimeter.antenna_gain = 10  # Коэффициент усиления антенны в дБ
            self.altimeter.tx_power = 0.1    # Мощность передатчика в Вт
        
        # Флаг для отображения спектра
        self.show_spectrum = False
            
            # Создаем основной контейнер с прокруткой
            logging.info("Создание основного контейнера")
            self.main_container = ttk.Frame(root)
            self.main_container.pack(fill=tk.BOTH, expand=True)
            
            # Создаем холст с прокруткой
            self.scroll_canvas = tk.Canvas(self.main_container)
            self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.scroll_canvas.yview)
            self.scrollable_frame = ttk.Frame(self.scroll_canvas)
            
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.scroll_canvas.configure(
                    scrollregion=self.scroll_canvas.bbox("all")
                )
            )
            
            self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
            
            # Размещаем холст и скроллбар
            self.scroll_canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
        
        # Создаем основной фрейм
            self.main_frame = ttk.Frame(self.scrollable_frame, padding="10")
            self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем фрейм для управления параметрами
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Параметры", padding="5")
            self.control_frame.pack(side="left", fill="y", padx=5, pady=5)
            
            # Создаем фрейм для графиков
            self.plot_frame = ttk.Frame(self.main_frame)
            self.plot_frame.pack(side="right", fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Добавляем новые элементы управления
            logging.info("Создание элементов управления")
            self.create_controls()
            
            # Создаем фигуру для графиков
            logging.info("Создание графиков")
            self.create_plots()
            
            # Инициализируем графики
            self.update_plots()
            
            # Настраиваем обработку колесика мыши для прокрутки
            self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            
            # Добавляем обработчик закрытия окна
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            logging.info("GUI успешно инициализирован")
            
        except Exception as e:
            logging.error(f"Ошибка при инициализации GUI: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def create_controls(self):
        """Создание элементов управления"""
        try:
            row = 0
            
            # Поляризация
            ttk.Label(self.control_frame, text="Поляризация:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.polarization_var = tk.StringVar(value=PolarizationType.VERTICAL.value)
            self.polarization_combo = ttk.Combobox(self.control_frame, textvariable=self.polarization_var, width=15)
            self.polarization_combo['values'] = [pt.value for pt in PolarizationType]
            self.polarization_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
            
            # Количество отражений
            ttk.Label(self.control_frame, text="Количество отражений:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.reflections_var = tk.IntVar(value=2)
            self.reflections_scale = ttk.Scale(self.control_frame, from_=1, to=5, variable=self.reflections_var,
                                            orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
            self.reflections_scale.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            self.reflections_label = ttk.Label(self.control_frame, text=f"{self.reflections_var.get()}")
            self.reflections_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
            row += 1
            
            # Использование GPU
            self.use_gpu_var = tk.BooleanVar(value=torch.cuda.is_available())
            self.gpu_check = ttk.Checkbutton(self.control_frame, text="Использовать GPU", 
                                           variable=self.use_gpu_var, command=self.toggle_gpu)
            self.gpu_check.grid(row=row, column=0, columnspan=2, pady=5, sticky="w")
            row += 1
        
        # Центральная частота
            ttk.Label(self.control_frame, text="Центральная частота (ГГц):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.f0_var = tk.DoubleVar(value=self.altimeter.f0 / 1e9)
            self.f0_entry = ttk.Entry(self.control_frame, textvariable=self.f0_var, width=15)
            self.f0_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
        
        # Полоса частот
            ttk.Label(self.control_frame, text="Полоса частот (МГц):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.bandwidth_var = tk.DoubleVar(value=self.altimeter.B / 1e6)
            self.bandwidth_entry = ttk.Entry(self.control_frame, textvariable=self.bandwidth_var, width=15)
            self.bandwidth_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
            
            # Коэффициент усиления антенны
            ttk.Label(self.control_frame, text="Коэффициент усиления (дБ):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.gain_var = tk.DoubleVar(value=self.altimeter.antenna_gain)
            self.gain_entry = ttk.Entry(self.control_frame, textvariable=self.gain_var, width=15)
            self.gain_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
            
            # Мощность передатчика
            ttk.Label(self.control_frame, text="Мощность передатчика (Вт):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.power_var = tk.DoubleVar(value=self.altimeter.tx_power)
            self.power_entry = ttk.Entry(self.control_frame, textvariable=self.power_var, width=15)
            self.power_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
        
        # Тип модуляции
            ttk.Label(self.control_frame, text="Тип модуляции:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.modulation_var = tk.StringVar(value=ModulationType.LFM.value)
            self.modulation_combo = ttk.Combobox(self.control_frame, textvariable=self.modulation_var, width=15)
        self.modulation_combo['values'] = [mt.value for mt in ModulationType]
            self.modulation_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.modulation_combo.bind('<<ComboboxSelected>>', lambda e: self.update_modulation_controls())
            row += 1
        
            # Фрейм для параметров импульсной модуляции
            self.pulse_frame = ttk.LabelFrame(self.control_frame, text="Параметры импульсной модуляции", padding="5")
            self.pulse_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Длительность импульса
            ttk.Label(self.pulse_frame, text="Длительность импульса (мкс):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pulse_width_var = tk.DoubleVar(value=self.altimeter.pulse_width * 1e6)
            self.pulse_width_entry = ttk.Entry(self.pulse_frame, textvariable=self.pulse_width_var, width=15)
            self.pulse_width_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Период повторения
            ttk.Label(self.pulse_frame, text="Период повторения (мс):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.pulse_period_var = tk.DoubleVar(value=self.altimeter.pulse_period * 1e3)
            self.pulse_period_entry = ttk.Entry(self.pulse_frame, textvariable=self.pulse_period_var, width=15)
            self.pulse_period_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            row += 1
        
        # Крен
            ttk.Label(self.control_frame, text="Крен (°):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.roll_var = tk.DoubleVar(value=0)
            self.roll_scale = ttk.Scale(self.control_frame, from_=-45, to=45, variable=self.roll_var,
                                  orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
            self.roll_scale.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            self.roll_label = ttk.Label(self.control_frame, text=f"{self.roll_var.get():.1f}°")
            self.roll_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
            row += 1
        
        # Тангаж
            ttk.Label(self.control_frame, text="Тангаж (°):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.pitch_var = tk.DoubleVar(value=0)
            self.pitch_scale = ttk.Scale(self.control_frame, from_=-45, to=45, variable=self.pitch_var,
                                   orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
            self.pitch_scale.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            self.pitch_label = ttk.Label(self.control_frame, text=f"{self.pitch_var.get():.1f}°")
            self.pitch_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")
            row += 1
        
        # Высота
            ttk.Label(self.control_frame, text="Высота (м):").grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.height_var = tk.DoubleVar(value=10)  # Начальная высота 10 м
            self.height_entry = ttk.Entry(self.control_frame, textvariable=self.height_var, width=15)
            self.height_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
        
        # Тип поверхности
            ttk.Label(self.control_frame, text="Тип поверхности:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.surface_var = tk.StringVar(value=SurfaceType.LAND.value)
            self.surface_combo = ttk.Combobox(self.control_frame, textvariable=self.surface_var, width=15)
        self.surface_combo['values'] = [st.value for st in SurfaceType]
            self.surface_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
        
        # Переключатель спектр/сигнал
        self.spectrum_var = tk.BooleanVar(value=False)
        self.spectrum_check = ttk.Checkbutton(self.control_frame, text="Показать спектр", 
                                            variable=self.spectrum_var, command=self.toggle_spectrum)
            self.spectrum_check.grid(row=row, column=0, columnspan=2, pady=5, sticky="w")
            row += 1
        
        # Кнопка обновления
        self.update_button = ttk.Button(self.control_frame, text="Обновить", command=self.update_plots)
            self.update_button.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
            row += 1
            
            # Фрейм для комбинированного ландшафта
            self.landscape_frame = ttk.LabelFrame(self.control_frame, text="Комбинированный ландшафт", padding="5")
            self.landscape_frame.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
            
            # Параметры первого участка
            ttk.Label(self.landscape_frame, text="Участок 1:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.area1_type = ttk.Combobox(self.landscape_frame, values=[st.value for st in SurfaceType if st != SurfaceType.COMBINED], width=10)
            self.area1_type.grid(row=0, column=1, padx=5, pady=2, sticky="w")
            self.area1_type.set(SurfaceType.LAND.value)
            
            ttk.Label(self.landscape_frame, text="X1:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
            self.area1_x1 = ttk.Entry(self.landscape_frame, width=5)
            self.area1_x1.grid(row=0, column=3, padx=5, pady=2, sticky="w")
            self.area1_x1.insert(0, "0")
            
            ttk.Label(self.landscape_frame, text="X2:").grid(row=0, column=4, padx=5, pady=2, sticky="w")
            self.area1_x2 = ttk.Entry(self.landscape_frame, width=5)
            self.area1_x2.grid(row=0, column=5, padx=5, pady=2, sticky="w")
            self.area1_x2.insert(0, "50")
            
            ttk.Label(self.landscape_frame, text="Y1:").grid(row=0, column=6, padx=5, pady=2, sticky="w")
            self.area1_y1 = ttk.Entry(self.landscape_frame, width=5)
            self.area1_y1.grid(row=0, column=7, padx=5, pady=2, sticky="w")
            self.area1_y1.insert(0, "0")
            
            ttk.Label(self.landscape_frame, text="Y2:").grid(row=0, column=8, padx=5, pady=2, sticky="w")
            self.area1_y2 = ttk.Entry(self.landscape_frame, width=5)
            self.area1_y2.grid(row=0, column=9, padx=5, pady=2, sticky="w")
            self.area1_y2.insert(0, "50")
            
            # Параметры второго участка
            ttk.Label(self.landscape_frame, text="Участок 2:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.area2_type = ttk.Combobox(self.landscape_frame, values=[st.value for st in SurfaceType if st != SurfaceType.COMBINED], width=10)
            self.area2_type.grid(row=1, column=1, padx=5, pady=2, sticky="w")
            self.area2_type.set(SurfaceType.SEA.value)
            
            ttk.Label(self.landscape_frame, text="X1:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
            self.area2_x1 = ttk.Entry(self.landscape_frame, width=5)
            self.area2_x1.grid(row=1, column=3, padx=5, pady=2, sticky="w")
            self.area2_x1.insert(0, "50")
            
            ttk.Label(self.landscape_frame, text="X2:").grid(row=1, column=4, padx=5, pady=2, sticky="w")
            self.area2_x2 = ttk.Entry(self.landscape_frame, width=5)
            self.area2_x2.grid(row=1, column=5, padx=5, pady=2, sticky="w")
            self.area2_x2.insert(0, "100")
            
            ttk.Label(self.landscape_frame, text="Y1:").grid(row=1, column=6, padx=5, pady=2, sticky="w")
            self.area2_y1 = ttk.Entry(self.landscape_frame, width=5)
            self.area2_y1.grid(row=1, column=7, padx=5, pady=2, sticky="w")
            self.area2_y1.insert(0, "0")
            
            ttk.Label(self.landscape_frame, text="Y2:").grid(row=1, column=8, padx=5, pady=2, sticky="w")
            self.area2_y2 = ttk.Entry(self.landscape_frame, width=5)
            self.area2_y2.grid(row=1, column=9, padx=5, pady=2, sticky="w")
            self.area2_y2.insert(0, "50")
            
            # Кнопка 3D визуализации
            self.show_3d_button = ttk.Button(self.control_frame, text="Показать 3D ландшафт", command=self.show_3d_landscape)
            self.show_3d_button.grid(row=row+1, column=0, columnspan=3, pady=10, sticky="ew")
            
        except Exception as e:
            logging.error(f"Ошибка при создании элементов управления: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def create_plots(self):
        """Создание графиков"""
        try:
        # Создаем фигуру для графиков
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
            self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Добавляем панель инструментов для масштабирования
            self.toolbar = NavigationToolbar2Tk(self.plot_canvas, self.plot_frame)
        self.toolbar.update()
            self.toolbar.pack(fill=tk.X)
            
        except Exception as e:
            logging.error(f"Ошибка при создании графиков: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def on_closing(self):
        """Обработчик закрытия окна"""
        try:
            logging.info("Закрытие приложения")
            plt.close('all')
            self.root.destroy()
        except Exception as e:
            logging.error(f"Ошибка при закрытии приложения: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def _on_mousewheel(self, event):
        """Обработчик прокрутки мыши"""
        try:
            self.scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except Exception as e:
            logging.error(f"Ошибка при обработке прокрутки мыши: {str(e)}")
            logging.error(traceback.format_exc())
        
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
        """Показывает 3D визуализацию ландшафта"""
        try:
            logging.info("Создание окна 3D визуализации")
            # Создаем новое окно для 3D визуализации
            landscape_window = tk.Toplevel(self.root)
            landscape_window.title("3D визуализация ландшафта")
            landscape_window.minsize(800, 600)
            
            # Создаем фрейм для управления
            control_frame = ttk.Frame(landscape_window, padding="5")
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Добавляем переключатель для отраженного сигнала
            self.show_reflection_var = tk.BooleanVar(value=True)
            reflection_check = ttk.Checkbutton(control_frame, text="Показать отраженный сигнал",
                                             variable=self.show_reflection_var,
                                             command=self.update_3d_plot)
            reflection_check.pack(side=tk.LEFT, padx=5)
            
            # Добавляем прогресс-бар
            self.progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
            progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Создаем фигуру для 3D графика
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Уменьшаем количество точек для оптимизации
            resolution = 50  # Уменьшаем разрешение для более быстрой отрисовки
            x = np.linspace(0, 100, resolution)
            y = np.linspace(0, 100, resolution)
            X, Y = np.meshgrid(x, y)
            
            # Кэшируем данные поверхности
            if not hasattr(self, 'cached_surface_data'):
                self.cached_surface_data = {}
            
            # Генерируем данные поверхности
            self.progress_var.set(0)
            landscape_window.update()
            
            # Функция для генерации шума Перлина
            def generate_perlin_noise(x, y, scale=0.1, octaves=6, persistence=0.5, lacunarity=2.0):
                noise = np.zeros_like(x)
                amplitude = 1.0
                frequency = scale
                
                for _ in range(octaves):
                    nx = x * frequency
                    ny = y * frequency
                    n = np.sin(nx) * np.cos(ny) + np.sin(ny) * np.cos(nx)
                    noise += amplitude * n
                    amplitude *= persistence
                    frequency *= lacunarity
                
                return noise
            
            # Генерируем базовый шум для всей поверхности
            base_noise = generate_perlin_noise(X, Y, scale=0.05)
            self.progress_var.set(20)
            landscape_window.update()
            
            # Создаем комбинированную поверхность
            combined_surface = CombinedSurface([
                (SurfaceType(self.area1_type.get()), 
                 float(self.area1_x1.get()), float(self.area1_x2.get()),
                 float(self.area1_y1.get()), float(self.area1_y2.get())),
                (SurfaceType(self.area2_type.get()),
                 float(self.area2_x1.get()), float(self.area2_x2.get()),
                 float(self.area2_y1.get()), float(self.area2_y2.get()))
            ])
            
            # Создаем Z-координаты и цвета
            Z = np.zeros_like(X)
            colors = np.zeros_like(X, dtype=object)
            signal_strength = np.zeros_like(X)
            
            # Цвета и высоты для разных типов поверхности
            color_map = {
                SurfaceType.SEA: 'blue',
                SurfaceType.LAND: 'green',
                SurfaceType.FOREST: 'darkgreen',
                SurfaceType.URBAN: 'gray',
                SurfaceType.ICE: 'lightblue',
                SurfaceType.DESERT: 'yellow'
            }
            
            base_height_map = {
                SurfaceType.SEA: 0,
                SurfaceType.ICE: 5,
                SurfaceType.LAND: 10,
                SurfaceType.DESERT: 15,
                SurfaceType.FOREST: 20,
                SurfaceType.URBAN: 30
            }
            
            height_variation_map = {
                SurfaceType.SEA: 0.5,
                SurfaceType.ICE: 2.0,
                SurfaceType.LAND: 5.0,
                SurfaceType.DESERT: 8.0,
                SurfaceType.FOREST: 10.0,
                SurfaceType.URBAN: 3.0
            }
            
            # Вычисляем данные поверхности
            total_points = len(x) * len(y)
            processed_points = 0
            
            for i in range(len(x)):
                for j in range(len(y)):
                    surface_type = combined_surface.get_surface_type(x[i], y[j])
                    
                    # Вычисляем высоту и цвет
                    base_height = base_height_map[surface_type]
                    height_variation = height_variation_map[surface_type]
                    noise_factor = base_noise[i,j] * height_variation
                    slope_factor = 1.0 + 0.5 * np.sin(x[i] * 0.1) * np.cos(y[j] * 0.1)
                    Z[i,j] = base_height + noise_factor * slope_factor
                    
                    # Вычисляем цвет
                    base_color = color_map[surface_type]
                    if surface_type == SurfaceType.LAND:
                        height_factor = (Z[i,j] - base_height) / height_variation
                        colors[i,j] = plt.cm.terrain(height_factor)
                    elif surface_type == SurfaceType.FOREST:
                        height_factor = (Z[i,j] - base_height) / height_variation
                        colors[i,j] = plt.cm.Greens(0.3 + 0.7 * height_factor)
                    else:
                        colors[i,j] = base_color
                    
                    # Вычисляем силу сигнала
                    dx = x[i] - 50
                    dy = y[j] - 50
                    distance = np.sqrt(dx**2 + dy**2)
                    grazing_angle = np.arctan2(self.height_var.get(), distance)
                    surface_params = SurfaceParameters(surface_type)
                    reflection_coeff = self.altimeter.calculate_reflection_coefficient(grazing_angle, surface_params)
                    attenuation = 1.0 / (distance**2 + self.height_var.get()**2)
                    signal_strength[i,j] = reflection_coeff * attenuation
                    
                    processed_points += 1
                    if processed_points % 100 == 0:
                        progress = 20 + (processed_points / total_points) * 60
                        self.progress_var.set(progress)
                        landscape_window.update()
            
            # Нормализуем силу сигнала
            signal_strength = signal_strength / np.max(signal_strength)
            
            # Кэшируем данные
            self.cached_surface_data = {
                'X': X,
                'Y': Y,
                'Z': Z,
                'colors': colors,
                'signal_strength': signal_strength
            }
            
            # Отображаем поверхность
            surf = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)
            signal_surf = ax.plot_surface(X, Y, Z + 5, facecolors=plt.cm.viridis(signal_strength), alpha=0.6)
            
            # Настраиваем график
            self.setup_3d_plot(ax)
            
            # Создаем холст
            canvas = FigureCanvasTkAgg(fig, master=landscape_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Добавляем панель инструментов
            toolbar = NavigationToolbar2Tk(canvas, landscape_window)
            toolbar.update()
            toolbar.pack()
            
            # Сохраняем ссылки на объекты
            self.ax = ax
            self.signal_surf = signal_surf
            self.canvas = canvas
            
            self.progress_var.set(100)
            logging.info("3D визуализация успешно создана")
            
        except Exception as e:
            logging.error(f"Ошибка при создании 3D визуализации: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def setup_3d_plot(self, ax):
        """Настраивает 3D график"""
        try:
            # Добавляем легенду
            legend_elements = []
            color_map = {
                SurfaceType.SEA: 'blue',
                SurfaceType.LAND: 'green',
                SurfaceType.FOREST: 'darkgreen',
                SurfaceType.URBAN: 'gray',
                SurfaceType.ICE: 'lightblue',
                SurfaceType.DESERT: 'yellow'
            }
            base_height_map = {
                SurfaceType.SEA: 0,
                SurfaceType.ICE: 5,
                SurfaceType.LAND: 10,
                SurfaceType.DESERT: 15,
                SurfaceType.FOREST: 20,
                SurfaceType.URBAN: 30
            }
            
            for surface_type, color in color_map.items():
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                                markerfacecolor=color, markersize=10,
                                                label=f"{surface_type.value} ({base_height_map[surface_type]}м)"))
            
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
            ax.text2D(0.02, 0.95, f"Крен: {self.roll_var.get():.1f}°\nТангаж: {self.pitch_var.get():.1f}°\nВысота: {self.height_var.get()}м", 
                     transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
        except Exception as e:
            logging.error(f"Ошибка при настройке 3D графика: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def update_3d_plot(self):
        """Обновляет 3D график при изменении параметров"""
        try:
            if not hasattr(self, 'cached_surface_data'):
                return
            
            # Получаем кэшированные данные
            X = self.cached_surface_data['X']
            Y = self.cached_surface_data['Y']
            Z = self.cached_surface_data['Z']
            colors = self.cached_surface_data['colors']
            signal_strength = self.cached_surface_data['signal_strength']
            
            # Очищаем предыдущие поверхности
            self.ax.clear()
            
            # Отображаем поверхность с цветами
            surf = self.ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)
            
            # Отображаем силу отраженного сигнала, если включено
            if self.show_reflection_var.get():
                signal_surf = self.ax.plot_surface(X, Y, Z + 5, facecolors=plt.cm.viridis(signal_strength), alpha=0.6)
                self.signal_surf = signal_surf
            
            # Настраиваем график
            self.setup_3d_plot(self.ax)
            
            # Обновляем холст
            self.canvas.draw()
            
            logging.info("3D график успешно обновлен")
            
        except Exception as e:
            logging.error(f"Ошибка при обновлении 3D графика: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def toggle_gpu(self):
        """Переключение использования GPU"""
        self.altimeter.use_gpu = self.use_gpu_var.get()
        self.altimeter.device = torch.device("cuda" if self.altimeter.use_gpu and torch.cuda.is_available() else "cpu")
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
        
        # Обновляем метки углов
        self.roll_label.config(text=f"{self.roll_var.get():.1f}°")
        self.pitch_label.config(text=f"{self.pitch_var.get():.1f}°")
        self.reflections_label.config(text=f"{self.reflections_var.get()}")
        
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
            surface_params = SurfaceParameters(SurfaceType(self.surface_var.get()))
        
        # Создаем градиенты поверхности
        height_map = np.zeros((100, 100))
        gradient = SurfaceGradient(height_map)
        
        # Моделируем отражение
        t, rx_signal = self.altimeter.simulate_multipath_reflection(
            self.height_var.get(),
            surface_params,
            gradient,
            self.reflections_var.get()
        )
        
        # Генерируем передаваемый сигнал
        _, tx_signal = self.altimeter.generate_signal(self.altimeter.T, self.altimeter.B)
        
        # Очищаем графики
        self.ax1.clear()
        self.ax2.clear()
        
        # Отображаем сигналы
        self.ax1.plot(t * 1e6, tx_signal, 'r--', label='Передаваемый', alpha=0.5)
        self.ax1.plot(t * 1e6, rx_signal, 'b-', label='Отраженный')
        self.ax1.set_title('Сигналы во временной области')
        self.ax1.set_xlabel('Время (мкс)')
        self.ax1.set_ylabel('Амплитуда')
        self.ax1.grid(True)
        self.ax1.legend()
        
        if self.show_spectrum:
            # Вычисляем спектр
            f, tx_spectrum = signal.welch(tx_signal, fs=1e9, nperseg=1024)
            f, rx_spectrum = signal.welch(rx_signal, fs=1e9, nperseg=1024)
            
            self.ax2.plot(f / 1e6, 10 * np.log10(tx_spectrum), 'r--', label='Передаваемый', alpha=0.5)
            self.ax2.plot(f / 1e6, 10 * np.log10(rx_spectrum), 'b-', label='Отраженный')
            self.ax2.set_title('Спектр сигналов')
            self.ax2.set_xlabel('Частота (МГц)')
            self.ax2.set_ylabel('Мощность (дБ)')
        else:
            # Отображаем огибающую
            tx_envelope = np.abs(signal.hilbert(tx_signal))
            rx_envelope = np.abs(signal.hilbert(rx_signal))
            
            self.ax2.plot(t * 1e6, tx_envelope, 'r--', label='Передаваемый', alpha=0.5)
            self.ax2.plot(t * 1e6, rx_envelope, 'b-', label='Отраженный')
            self.ax2.set_title('Огибающая сигналов')
            self.ax2.set_xlabel('Время (мкс)')
            self.ax2.set_ylabel('Амплитуда')
        
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Обновляем холст
        self.plot_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarAltimeterGUI(root)
    root.mainloop() 