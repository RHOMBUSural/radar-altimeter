import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType, ModulationType, CombinedSurface
import json
import os
from pathlib import Path

class RadarAltimeterGUI:
    def draw_aircraft(self, ax, x, y, z, roll, pitch, scale=0.5):
        """Отрисовка модели летательного аппарата"""
        # Основные размеры самолета
        length = 2.5 * scale
        width = 1 * scale
        height = 0.5 * scale
        
        # Создаем точки для фюзеляжа
        fuselage_x = np.array([-length/2, length/2])
        fuselage_y = np.array([0, 0])
        fuselage_z = np.array([0, 0])
        
        # Создаем точки для крыльев
        wing_x = np.array([0, 0])
        wing_y = np.array([-width, width])
        wing_z = np.array([0, 0])
        
        # Создаем точки для хвоста
        tail_x = np.array([length/2, length/2])
        tail_y = np.array([0, 0])
        tail_z = np.array([0, height])
        
        # Применяем повороты
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        
        # Матрица поворота для крена
        roll_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        # Матрица поворота для тангажа
        pitch_matrix = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        # Поворачиваем все части самолета
        for points in [(fuselage_x, fuselage_y, fuselage_z),
                     (wing_x, wing_y, wing_z),
                     (tail_x, tail_y, tail_z)]:
            points_array = np.vstack(points).T
            points_array = points_array @ roll_matrix @ pitch_matrix
            points[0][:] = points_array[:, 0] + x
            points[1][:] = points_array[:, 1] + y
            points[2][:] = points_array[:, 2] + z
        
        # Рисуем части самолета
        ax.plot(fuselage_x, fuselage_y, fuselage_z, 'gray', linewidth=2)
        ax.plot(wing_x, wing_y, wing_z, 'gray', linewidth=2)
        ax.plot(tail_x, tail_y, tail_z, 'gray', linewidth=2)

    def __init__(self, root):
        self.root = root
        self.root.title("Радиовысотомер")
        
        # Определяем путь к файлу настроек в директории пользователя
        self.settings_file = os.path.join(str(Path.home()), 'radar_altimeter_settings.json')
        
        # Создаем экземпляр радиовысотомера
        self.altimeter = RadarAltimeter()
        
        # Инициализируем переменные по умолчанию
        self.f0_var = tk.DoubleVar(value=self.altimeter.f0 / 1e9)
        self.bandwidth_var = tk.DoubleVar(value=self.altimeter.B / 1e6)
        self.modulation_var = tk.StringVar(value=ModulationType.LFM.value)
        self.pulse_width_var = tk.DoubleVar(value=self.altimeter.pulse_width * 1e6)
        self.pulse_period_var = tk.DoubleVar(value=self.altimeter.pulse_period * 1e3)
        self.gain_var = tk.DoubleVar(value=self.altimeter.antenna_gain)
        self.power_var = tk.DoubleVar(value=self.altimeter.tx_power)
        self.roll_var = tk.DoubleVar(value=self.altimeter.roll)
        self.pitch_var = tk.DoubleVar(value=self.altimeter.pitch)
        self.height_var = tk.DoubleVar(value=10)
        self.surface_var = tk.StringVar(value=SurfaceType.LAND.value)
        
        # Флаг для отображения спектра
        self.show_spectrum = False
        
        # Флаг для отображения отраженного сигнала
        self.show_signal_var = tk.BooleanVar(value=True)
        
        # Создаем основной фрейм
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем фрейм для управления параметрами
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Параметры", padding="5")
        self.control_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Создаем фрейм для комбинированного ландшафта
        self.landscape_frame = ttk.LabelFrame(self.control_frame, text="Комбинированный ландшафт", padding="5")
        self.landscape_frame.grid(row=12, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Инициализируем элементы управления комбинированным ландшафтом
        self.area1_type = ttk.Combobox(self.landscape_frame, values=[st.value for st in SurfaceType if st != SurfaceType.COMBINED])
        self.area1_x1 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_x2 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_y1 = ttk.Entry(self.landscape_frame, width=5)
        self.area1_y2 = ttk.Entry(self.landscape_frame, width=5)
        
        self.area2_type = ttk.Combobox(self.landscape_frame, values=[st.value for st in SurfaceType if st != SurfaceType.COMBINED])
        self.area2_x1 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_x2 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_y1 = ttk.Entry(self.landscape_frame, width=5)
        self.area2_y2 = ttk.Entry(self.landscape_frame, width=5)
        
        # Загружаем сохраненные настройки
        self.load_settings()
        
        # Создаем остальные элементы интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Центральная частота
        ttk.Label(self.control_frame, text="Центральная частота (ГГц):").grid(row=0, column=0, padx=5, pady=5)
        self.f0_entry = ttk.Entry(self.control_frame, textvariable=self.f0_var)
        self.f0_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Полоса частот
        ttk.Label(self.control_frame, text="Полоса частот (МГц):").grid(row=1, column=0, padx=5, pady=5)
        self.bandwidth_entry = ttk.Entry(self.control_frame, textvariable=self.bandwidth_var)
        self.bandwidth_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Тип модуляции
        ttk.Label(self.control_frame, text="Тип модуляции:").grid(row=2, column=0, padx=5, pady=5)
        self.modulation_combo = ttk.Combobox(self.control_frame, textvariable=self.modulation_var)
        self.modulation_combo['values'] = [mt.value for mt in ModulationType]
        self.modulation_combo.grid(row=2, column=1, padx=5, pady=5)
        self.modulation_combo.bind('<<ComboboxSelected>>', lambda e: self.update_modulation_controls())
        
        # Параметры импульсной модуляции
        self.pulse_frame = ttk.LabelFrame(self.control_frame, text="Параметры ИМ", padding="5")
        self.pulse_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Длительность импульса
        ttk.Label(self.pulse_frame, text="Длительность импульса (мкс):").grid(row=0, column=0, padx=5, pady=2)
        self.pulse_width_entry = ttk.Entry(self.pulse_frame, textvariable=self.pulse_width_var)
        self.pulse_width_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Период повторения
        ttk.Label(self.pulse_frame, text="Период повторения (мс):").grid(row=1, column=0, padx=5, pady=2)
        self.pulse_period_entry = ttk.Entry(self.pulse_frame, textvariable=self.pulse_period_var)
        self.pulse_period_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Коэффициент усиления антенны
        ttk.Label(self.control_frame, text="Усиление антенны (дБ):").grid(row=4, column=0, padx=5, pady=5)
        self.gain_entry = ttk.Entry(self.control_frame, textvariable=self.gain_var)
        self.gain_entry.grid(row=4, column=1, padx=5, pady=5)
        
        # Мощность передатчика
        ttk.Label(self.control_frame, text="Мощность передатчика (Вт):").grid(row=5, column=0, padx=5, pady=5)
        self.power_entry = ttk.Entry(self.control_frame, textvariable=self.power_var)
        self.power_entry.grid(row=5, column=1, padx=5, pady=5)
        
        # Крен
        ttk.Label(self.control_frame, text="Крен (градусы):").grid(row=6, column=0, padx=5, pady=5)
        self.roll_scale = ttk.Scale(self.control_frame, from_=-30, to=30, variable=self.roll_var, 
                                  orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
        self.roll_scale.grid(row=6, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.roll_label = ttk.Label(self.control_frame, text=f"{self.roll_var.get():.1f}°")
        self.roll_label.grid(row=6, column=2, padx=5, pady=5)
        
        # Тангаж
        ttk.Label(self.control_frame, text="Тангаж (градусы):").grid(row=7, column=0, padx=5, pady=5)
        self.pitch_scale = ttk.Scale(self.control_frame, from_=-30, to=30, variable=self.pitch_var,
                                   orient=tk.HORIZONTAL, command=lambda x: self.update_plots())
        self.pitch_scale.grid(row=7, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.pitch_label = ttk.Label(self.control_frame, text=f"{self.pitch_var.get():.1f}°")
        self.pitch_label.grid(row=7, column=2, padx=5, pady=5)
        
        # Высота
        ttk.Label(self.control_frame, text="Высота (м):").grid(row=8, column=0, padx=5, pady=5)
        self.height_entry = ttk.Entry(self.control_frame, textvariable=self.height_var)
        self.height_entry.grid(row=8, column=1, padx=5, pady=5)
        
        # Тип поверхности
        ttk.Label(self.control_frame, text="Тип поверхности:").grid(row=9, column=0, padx=5, pady=5)
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
        
        # Параметры первого участка
        ttk.Label(self.landscape_frame, text="Участок 1:").grid(row=0, column=0, padx=5, pady=2)
        self.area1_type.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="X1:").grid(row=0, column=2, padx=5, pady=2)
        self.area1_x1.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="X2:").grid(row=0, column=4, padx=5, pady=2)
        self.area1_x2.grid(row=0, column=5, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="Y1:").grid(row=0, column=6, padx=5, pady=2)
        self.area1_y1.grid(row=0, column=7, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="Y2:").grid(row=0, column=8, padx=5, pady=2)
        self.area1_y2.grid(row=0, column=9, padx=5, pady=2)
        
        # Параметры второго участка
        ttk.Label(self.landscape_frame, text="Участок 2:").grid(row=1, column=0, padx=5, pady=2)
        self.area2_type.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="X1:").grid(row=1, column=2, padx=5, pady=2)
        self.area2_x1.grid(row=1, column=3, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="X2:").grid(row=1, column=4, padx=5, pady=2)
        self.area2_x2.grid(row=1, column=5, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="Y1:").grid(row=1, column=6, padx=5, pady=2)
        self.area2_y1.grid(row=1, column=7, padx=5, pady=2)
        
        ttk.Label(self.landscape_frame, text="Y2:").grid(row=1, column=8, padx=5, pady=2)
        self.area2_y2.grid(row=1, column=9, padx=5, pady=2)
        
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
        
        # Создаем фрейм для управления
        control_frame = ttk.Frame(landscape_window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Добавляем чекбокс для отображения отраженного сигнала
        show_signal_var = tk.BooleanVar(value=False)
        show_signal_check = ttk.Checkbutton(control_frame, text="Показать отраженный сигнал", 
                                          variable=show_signal_var,
                                          command=lambda: self.update_3d_plot(fig, ax, X, Y, Z, colors, 
                                                                             signal_strength, show_signal_var.get(),
                                                                             show_scan_area_var.get()))
        show_signal_check.pack(side=tk.LEFT, padx=5)
        
        # Добавляем чекбокс для отображения области зондирования
        show_scan_area_var = tk.BooleanVar(value=False)
        show_scan_area_check = ttk.Checkbutton(control_frame, text="Показать область зондирования", 
                                             variable=show_scan_area_var,
                                             command=lambda: self.update_3d_plot(fig, ax, X, Y, Z, colors, 
                                                                                signal_strength, show_signal_var.get(),
                                                                                show_scan_area_var.get()))
        show_scan_area_check.pack(side=tk.LEFT, padx=5)
        
        # Создаем фигуру для 3D графика
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Создаем сетку для поверхности с большим разрешением
        resolution = 100
        x = np.linspace(0, 100, resolution)
        y = np.linspace(0, 100, resolution)
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
        
        # Создаем Z-координаты и цвета
        Z = np.zeros_like(X)
        colors = np.zeros_like(X, dtype=object)
        
        # Базовые высоты для разных типов поверхности
        base_heights = {
            SurfaceType.SEA: 0,
            SurfaceType.ICE: 5,
            SurfaceType.LAND: 10,
            SurfaceType.DESERT: 15,
            SurfaceType.FOREST: 20,
            SurfaceType.URBAN: 30
        }
        
        # Цвета для разных типов поверхности
        color_map = {
            SurfaceType.SEA: 'blue',
            SurfaceType.LAND: 'brown',
            SurfaceType.FOREST: 'darkgreen',
            SurfaceType.URBAN: 'gray',
            SurfaceType.ICE: 'lightblue',
            SurfaceType.DESERT: 'yellow'
        }
        
        # Функция для генерации шума Перлина
        def perlin_noise(x, y, scale=0.1, octaves=6, persistence=0.5, lacunarity=2.0):
            noise = np.zeros_like(x)
            amplitude = 1.0
            frequency = 1.0
            for _ in range(octaves):
                noise += amplitude * np.sin(2 * np.pi * frequency * (x * scale + y * scale))
                amplitude *= persistence
                frequency *= lacunarity
            return noise
        
        # Функция для расчета градиентов высот
        def calculate_gradients(Z, dx, dy):
            # Используем центральные разности для расчета градиентов
            grad_x = np.gradient(Z, dx, axis=1)
            grad_y = np.gradient(Z, dy, axis=0)
            return grad_x, grad_y
        
        # Функция для расчета локальных углов падения
        def calculate_local_angles(grad_x, grad_y):
            # Вычисляем угол наклона поверхности
            slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
            # Вычисляем азимут наклона
            aspect = np.arctan2(grad_y, grad_x)
            return slope, aspect
        
        # Функция для расчета многолучевого распространения
        def calculate_multipath(X, Y, Z, height, max_reflections=2):
            # Создаем массив для хранения силы сигнала с учетом многолучевости
            signal_strength = np.zeros_like(X)
            
            # Создаем комбинированную поверхность
            combined_surface = CombinedSurface([
                (SurfaceType(self.area1_type.get()), 
                 float(self.area1_x1.get()), float(self.area1_x2.get()),
                 float(self.area1_y1.get()), float(self.area1_y2.get())),
                (SurfaceType(self.area2_type.get()),
                 float(self.area2_x1.get()), float(self.area2_x2.get()),
                 float(self.area2_y1.get()), float(self.area2_y2.get()))
            ])
            
            # Для каждой точки поверхности
            for i in range(len(x)):
                for j in range(len(y)):
                    # Прямой путь
                    dx = X[i,j] - 50  # Центр поверхности
                    dy = Y[i,j] - 50
                    dz = Z[i,j] - height
                    direct_path = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Рассчитываем силу сигнала для прямого пути
                    surface_type = combined_surface.get_surface_type(X[i,j], Y[i,j])
                    surface_params = SurfaceParameters(surface_type)
                    grazing_angle = np.arctan2(height - Z[i,j], np.sqrt(dx**2 + dy**2))
                    reflection_coeff = self.altimeter.calculate_reflection_coefficient(grazing_angle, surface_params)
                    direct_signal = reflection_coeff / (direct_path**2)
                    
                    # Добавляем вторичные отражения
                    total_signal = direct_signal
                    for reflection in range(max_reflections):
                        # Ищем ближайшие точки для вторичного отражения
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                
                                ni, nj = i + di, j + dj
                                if 0 <= ni < len(x) and 0 <= nj < len(y):
                                    # Рассчитываем путь через вторичное отражение
                                    dx1 = X[ni,nj] - 50
                                    dy1 = Y[ni,nj] - 50
                                    dz1 = Z[ni,nj] - height
                                    path1 = np.sqrt(dx1**2 + dy1**2 + dz1**2)
                                    
                                    dx2 = X[i,j] - X[ni,nj]
                                    dy2 = Y[i,j] - Y[ni,nj]
                                    dz2 = Z[i,j] - Z[ni,nj]
                                    path2 = np.sqrt(dx2**2 + dy2**2 + dz2**2)
                                    
                                    # Рассчитываем силу сигнала для вторичного отражения
                                    surface_type1 = combined_surface.get_surface_type(X[ni,nj], Y[ni,nj])
                                    surface_params1 = SurfaceParameters(surface_type1)
                                    grazing_angle1 = np.arctan2(height - Z[ni,nj], np.sqrt(dx1**2 + dy1**2))
                                    reflection_coeff1 = self.altimeter.calculate_reflection_coefficient(grazing_angle1, surface_params1)
                                    
                                    secondary_signal = reflection_coeff1 / (path1 * path2)
                                    total_signal += secondary_signal * 0.1  # Ослабление вторичного сигнала
                    
                    signal_strength[i,j] = total_signal
            
            return signal_strength
        
        # Генерируем базовый шум для всей поверхности
        base_noise = perlin_noise(X, Y, scale=0.05)
        
        # Создаем массив для отраженного сигнала
        signal_strength = np.zeros_like(X)
        
        # Генерируем ландшафт
        for i in range(len(x)):
            for j in range(len(y)):
                surface_type = combined_surface.get_surface_type(x[i], y[j])
                base_height = base_heights[surface_type]
                
                # Добавляем детализацию в зависимости от типа поверхности
                if surface_type == SurfaceType.SEA:
                    # Волны на море
                    wave_height = 2 * np.sin(0.2 * x[i]) * np.cos(0.2 * y[j])
                    Z[i,j] = base_height + wave_height
                elif surface_type == SurfaceType.FOREST:
                    # Неровности леса
                    forest_noise = perlin_noise(x[i], y[j], scale=0.2) * 5
                    Z[i,j] = base_height + forest_noise
                elif surface_type == SurfaceType.URBAN:
                    # Здания и сооружения
                    building_height = np.random.normal(5, 2) if np.random.random() < 0.3 else 0
                    Z[i,j] = base_height + building_height
                elif surface_type == SurfaceType.DESERT:
                    # Дюны
                    dune_height = 3 * np.sin(0.1 * x[i]) * np.cos(0.1 * y[j])
                    Z[i,j] = base_height + dune_height
                elif surface_type == SurfaceType.ICE:
                    # Трещины и неровности льда
                    ice_noise = perlin_noise(x[i], y[j], scale=0.15) * 3
                    Z[i,j] = base_height + ice_noise
                else:
                    # Обычный ландшафт
                    Z[i,j] = base_height + base_noise[i,j] * 5
                
                # Добавляем общий шум для реалистичности
                Z[i,j] += np.random.normal(0, 0.5)
                
                # Устанавливаем цвет
                colors[i,j] = color_map[surface_type]
        
        # Сглаживаем высоты для плавных переходов
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=1.0)
        
        # Рассчитываем градиенты высот
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        grad_x, grad_y = calculate_gradients(Z, dx, dy)
        
        # Рассчитываем локальные углы падения
        slope, aspect = calculate_local_angles(grad_x, grad_y)
        
        # Рассчитываем многолучевое распространение
        signal_strength = calculate_multipath(X, Y, Z, self.height_var.get())
        
        # Нормализуем силу сигнала
        signal_strength = signal_strength / np.max(signal_strength)
        
        # Отображаем поверхность с цветами
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)
        
        # Отображаем силу отраженного сигнала, если включено
        if show_signal_var.get():
            signal_surf = ax.plot_surface(X, Y, Z + 2, facecolors=plt.cm.viridis(signal_strength), alpha=0.6)
        
        # Добавляем легенду
        legend_elements = []
        for surface_type, color in color_map.items():
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=f"{surface_type.value} ({base_heights[surface_type]}м)"))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Настраиваем оси
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Высота (м)')
        
        # Устанавливаем начальный угол обзора
        ax.view_init(elev=30, azim=45)
        
        # Добавляем сетку
        ax.grid(True)
        
        # Добавляем подпись с текущими параметрами
        ax.text2D(0.02, 0.95, 
                 f"Крен: {self.roll_var.get():.1f}°\n"
                 f"Тангаж: {self.pitch_var.get():.1f}°\n"
                 f"Высота: {self.height_var.get()}м\n"
                 f"Разрешение: {resolution}x{resolution}",
                 transform=ax.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
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

    def update_3d_plot(self, fig, ax, X, Y, Z, colors, signal_strength, show_signal, show_scan_area):
        # Очищаем текущий график
        ax.clear()
        
        # Отображаем поверхность с цветами
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8, antialiased=True)
        
        # Отображаем силу отраженного сигнала, если включено
        if show_signal:
            signal_surf = ax.plot_surface(X, Y, Z + 2, facecolors=plt.cm.viridis(signal_strength), alpha=0.6, antialiased=True)
        
        # Отображаем область зондирования, если включено
        if show_scan_area:
            # Рассчитываем область зондирования на основе высоты и углов
            height = self.height_var.get()
            roll = np.radians(self.roll_var.get())
            pitch = np.radians(self.pitch_var.get())
            
            # Параметры области зондирования
            beam_width = np.radians(30)  # Ширина луча в градусах
            max_range = height * 2  # Максимальная дальность
            
            # Создаем точки для конуса зондирования
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.linspace(0, max_range, 100)
            T, R = np.meshgrid(theta, r)
            
            # Преобразуем в декартовы координаты
            X_cone = R * np.cos(T) * np.sin(beam_width/2)
            Y_cone = R * np.sin(T) * np.sin(beam_width/2)
            Z_cone = R * np.cos(beam_width/2)
            
            # Применяем повороты
            def rotate_points(x, y, z, roll, pitch):
                # Поворот по крену (вокруг оси X)
                y_roll = y * np.cos(roll) - z * np.sin(roll)
                z_roll = y * np.sin(roll) + z * np.cos(roll)
                
                # Поворот по тангажу (вокруг оси Y)
                x_pitch = x * np.cos(pitch) + z_roll * np.sin(pitch)
                z_pitch = -x * np.sin(pitch) + z_roll * np.cos(pitch)
                
                return x_pitch, y_roll, z_pitch
            
            X_cone, Y_cone, Z_cone = rotate_points(X_cone, Y_cone, Z_cone, roll, pitch)
            
            # Смещаем конус в позицию ЛА
            X_cone += 50  # Центр поверхности по X
            Y_cone += 50  # Центр поверхности по Y
            Z_cone += height  # Высота ЛА
            
            # Отображаем конус зондирования
            scan_area = ax.plot_surface(X_cone, Y_cone, Z_cone, color='red', alpha=0.2, antialiased=True)
        
        # Рисуем летательный аппарат
        self.draw_aircraft(ax, 50, 50, self.height_var.get(), self.roll_var.get(), self.pitch_var.get())
        
        # Настраиваем оси
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Высота (м)')
        
        # Устанавливаем начальный угол обзора
        ax.view_init(elev=30, azim=45)
        
        # Добавляем сетку
        ax.grid(True)
        
        # Добавляем подпись с текущими параметрами
        ax.text2D(0.02, 0.95, 
                 f"Крен: {self.roll_var.get():.1f}°\n"
                 f"Тангаж: {self.pitch_var.get():.1f}°\n"
                 f"Высота: {self.height_var.get()}м\n"
                 f"Разрешение: {len(X)}x{len(Y)}",
                 transform=ax.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Добавляем легенду
        legend_elements = []
        for surface_type, color in color_map.items():
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=f"{surface_type.value} ({base_heights[surface_type]}м)"))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Обновляем холст
        fig.canvas.draw()

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
        
        # Нормализуем сигналы, избегая деления на ноль
        max_tx = np.max(np.abs(tx_signal))
        max_rx = np.max(np.abs(rx_signal))
        
        if max_tx > 0:
            tx_signal = tx_signal / max_tx
        if max_rx > 0:
            rx_signal = rx_signal / max_rx
        
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

    def save_settings(self):
        """Сохранение настроек в JSON файл"""
        try:
            settings = {
                'f0': self.f0_var.get(),
                'bandwidth': self.bandwidth_var.get(),
                'modulation': self.modulation_var.get(),
                'pulse_width': self.pulse_width_var.get(),
                'pulse_period': self.pulse_period_var.get(),
                'gain': self.gain_var.get(),
                'power': self.power_var.get(),
                'roll': self.roll_var.get(),
                'pitch': self.pitch_var.get(),
                'height': self.height_var.get(),
                'surface': self.surface_var.get(),
                'area1_type': self.area1_type.get(),
                'area1_x1': self.area1_x1.get(),
                'area1_x2': self.area1_x2.get(),
                'area1_y1': self.area1_y1.get(),
                'area1_y2': self.area1_y2.get(),
                'area2_type': self.area2_type.get(),
                'area2_x1': self.area2_x1.get(),
                'area2_x2': self.area2_x2.get(),
                'area2_y1': self.area2_y1.get(),
                'area2_y2': self.area2_y2.get()
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Ошибка при сохранении настроек: {e}")

    def load_settings(self):
        """Загрузка настроек из JSON файла"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    
                # Обновляем значения переменных
                self.f0_var.set(settings.get('f0', self.f0_var.get()))
                self.bandwidth_var.set(settings.get('bandwidth', self.bandwidth_var.get()))
                self.modulation_var.set(settings.get('modulation', self.modulation_var.get()))
                self.pulse_width_var.set(settings.get('pulse_width', self.pulse_width_var.get()))
                self.pulse_period_var.set(settings.get('pulse_period', self.pulse_period_var.get()))
                self.gain_var.set(settings.get('gain', self.gain_var.get()))
                self.power_var.set(settings.get('power', self.power_var.get()))
                self.roll_var.set(settings.get('roll', self.roll_var.get()))
                self.pitch_var.set(settings.get('pitch', self.pitch_var.get()))
                self.height_var.set(settings.get('height', self.height_var.get()))
                self.surface_var.set(settings.get('surface', self.surface_var.get()))
                
                # Обновляем настройки комбинированного ландшафта
                self.area1_type.set(settings.get('area1_type', SurfaceType.LAND.value))
                self.area1_x1.delete(0, tk.END)
                self.area1_x1.insert(0, settings.get('area1_x1', "0"))
                self.area1_x2.delete(0, tk.END)
                self.area1_x2.insert(0, settings.get('area1_x2', "50"))
                self.area1_y1.delete(0, tk.END)
                self.area1_y1.insert(0, settings.get('area1_y1', "0"))
                self.area1_y2.delete(0, tk.END)
                self.area1_y2.insert(0, settings.get('area1_y2', "50"))
                
                self.area2_type.set(settings.get('area2_type', SurfaceType.SEA.value))
                self.area2_x1.delete(0, tk.END)
                self.area2_x1.insert(0, settings.get('area2_x1', "50"))
                self.area2_x2.delete(0, tk.END)
                self.area2_x2.insert(0, settings.get('area2_x2', "100"))
                self.area2_y1.delete(0, tk.END)
                self.area2_y1.insert(0, settings.get('area2_y1', "0"))
                self.area2_y2.delete(0, tk.END)
                self.area2_y2.insert(0, settings.get('area2_y2', "50"))
        except Exception as e:
            print(f"Ошибка при загрузке настроек: {e}")
            # Если файл не найден или поврежден, оставляем значения по умолчанию
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarAltimeterGUI(root)
    
    # Добавляем обработчик закрытия окна
    def on_closing():
        app.save_settings()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 