from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class ForestSimulation:
    def __init__(self, heights=None):
        """
        Инициализация симуляции лесной поверхности
        
        Args:
            heights (list): Список высот для симуляции в метрах
        """
        self.altimeter = RadarAltimeter()
        self.heights = heights if heights is not None else [1000]
        self.forest_conditions = self._create_forest_conditions()
        
    def _create_forest_conditions(self):
        """
        Создание набора параметров для различных состояний леса
        
        Returns:
            list: Список кортежей (название состояния, параметры поверхности)
        """
        return [
            ("Сухой лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.2, temperature=25)),
            ("Влажный лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.8, temperature=20)),
            ("Лес после дождя", SurfaceParameters(SurfaceType.FOREST, roughness=0.9, moisture=0.95, temperature=18)),
            ("Зимний лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.7, moisture=0.3, temperature=-10))
        ]
    
    def run_simulation(self, height=None, roll_angle=0, pitch_angle=0, selected_forest=None):
        """
        Запуск симуляции для всех состояний леса или выбранного типа на заданной высоте, угле крена и тангажа
        
        Args:
            height (float): Высота для симуляции в метрах
            roll_angle (float): Угол крена в градусах
            pitch_angle (float): Угол тангажа в градусах
            selected_forest (SurfaceParameters): Параметры выбранного типа леса
        """
        if height is not None:
            self.heights = [height]
            
        for height in self.heights:
            print(f"\nСимуляция на высоте {height} метров при крене {roll_angle}° и тангаже {pitch_angle}°:")
            self._run_single_height_simulation(height, roll_angle, pitch_angle, selected_forest)
    
    def _run_single_height_simulation(self, height, roll_angle, pitch_angle, selected_forest=None):
        """
        Запуск симуляции для одной высоты, угла крена и тангажа
        
        Args:
            height (float): Высота в метрах
            roll_angle (float): Угол крена в градусах
            pitch_angle (float): Угол тангажа в градусах
            selected_forest (SurfaceParameters): Параметры выбранного типа леса
        """
        # Генерация передаваемого сигнала
        _, tx_signal = self.altimeter.generate_chirp_signal(1e-6, 1e9)
        
        # Создание графика
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Симуляция на высоте {height} метров\nпри крене {roll_angle}° и тангаже {pitch_angle}°', fontsize=16)
        
        if selected_forest is not None:
            # Симуляция только для выбранного типа леса
            for name, params in self.forest_conditions:
                if params == selected_forest:
                    self._plot_simulation(1, name, params, tx_signal, height, roll_angle, pitch_angle)
                    self._print_simulation_info(name, params, height, roll_angle, pitch_angle)
                    break
        else:
            # Моделирование для каждого состояния леса
            for i, (condition, surface_params) in enumerate(self.forest_conditions):
                self._plot_simulation(i+1, condition, surface_params, tx_signal, height, roll_angle, pitch_angle)
                self._print_simulation_info(condition, surface_params, height, roll_angle, pitch_angle)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_simulation(self, subplot_index, condition, surface_params, tx_signal, height, roll_angle, pitch_angle):
        """
        Построение графика для одного состояния леса
        
        Args:
            subplot_index (int): Номер подграфика
            condition (str): Название состояния
            surface_params (SurfaceParameters): Параметры поверхности
            tx_signal (array): Передаваемый сигнал
            height (float): Высота в метрах
            roll_angle (float): Угол крена в градусах
            pitch_angle (float): Угол тангажа в градусах
        """
        t, rx_signal = self.altimeter.simulate_reflection(height, surface_params)
        
        plt.subplot(2, 2, subplot_index)
        plt.plot(t * 1e6, tx_signal, 'r--', label='Передаваемый сигнал', alpha=0.5)
        plt.plot(t * 1e6, rx_signal, 'b-', label='Отраженный сигнал')
        plt.title(f'Отражение от {condition}')
        plt.xlabel('Время (мкс)')
        plt.ylabel('Амплитуда')
        plt.legend()
        plt.grid(True)
    
    def _print_simulation_info(self, condition, surface_params, height, roll_angle, pitch_angle):
        """
        Вывод информации о симуляции
        
        Args:
            condition (str): Название состояния
            surface_params (SurfaceParameters): Параметры поверхности
            height (float): Высота в метрах
            roll_angle (float): Угол крена в градусах
            pitch_angle (float): Угол тангажа в градусах
        """
        # Расчет угла скольжения с учетом крена и тангажа
        base_grazing_angle = np.arctan(1.0 / height)
        roll_rad = np.radians(roll_angle)
        pitch_rad = np.radians(pitch_angle)
        
        # Расчет эффективного угла скольжения
        effective_grazing_angle = np.arccos(
            np.cos(base_grazing_angle) * 
            np.cos(roll_rad) * 
            np.cos(pitch_rad)
        )
        
        reflection_coeff = self.altimeter.calculate_reflection_coefficient(effective_grazing_angle, surface_params)
        
        print(f"{condition}:")
        print(f"  - Шероховатость: {surface_params.roughness}")
        print(f"  - Влажность: {surface_params.moisture}")
        print(f"  - Температура: {surface_params.temperature}°C")
        print(f"  - Базовый угол скольжения: {np.degrees(base_grazing_angle):.2f}°")
        print(f"  - Эффективный угол скольжения: {np.degrees(effective_grazing_angle):.2f}°")
        print(f"  - Коэффициент отражения: {reflection_coeff:.4f}\n")

class ForestAnalysis:
    def __init__(self, simulation):
        """
        Инициализация анализа результатов симуляции
        
        Args:
            simulation (ForestSimulation): Объект симуляции
        """
        self.simulation = simulation
    
    def analyze_reflection_coefficients(self, roll_angle=0, pitch_angle=0):
        """
        Анализ коэффициентов отражения для разных состояний леса на разных высотах
        
        Args:
            roll_angle (float): Угол крена в градусах
            pitch_angle (float): Угол тангажа в градусах
        """
        print(f"\nАнализ коэффициентов отражения при крене {roll_angle}° и тангаже {pitch_angle}°:")
        
        for height in self.simulation.heights:
            print(f"\nВысота {height} метров:")
            coefficients = []
            
            for condition, surface_params in self.simulation.forest_conditions:
                base_grazing_angle = np.arctan(1.0 / height)
                roll_rad = np.radians(roll_angle)
                pitch_rad = np.radians(pitch_angle)
                
                effective_grazing_angle = np.arccos(
                    np.cos(base_grazing_angle) * 
                    np.cos(roll_rad) * 
                    np.cos(pitch_rad)
                )
                
                reflection_coeff = self.simulation.altimeter.calculate_reflection_coefficient(
                    effective_grazing_angle, surface_params
                )
                coefficients.append((condition, reflection_coeff))
            
            # Сортировка по коэффициенту отражения
            coefficients.sort(key=lambda x: x[1], reverse=True)
            
            for condition, coeff in coefficients:
                print(f"{condition}: {coeff:.4f}")
    
    def plot_height_dependence(self, roll_angle=0, pitch_angle=0):
        """
        Построение графика зависимости коэффициента отражения от высоты
        
        Args:
            roll_angle (float): Угол крена в градусах
            pitch_angle (float): Угол тангажа в градусах
        """
        heights = np.linspace(100, 5000, 50)  # Высоты от 100 до 5000 метров
        plt.figure(figsize=(12, 6))
        
        for condition, surface_params in self.simulation.forest_conditions:
            reflection_coeffs = []
            for height in heights:
                base_grazing_angle = np.arctan(1.0 / height)
                roll_rad = np.radians(roll_angle)
                pitch_rad = np.radians(pitch_angle)
                
                effective_grazing_angle = np.arccos(
                    np.cos(base_grazing_angle) * 
                    np.cos(roll_rad) * 
                    np.cos(pitch_rad)
                )
                
                reflection_coeff = self.simulation.altimeter.calculate_reflection_coefficient(
                    effective_grazing_angle, surface_params
                )
                reflection_coeffs.append(reflection_coeff)
            
            plt.plot(heights, reflection_coeffs, label=condition)
        
        plt.title(f'Зависимость коэффициента отражения от высоты\nпри крене {roll_angle}° и тангаже {pitch_angle}°')
        plt.xlabel('Высота (м)')
        plt.ylabel('Коэффициент отражения')
        plt.grid(True)
        plt.legend()
        plt.show()

class ForestSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Симуляция отражения сигнала от лесной поверхности")
        
        # Создание симуляции
        self.simulation = ForestSimulation()
        self.analysis = ForestAnalysis(self.simulation)
        
        # Создание основного фрейма
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создание фрейма для параметров
        params_frame = ttk.LabelFrame(main_frame, text="Параметры симуляции", padding="5")
        params_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Выбор типа леса
        ttk.Label(params_frame, text="Тип леса:").grid(row=0, column=0, padx=5, pady=5)
        self.forest_type_var = tk.StringVar()
        self.forest_types = [name for name, _ in self.simulation.forest_conditions]
        forest_type_combo = ttk.Combobox(params_frame, textvariable=self.forest_type_var, values=self.forest_types)
        forest_type_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        forest_type_combo.current(0)  # Выбор первого типа по умолчанию
        
        # Высота
        ttk.Label(params_frame, text="Высота (м):").grid(row=1, column=0, padx=5, pady=5)
        self.height_var = tk.DoubleVar(value=1000)
        height_scale = ttk.Scale(params_frame, from_=100, to=5000, orient=tk.HORIZONTAL, 
                               variable=self.height_var, command=self.update_height_label)
        height_scale.grid(row=1, column=1, padx=5, pady=5)
        self.height_label = ttk.Label(params_frame, text="1000")
        self.height_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Крен
        ttk.Label(params_frame, text="Крен (°):").grid(row=2, column=0, padx=5, pady=5)
        self.roll_var = tk.DoubleVar(value=35)
        roll_scale = ttk.Scale(params_frame, from_=0, to=45, orient=tk.HORIZONTAL,
                             variable=self.roll_var, command=self.update_roll_label)
        roll_scale.grid(row=2, column=1, padx=5, pady=5)
        self.roll_label = ttk.Label(params_frame, text="35")
        self.roll_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Тангаж
        ttk.Label(params_frame, text="Тангаж (°):").grid(row=3, column=0, padx=5, pady=5)
        self.pitch_var = tk.DoubleVar(value=5)
        pitch_scale = ttk.Scale(params_frame, from_=-20, to=20, orient=tk.HORIZONTAL,
                              variable=self.pitch_var, command=self.update_pitch_label)
        pitch_scale.grid(row=3, column=1, padx=5, pady=5)
        self.pitch_label = ttk.Label(params_frame, text="5")
        self.pitch_label.grid(row=3, column=2, padx=5, pady=5)
        
        # Кнопка запуска симуляции
        ttk.Button(params_frame, text="Запустить симуляцию", command=self.run_simulation).grid(
            row=4, column=0, columnspan=3, pady=10)
        
        # Фрейм для результатов
        results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="5")
        results_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Текстовое поле для вывода результатов
        self.results_text = tk.Text(results_frame, height=10, width=50)
        self.results_text.grid(row=0, column=0, padx=5, pady=5)
        
        # Фрейм для графиков
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка расширения
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def update_height_label(self, *args):
        self.height_label.config(text=f"{self.height_var.get():.0f}")
    
    def update_roll_label(self, *args):
        self.roll_label.config(text=f"{self.roll_var.get():.1f}")
    
    def update_pitch_label(self, *args):
        self.pitch_label.config(text=f"{self.pitch_var.get():.1f}")
    
    def run_simulation(self):
        # Очистка предыдущих результатов
        self.results_text.delete(1.0, tk.END)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Запуск симуляции в отдельном потоке
        thread = threading.Thread(target=self._run_simulation_thread)
        thread.daemon = True
        thread.start()
    
    def _run_simulation_thread(self):
        height = self.height_var.get()
        roll = self.roll_var.get()
        pitch = self.pitch_var.get()
        forest_type = self.forest_type_var.get()
        
        # Находим параметры выбранного типа леса
        selected_params = None
        for name, params in self.simulation.forest_conditions:
            if name == forest_type:
                selected_params = params
                break
        
        if selected_params is None:
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Ошибка: не выбран тип леса\n"))
            return
        
        # Запуск симуляции только для выбранного типа леса
        self.simulation.run_simulation(height=height, roll_angle=roll, pitch_angle=pitch, 
                                     selected_forest=selected_params)
        
        # Анализ результатов
        self.analysis.analyze_reflection_coefficients(roll_angle=roll, pitch_angle=pitch)
        
        # Вывод результатов в текстовое поле
        self.root.after(0, self._update_results_text)
        
        # Построение графика зависимости от высоты
        self.root.after(0, self._plot_height_dependence, roll, pitch)
    
    def _update_results_text(self):
        forest_type = self.forest_type_var.get()
        self.results_text.insert(tk.END, f"Тип леса: {forest_type}\n")
        self.results_text.insert(tk.END, f"Высота: {self.height_var.get():.0f} м\n")
        self.results_text.insert(tk.END, f"Крен: {self.roll_var.get():.1f}°\n")
        self.results_text.insert(tk.END, f"Тангаж: {self.pitch_var.get():.1f}°\n")
        
        # Добавляем информацию о параметрах выбранного типа леса
        for name, params in self.simulation.forest_conditions:
            if name == forest_type:
                self.results_text.insert(tk.END, f"\nПараметры леса:\n")
                self.results_text.insert(tk.END, f"Шероховатость: {params.roughness}\n")
                self.results_text.insert(tk.END, f"Влажность: {params.moisture}\n")
                self.results_text.insert(tk.END, f"Температура: {params.temperature}°C\n")
                break
    
    def _plot_height_dependence(self, roll, pitch):
        fig = plt.figure(figsize=(8, 4))
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Построение графика
        self.analysis.plot_height_dependence(roll_angle=roll, pitch_angle=pitch)
        canvas.draw()

def main():
    # Создание GUI
    root = tk.Tk()
    app = ForestSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 