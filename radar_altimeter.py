import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from enum import Enum
import torch
from typing import List, Tuple

class ModulationType(Enum):
    LFM = "ЛЧМ"  # Линейная частотная модуляция
    CW = "НМ"    # Непрерывная волна
    PULSE = "ИМ" # Импульсная модуляция

class SurfaceType(Enum):
    LAND = "суша"
    SEA = "море"
    ICE = "лёд"
    FOREST = "лес"
    URBAN = "город"
    DESERT = "пустыня"
    COMBINED = "комбинированный"

class PolarizationType(Enum):
    VERTICAL = "Вертикальная"
    HORIZONTAL = "Горизонтальная"

class CombinedSurface:
    def __init__(self, surfaces):
        """
        Инициализация комбинированной поверхности
        
        Args:
            surfaces (list): Список кортежей (surface_type, x_start, x_end, y_start, y_end)
        """
        self.surfaces = surfaces
        
    def get_surface_type(self, x, y):
        """
        Получение типа поверхности для заданных координат
        
        Args:
            x (float): X-координата
            y (float): Y-координата
            
        Returns:
            SurfaceType: Тип поверхности в заданной точке
        """
        for surface_type, x1, x2, y1, y2 in self.surfaces:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return surface_type
        return SurfaceType.LAND  # По умолчанию возвращаем сушу

class SurfaceGradient:
    def __init__(self, height_map: np.ndarray, resolution: float = 1.0):
        """
        Инициализация расчета градиентов поверхности
        
        Args:
            height_map (np.ndarray): Карта высот поверхности
            resolution (float): Разрешение карты (метры на пиксель)
        """
        self.height_map = height_map
        self.resolution = resolution
        self.grad_x = None
        self.grad_y = None
        self._calculate_gradients()
    
    def _calculate_gradients(self):
        """Расчет градиентов высоты по осям X и Y"""
        self.grad_x = np.gradient(self.height_map, axis=1) / self.resolution
        self.grad_y = np.gradient(self.height_map, axis=0) / self.resolution
    
    def get_local_slope(self, x: int, y: int) -> Tuple[float, float]:
        """
        Получение локального наклона поверхности в точке (x,y)
        
        Returns:
            Tuple[float, float]: Углы наклона по осям X и Y в радианах
        """
        if x < 0 or x >= self.height_map.shape[1] or y < 0 or y >= self.height_map.shape[0]:
            return 0.0, 0.0
        return np.arctan(self.grad_x[y, x]), np.arctan(self.grad_y[y, x])
    
    def get_effective_grazing_angle(self, x: int, y: int, base_angle: float) -> float:
        """
        Расчет эффективного угла скольжения с учетом локального наклона
        
        Args:
            x (int): X-координата
            y (int): Y-координата
            base_angle (float): Базовый угол скольжения
            
        Returns:
            float: Эффективный угол скольжения
        """
        slope_x, slope_y = self.get_local_slope(x, y)
        return base_angle + np.arctan(np.sqrt(np.tan(slope_x)**2 + np.tan(slope_y)**2))

class SurfaceParameters:
    def __init__(self, surface_type: SurfaceType, roughness: float = 0.1, 
                 moisture: float = 0.0, temperature: float = 20.0,
                 dielectric_constant: float = None):
        self.surface_type = surface_type
        self.roughness = roughness
        self.moisture = moisture
        self.temperature = temperature
        self.dielectric_constant = dielectric_constant or self._get_default_dielectric_constant()
    
    def _get_default_dielectric_constant(self) -> float:
        """Получение диэлектрической проницаемости по умолчанию для типа поверхности"""
        constants = {
            SurfaceType.SEA: 80.0,
            SurfaceType.LAND: 15.0,
            SurfaceType.FOREST: 12.0,
            SurfaceType.URBAN: 5.0,
            SurfaceType.ICE: 3.2,
            SurfaceType.DESERT: 3.0
        }
        return constants.get(self.surface_type, 15.0)

class RadarAltimeter:
    def __init__(self, use_gpu: bool = False):
        self.c = 3e8  # Скорость света, м/с
        self.f0 = 4.3e9  # Центральная частота, Гц
        self.B = 160e6  # Полоса частот, Гц
        self.T = 1e-6  # Длительность сигнала, с
        self.roll = 0.0  # Крен, градусы
        self.pitch = 0.0  # Тангаж, градусы
        self.modulation_type = ModulationType.LFM  # Тип модуляции
        self.pulse_width = 1e-6  # Длительность импульса для ИМ, с
        self.pulse_period = 1e-3    # Период повторения импульсов для ИМ, с
        self.tx_power = 0.1  # Мощность передатчика, Вт
        self.antenna_gain = 10  # Усиление антенны, дБ
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
    def calculate_reflection_coefficient(self, grazing_angle, surface_params, polarization: PolarizationType = PolarizationType.VERTICAL):
        """
        Расчет коэффициента отражения с учетом поляризации
        
        Args:
            grazing_angle (float): Угол скольжения, рад
            surface_params (SurfaceParameters): Параметры поверхности
            polarization (PolarizationType): Тип поляризации
            
        Returns:
            float: Коэффициент отражения
        """
        # Учитываем крен и тангаж при расчете эффективного угла скольжения
        effective_angle = grazing_angle + np.radians(self.roll) * np.cos(np.radians(self.pitch))
        
        # Расчет коэффициентов Френеля
        eps = surface_params.dielectric_constant
        sin_theta = np.sin(effective_angle)
        cos_theta = np.cos(effective_angle)
        
        if polarization == PolarizationType.VERTICAL:
            # Вертикальная поляризация
            R = (eps * cos_theta - np.sqrt(eps - sin_theta**2)) / \
                (eps * cos_theta + np.sqrt(eps - sin_theta**2))
        else:
            # Горизонтальная поляризация
            R = (cos_theta - np.sqrt(eps - sin_theta**2)) / \
                (cos_theta + np.sqrt(eps - sin_theta**2))
        
        # Учет шероховатости поверхности
        roughness_factor = np.exp(-2 * (2 * np.pi * self.f0 / self.c * surface_params.roughness * sin_theta)**2)
        
        # Учет влажности
        moisture_factor = 1 - 0.5 * surface_params.moisture * (1 - np.cos(effective_angle))
        
        # Учет температуры
        temp_factor = 1 - 0.1 * abs(surface_params.temperature) / 30
        
        return np.abs(R) * roughness_factor * moisture_factor * temp_factor

    def generate_signal(self, T, B):
        """
        Генерация сигнала в зависимости от выбранного типа модуляции
        
        Args:
            T (float): Длительность сигнала, с
            B (float): Полоса частот, Гц
            
        Returns:
            tuple: (массив времени, массив сигнала)
        """
        if self.modulation_type == ModulationType.LFM:
            return self.generate_chirp_signal(T, B)
        elif self.modulation_type == ModulationType.CW:
            return self.generate_cw_signal(T)
        elif self.modulation_type == ModulationType.PULSE:
            return self.generate_pulse_signal(T)
        else:
            return self.generate_chirp_signal(T, B)

    def generate_chirp_signal(self, T, B):
        """
        Генерация ЛЧМ сигнала
        
        Args:
            T (float): Длительность сигнала, с
            B (float): Полоса частот, Гц
            
        Returns:
            tuple: (массив времени, массив сигнала)
        """
        t = np.linspace(0, T, int(T * 1e9))
        k = B / T  # Скорость изменения частоты
        signal = np.cos(2 * np.pi * (self.f0 * t + 0.5 * k * t**2))
        return t, signal

    def generate_cw_signal(self, T):
        """
        Генерация сигнала с непрерывной волной
        
        Args:
            T (float): Длительность сигнала, с
            
        Returns:
            tuple: (массив времени, массив сигнала)
        """
        t = np.linspace(0, T, int(T * 1e9))
        signal = np.cos(2 * np.pi * self.f0 * t)
        return t, signal

    def generate_pulse_signal(self, T):
        """
        Генерация импульсного сигнала
        
        Args:
            T (float): Длительность сигнала, с
            
        Returns:
            tuple: (массив времени, массив сигнала)
        """
        t = np.linspace(0, T, int(T * 1e9))
        signal = np.zeros_like(t)
        
        # Создаем последовательность импульсов
        num_pulses = int(T / self.pulse_period)
        for i in range(num_pulses):
            pulse_start = i * self.pulse_period
            pulse_end = pulse_start + self.pulse_width
            mask = (t >= pulse_start) & (t < pulse_end)
            signal[mask] = np.cos(2 * np.pi * self.f0 * t[mask])
            
        return t, signal

    def simulate_multipath_reflection(self, height: float, surface_params: SurfaceParameters,
                                    gradient: SurfaceGradient, max_reflections: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Моделирование многолучевого отражения
        
        Args:
            height (float): Высота полета
            surface_params (SurfaceParameters): Параметры поверхности
            gradient (SurfaceGradient): Градиенты поверхности
            max_reflections (int): Максимальное количество отражений
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Временная шкала и отраженный сигнал
        """
        # Генерация передаваемого сигнала
        t, tx_signal = self.generate_signal(self.T, self.B)
        
        if self.use_gpu:
            tx_signal = torch.from_numpy(tx_signal).to(self.device)
        
        # Создание массива для отраженного сигнала
        rx_signal = np.zeros_like(tx_signal)
        
        # Основное отражение
        base_delay = 2 * height / self.c
        base_grazing_angle = np.arctan(1.0 / height)
        
        # Расчет коэффициента отражения для основного луча
        base_coeff = self.calculate_reflection_coefficient(base_grazing_angle, surface_params)
        
        # Добавление основного отражения
        delay_samples = int(base_delay * 1e9)
        if delay_samples < len(tx_signal):
            rx_signal = np.zeros_like(tx_signal)
            rx_signal[delay_samples:] = tx_signal[:-delay_samples] * base_coeff
        
        # Добавление вторичных отражений
        for reflection in range(1, max_reflections + 1):
            # Расчет задержки для вторичного отражения
            secondary_delay = base_delay * (1 + 0.1 * reflection)
            secondary_coeff = base_coeff * (0.5 ** reflection)  # Уменьшение коэффициента для каждого отражения
            
            delay_samples = int(secondary_delay * 1e9)
            if delay_samples < len(tx_signal):
                temp_signal = np.zeros_like(tx_signal)
                temp_signal[delay_samples:] = tx_signal[:-delay_samples] * secondary_coeff
                rx_signal = rx_signal + temp_signal
        
        # Добавление шума
        noise_power = 0.1 * np.max(np.abs(rx_signal))
        rx_signal += np.random.normal(0, noise_power, len(rx_signal))
        
        # Нормализация сигнала
        rx_signal = rx_signal / np.max(np.abs(rx_signal))
        
        return t, rx_signal

    def plot_results(self, t, tx_signal, rx_signal):
        """
        Визуализация результатов моделирования
        
        Args:
            t (array): Временная шкала
            tx_signal (array): Передаваемый сигнал
            rx_signal (array): Отраженный сигнал
        """
        plt.figure(figsize=(12, 8))
        
        # Добавляем информацию об углах
        plt.figtext(0.02, 0.95, f"Крен: {self.roll:.1f}°", fontsize=10)
        plt.figtext(0.02, 0.92, f"Тангаж: {self.pitch:.1f}°", fontsize=10)
        
        plt.subplot(2, 1, 1)
        plt.plot(t * 1e6, tx_signal)
        plt.title('Передаваемый сигнал')
        plt.xlabel('Время (мкс)')
        plt.ylabel('Амплитуда')
        
        plt.subplot(2, 1, 2)
        plt.plot(t * 1e6, rx_signal)
        plt.title('Отраженный сигнал')
        plt.xlabel('Время (мкс)')
        plt.ylabel('Амплитуда')
        
        plt.tight_layout()
        plt.show()

# Пример использования
if __name__ == "__main__":
    # Создание экземпляра радиовысотомера
    altimeter = RadarAltimeter(use_gpu=True)
    
    # Параметры моделирования
    height = 10  # Высота в метрах
    
    # Примеры различных типов поверхностей
    surfaces = [
        SurfaceParameters(SurfaceType.LAND, roughness=0.2, moisture=0.3),
        SurfaceParameters(SurfaceType.SEA, roughness=0.4, temperature=15),
        SurfaceParameters(SurfaceType.ICE, temperature=-10, moisture=0.1),
        SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.6),
        SurfaceParameters(SurfaceType.URBAN, roughness=0.3),
        SurfaceParameters(SurfaceType.DESERT, temperature=40, moisture=0.05)
    ]
    
    # Моделирование для каждой поверхности
    for surface in surfaces:
        print(f"\nМоделирование для поверхности: {surface.surface_type.value}")
        t, rx_signal = altimeter.simulate_multipath_reflection(height, surface, SurfaceGradient(np.zeros((100, 100))), 2)
        _, tx_signal = altimeter.generate_signal(1e-6, 1e9)
        altimeter.plot_results(t, tx_signal, rx_signal) 