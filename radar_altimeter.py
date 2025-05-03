import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from enum import Enum

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

class SurfaceParameters:
    def __init__(self, surface_type, roughness=0.1, moisture=0.0, temperature=20.0):
        self.surface_type = surface_type
        self.roughness = roughness  # Шероховатость поверхности (0-1)
        self.moisture = moisture    # Влажность (0-1)
        self.temperature = temperature  # Температура в градусах Цельсия

class RadarAltimeter:
    def __init__(self):
        self.c = 3e8  # Скорость света, м/с
        self.f0 = 4.3e9  # Центральная частота, Гц
        self.B = 144e6  # Полоса частот, Гц
        self.T = 1e-6  # Длительность сигнала, с
        self.roll = 0.0  # Крен, градусы
        self.pitch = 0.0  # Тангаж, градусы
        self.modulation_type = ModulationType.LFM  # Тип модуляции
        self.pulse_width = 0.1e-6  # Длительность импульса для ИМ, с
        self.pulse_period = 1e-3    # Период повторения импульсов для ИМ, с
        self.tx_power = 0.1  # Мощность передатчика, Вт
        self.antenna_gain = 10  # Усиление антенны, дБ
        
    def calculate_reflection_coefficient(self, grazing_angle, surface_params):
        """
        Расчет коэффициента отражения
        
        Args:
            grazing_angle (float): Угол скольжения, рад
            surface_params (SurfaceParameters): Параметры поверхности
            
        Returns:
            float: Коэффициент отражения
        """
        # Учитываем крен и тангаж при расчете эффективного угла скольжения
        effective_angle = grazing_angle + np.radians(self.roll) * np.cos(np.radians(self.pitch))
        
        # Базовый коэффициент отражения
        base_coeff = 0.5 * (1 + np.cos(effective_angle))
        
        # Учет шероховатости поверхности
        roughness_factor = 1 - surface_params.roughness * np.sin(effective_angle)
        
        # Учет влажности
        moisture_factor = 1 - 0.5 * surface_params.moisture * (1 - np.cos(effective_angle))
        
        # Учет температуры
        temp_factor = 1 - 0.1 * abs(surface_params.temperature) / 30
        
        # Итоговый коэффициент отражения
        reflection_coeff = base_coeff * roughness_factor * moisture_factor * temp_factor
        
        return reflection_coeff

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

    def simulate_reflection(self, height, surface_params):
        """
        Моделирование отражения сигнала от поверхности
        
        Args:
            height (float): Высота полета, м
            surface_params (SurfaceParameters): Параметры поверхности
            
        Returns:
            tuple: (массив времени, массив отраженного сигнала)
        """
        # Генерация передаваемого сигнала
        t, tx_signal = self.generate_signal(self.T, self.B)
        
        # Расчет времени задержки с учетом крена и тангажа
        effective_height = height / np.cos(np.radians(self.roll)) / np.cos(np.radians(self.pitch))
        delay = 2 * effective_height / self.c
        
        # Расчет затухания сигнала
        R = effective_height / np.cos(np.arctan(1.0 / effective_height))  # Наклонная дальность
        attenuation = 1.0 / (R**2)  # Обратно пропорционально квадрату расстояния
        
        # Расчет коэффициента отражения
        grazing_angle = np.arctan(1.0 / effective_height)
        reflection_coeff = self.calculate_reflection_coefficient(grazing_angle, surface_params)
        
        # Учет мощности передатчика и усиления антенны
        power_factor = np.sqrt(self.tx_power) * 10**(self.antenna_gain/20)
        
        # Создание отраженного сигнала с учетом задержки и затухания
        rx_signal = np.zeros_like(tx_signal)
        delay_samples = int(delay * 1e9)  # Преобразуем задержку в количество отсчетов
        
        if delay_samples < len(tx_signal):
            rx_signal[delay_samples:] = tx_signal[:-delay_samples] * attenuation * reflection_coeff * power_factor
            
            # Добавляем шум
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
    altimeter = RadarAltimeter()
    
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
        t, rx_signal = altimeter.simulate_reflection(height, surface)
        _, tx_signal = altimeter.generate_signal(1e-6, 1e9)
        altimeter.plot_results(t, tx_signal, rx_signal) 