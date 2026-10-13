import numpy as np

print("=" * 50)
print("ПРАКТИЧНА РОБОТА №1 — ВАРІАНТ 3 (ОДЕСА)")
print("=" * 50)


print("\n[ЗАВДАННЯ 1] Створення масивів та перевірка shape / dtype")
temps = np.array([-1, 0, 4, 10, 16, 20, 23, 22, 17, 11, 5, 1])
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
month_numbers = np.arange(1, 13)

print("temps — shape:", temps.shape, "| dtype:", temps.dtype)
print("days_in_month — shape:", days_in_month.shape, "| dtype:", days_in_month.dtype)
print("month_numbers — shape:", month_numbers.shape, "| dtype:", month_numbers.dtype)



print("\n[ЗАВДАННЯ 2] Векторизовані обчислення")

temps_f = temps * 9 / 5 + 32
print("Температури у °F:", temps_f)


warm_days_count = days_in_month[temps > 15].sum()
print("Кількість днів у році з температурою > 15°C:", warm_days_count)


total_days = days_in_month.sum()
print("Сума днів у році:", total_days)



print("\n[ЗАВДАННЯ 3] Індексація і фільтрація")
max_temp = temps.max()
max_index = temps.argmax()
min_temp = temps.min()
min_index = temps.argmin()

print(f"Найтепліший місяць: індекс {max_index}, температура {max_temp}°C")
print(f"Найхолодніший місяць: індекс {min_index}, температура {min_temp}°C")


cold_months = month_numbers[temps < 0]
print("Номери місяців з мінусовою температурою:", cold_months)


sorted_temps = np.sort(temps)
print("Відсортовані температури (зростання):", sorted_temps)
print("Оригінальний масив залишився незмінним:", temps)



print("\n[ЗАВДАННЯ 4] Двовимірний масив і агрегація (axis)")

sunny_hours = np.array([60, 65, 120, 180, 250, 290, 310, 290, 220, 150, 70, 50])


data = np.array([temps, days_in_month, sunny_hours])
print("Форма масиву data:", data.shape)


row_sums = data.sum(axis=1)
row_means = data.mean(axis=1)
print("Суми по рядках:", row_sums)
print("Середні значення по рядках:", row_means)


col_sums = data.sum(axis=0)
print("Суми по стовпцях (по місяцях):", col_sums)



print("\n[ЗАВДАННЯ 5] reshape і broadcasting")
rows = np.array([temps, days_in_month])
correction = np.array([1, 0])

print("Форма rows:", rows.shape)
print("Форма correction:", correction.shape)


try:
  result_error = rows + correction
except Exception as e:
  print("Очікувана помилка broadcasting:", e)


correction_reshaped = correction.reshape(2, 1)
print("Форма correction після reshape:", correction_reshaped.shape)

rows_corrected = rows + correction_reshaped
print("Успішний результат додавання після corrections:\n", rows_corrected)
print("=" * 50)


# =========================================================================
# ПИСЬМОВІ ВІДПОВІДІ НА ПИТАННЯ ДО ПРАКТИЧНОЇ 
# =========================================================================

# 1. Чи можете ви пояснити різницю між shape == (12,) і shape == (1, 12)?
# ВІДПОВІДЬ: shape == (12,) означає одновимірний масив (вектор) із 12 елементів 
# (наприклад, послідовність значень для 12 місяців). shape == (1, 12) — це двовимірний 
# масив (матриця), що складається з 1 рядка та 12 стовпців. Хоча загальна кількість 
# елементів однакова (12), їхня внутрішня структура різна, що впливає на матричні операції.

# 2. Чи розумієте ви, чому temps[temps > 15] повертає масив іншої довжини, ніж temps?
# ВІДПОВІДЬ: Булева індексація фільтрує елементи масиву за заданою умовою. 
# У масиві temps для Одеси лише 5 місяців мають температуру вище 15°C, тому повертається 
# масив, що містить тільки ці 5 елементів, і його довжина дорівнює 5, а не 12.

# 3.Чи можете ви пояснити, що станеться, якщо додати масив форми (3,) до масиву форми (3, 12)?
# ВІДПОВІДЬ: Виникне помилка ValueError через несумісність форм. Правила broadcasting 
# порівнюють розмірності з кінця (праворуч наліво): остання розмірність масиву (3, 12) дорівнює 12, 
# а масиву (3,) — 3. Оскільки 12 і 3 не збігаються (і жодна з них не дорівнює 1), NumPy 
# не може виконати трансляцію (broadcasting) і видає помилку.

# 4. Чи можете ви пояснити, чому rows + correction спочатку дає помилку, а rows + correction.reshape(2, 1) — ні?
# ВІДПОВІДЬ: Спочатку rows має форму (2, 12), а correction — (2,). NumPy намагається вирівняти їх 
# з кінця: порівнює 12 та 2, що викликає конфлікт. Після застосування .reshape(2, 1) форма 
# correction стає (2, 1). Тепер під час broadcasting остання розмірність (1) успішно 
# розширюється (дублюється) уздовж 12 стовпців для кожного з 2 рядків окремо, що дозволяє 
# коректно виконати поелементне додавання без помилок.