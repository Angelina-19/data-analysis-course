import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress

# ==============================================================================
# ПРАКТИЧНА РОБОТА №22
# ==============================================================================

print("=" * 60)
print("ЗАВДАННЯ 1: Побудова багаторічного ряду (Варіант 1 — Київ)")
print("=" * 60)

np.random.seed(1)                
base_temp = 9.5                  
amplitude = 14                   
trend_per_year = 0.04             
noise_scale = 1.0                
city = "Київ"

n_years = 10
dates = pd.date_range(start="2015-01-01", periods=n_years * 12, freq="MS")
month = dates.month
years_elapsed = (dates.year - dates.year[0]) + (dates.month - 1) / 12   

seasonal = amplitude * np.cos((month - 7) / 12 * 2 * np.pi)   
trend = trend_per_year * years_elapsed
noise = np.random.normal(0, noise_scale, len(dates))

temp = np.round(base_temp + trend + seasonal + noise, 1)
climate_ts = pd.Series(temp, index=dates, name="температура")

print("Перші 12 значень ряду:")
print(climate_ts.head(12))
print(f"Довжина ряду: {len(climate_ts)} місяців\n")


fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(climate_ts.index, climate_ts.values, color='tab:red', alpha=0.8)
ax.set_xlabel("Час (Місяць)")
ax.set_ylabel("Температура, °C")
ax.set_title(f"Завдання 1: Багаторічний кліматичний ряд — {city} (2015–2024)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.show()


print("=" * 60)
print("ЗАВДАННЯ 2: Ковзне середнє при різних вікнах")
print("=" * 60)

ma3 = climate_ts.rolling(window=3, center=True).mean()
ma12_centered = climate_ts.rolling(window=12, center=True).mean()
ma12_trailing = climate_ts.rolling(window=12, center=False).mean()


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(climate_ts.index, climate_ts.values, alpha=0.3, label="оригінал", color="gray")
ax.plot(climate_ts.index, ma3.values, label="вікно = 3 (центроване)", color="orange")
ax.plot(climate_ts.index, ma12_centered.values, label="вікно = 12 (центроване)", color="blue", linewidth=2)
ax.plot(climate_ts.index, ma12_trailing.values, label="вікно = 12 (трейлінгове)", color="green", linestyle="--")

ax.legend()
ax.set_title("Завдання 2: Ковзне середнє — вплив розміру вікна та лаг")
ax.set_xlabel("Час")
ax.set_ylabel("Температура, °C")
ax.grid(True, linestyle="--", alpha=0.5)
plt.show()


print("=" * 60)
print("ЗАВДАННЯ 3: Декомпозиція seasonal_decompose")
print("=" * 60)

decomposition = seasonal_decompose(climate_ts, model="additive", period=12)

fig = decomposition.plot()
fig.set_size_inches(9, 7)
plt.tight_layout()
plt.show()

print("Перші 12 значень сезонної складової:")
print(decomposition.seasonal.head(12).round(1))
print(f"Стандартне відхилення залишку (resid std): {decomposition.resid.dropna().std():.2f}\n")


print("=" * 60)
print("ЗАВДАННЯ 4: Прогноз на 12 місяців уперед")
print("=" * 60)

t = np.arange(len(climate_ts))
known = decomposition.trend.notna().values
fit = linregress(t[known], decomposition.trend[known])

n_ahead = 12
t_future = np.arange(len(climate_ts), len(climate_ts) + n_ahead)
trend_forecast = fit.intercept + fit.slope * t_future

seasonal_by_month = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()
future_dates = pd.date_range(climate_ts.index[-1] + pd.offsets.MonthBegin(1),
                              periods=n_ahead, freq="MS")
seasonal_forecast = seasonal_by_month.reindex(future_dates.month).values

forecast = pd.Series(trend_forecast + seasonal_forecast, index=future_dates)
print("Прогноз на 2025 рік (°C):")
print(forecast.round(1))
print()


fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(climate_ts.index, climate_ts.values, label="Історія (2015-2024)", color="gray", alpha=0.6)
ax.plot(forecast.index, forecast.values, label="Прогноз (2025)", color="red", marker="o")
ax.legend()
ax.set_title("Завдання 4: Прогноз температури для м. Київ на 2025 рік")
ax.set_xlabel("Час")
ax.set_ylabel("Температура, °C")
ax.grid(True, linestyle="--", alpha=0.5)
plt.show()


print("=" * 60)
print("ЗАВДАННЯ 5: Перевірка статистичної значущості тренду")
print("=" * 60)

deseason = climate_ts - decomposition.seasonal
t_years = np.arange(len(climate_ts)) / 12
trend_test = linregress(t_years, deseason.values)

print(f"Нахил = {trend_test.slope:.4f} °C/рік")
print(f"Стандартна похибка (stderr) = {trend_test.stderr:.4f}")
print(f"p-value = {trend_test.pvalue:.4e}")

ci_lower = trend_test.slope - 1.96 * trend_test.stderr
ci_upper = trend_test.slope + 1.96 * trend_test.stderr
print(f"95% довірчий інтервал нахилу: [{ci_lower:.4f}, {ci_upper:.4f}]\n")


# ==============================================================================
# ВІДПОВІДІ НА КОНТРОЛЬНІ ПИТАННЯ 
# ==============================================================================

# ПИТАННЯ 1: Чому часовий ряд не можна аналізувати тими самими методами, що 
# звичайну незалежну вибірку (Лекції 12–16), і що саме в даних це порушує?
# КОМЕНТАР: Звичайна вибірка вимагає незалежності та однакового розподілу 
# спостережень (i.i.d.). У часових рядах сусідні точки міцно пов'язані між собою 
# часовою залежністю (автокореляцією). Це порушує припущення про незалежність: 
# температура поточного місяця сильно залежить від попереднього місяця та сезону, 
# що робить стандартні тести для незалежних вибірок недійсними або дає 
# оманливо малі p-value.

# ПИТАННЯ 2: Чому надто мале й надто велике вікно ковзного середнього — обидва 
# проблема, хоч і по-різному, і як розмір вікна варто узгоджувати з періодом сезонності?
# КОМЕНТАР: Надто мале вікно (наприклад, 2-3 місяці) не здатне повністю прибрати 
# річний шум і сезонні коливання. Надто велике вікно (наприклад, 24 місяці чи більше) 
# "розмиває" і руйнує справжню структуру тренду та важливі локальні зміни. 
# Розмір вікна обов'язково варто узгоджувати з періодом сезонності: для річної 
# сезонності ідеальним є вікно, кратне 12 місяцям, оскільки повний цикл 
# взаємно компенсується.

# ПИТАННЯ 3: У чому різниця між трейлінговим (center=False) і центрованим 
# (center=True) ковзним середнім, і чому центроване недоступне "в реальному часі"?
# КОМЕНТАР: Трейлінгове середнє обчислюється на основі поточного та попередніх 
# періодів, тому воно запізнюється (має лаг) відносно реального тренду. 
# Центроване середнє розподіляє вікно симетрично навколо поточного моменту часу, 
# усуваючи лаг. Центроване середнє неможливо використати "в реальному часі" 
# для прогнозування майбутнього, оскільки для його розрахунку потрібні дані 
# з майбутнього (події, що ще не настали).

# ПИТАННЯ 4: Чому саме p-value нахилу тренду, а не тільки знак slope, визначає, 
# чи є підстави говорити про статистично підтверджений тренд, а не про шум?
# КОМЕНТАР: Знак slope показує лише напрямок лінії на графіку (вгору чи вниз) 
# на наявній вибірці. Проте через випадковий шум (особливо на коротких чи 
# зашумлених рядах) лінія може випадково йти вгору навіть за відсутності реального 
# тренду. Саме p-value (у поєднанні з довірчим інтервалом) дає змогу оцінити 
# ймовірність того, що такий нахил виник випадково через шум. Якщо p-value вище 
# за рівень значущості (наприклад, 0.05), нахил вважається статистично 
# випадковим, а не підтвердженим трендом.