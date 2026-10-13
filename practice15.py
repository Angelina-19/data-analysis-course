import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare

print("=" * 60)
print("Практична робота №15, Варіант 1: Київ")
print("=" * 60)


np.random.seed(42)

base_temp = 9.5    
amplitude = 14     
city = "Київ"

def season(month):
    if month in (12, 1, 2):
        return "зима"
    if month in (3, 4, 5):
        return "весна"
    if month in (6, 7, 8):
        return "літо"
    return "осінь"

rows = []
for year in [2021, 2022, 2023, 2024]:
    for month in range(1, 13):
        seasonal = amplitude * np.cos((month - 7) / 12 * 2 * np.pi)
        noise = np.random.normal(0, 1.0)
        temp = round(base_temp + seasonal + noise, 1)
        diff = temp - base_temp
        if diff < -3:
            norm_cat = "холодніше"
        elif diff > 3:
            norm_cat = "тепліше"
        else:
            norm_cat = "звичайно"
        rows.append({
            "місто": city, "рік": year, "місяць": month,
            "температура": temp, "сезон": season(month),
            "відхилення_від_норми": norm_cat,
        })

climate = pd.DataFrame(rows)  
print("Перші 5 рядків датасету climate:")
print(climate.head())

print("\n" + "=" * 60)
print("ЗАВДАННЯ 1. Таблиця спряженості")
print("=" * 60)

table = pd.crosstab(climate["сезон"], climate["відхилення_від_норми"])
print("Абсолютні частоти:")
print(table)


table_norm = pd.crosstab(climate["сезон"], climate["відхилення_від_норми"], normalize="index")
print("\nНормалізована таблиця спряженості (normalize='index'):")
print(table_norm.round(2))

print("\n" + "=" * 60)
print("ЗАВДАННЯ 2. Критерій незалежності")
print("=" * 60)
chi2, pvalue, dof, expected = chi2_contingency(table)
print(f"Статистика χ²: {chi2:.2f}")
print(f"p-value: {pvalue:.4e}")
print(f"Ступені свободи (dof): {dof}")

print("\nМатриця очікуваних частот (expected):")
expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
print(expected_df.round(1))

print("\n" + "=" * 60)
print("ЗАВДАННЯ 3. Перевірка умови застосовності та об'єднання категорій")
print("=" * 60)
print("Попередження: в матриці expected значення для колонки 'звичайно' становлять 2.67 (< 5).")
print("Об'єднуємо 'холодніше' та 'звичайно' в категорію 'не тепліше'.")


climate["відхилення_об'єднане"] = climate["відхилення_від_норми"].apply(
    lambda x: "не тепліше" if x in ("холодніше", "звичайно") else "тепліше"
)

table_merged = pd.crosstab(climate["сезон"], climate["відхилення_об'єднане"])
print("\nНова таблиця спряженості (з об'єднаними категоріями):")
print(table_merged)

chi2_m, pvalue_m, dof_m, expected_m = chi2_contingency(table_merged)
print(f"\nchi2 = {chi2_m:.2f}, p-value = {pvalue_m:.4e}, dof = {dof_m}")
print("\nНові очікувані частоти (усі >= 5):")
print(pd.DataFrame(expected_m, index=table_merged.index, columns=table_merged.columns).round(1))

print("\n" + "=" * 60)
print("ЗАВДАННЯ 4. Критерій узгодженості для сезонів")
print("=" * 60)
observed_seasons = climate["сезон"].value_counts().reindex(["зима", "весна", "літо", "осінь"])
n_seasons = len(climate)
expected_seasons = [0.25 * n_seasons] * 4

res_seasons = chisquare(f_obs=observed_seasons, f_exp=expected_seasons)
print(f"Фактичні частоти сезонів:\n{observed_seasons}")
print(f"Статистика критерію узгодженості: {res_seasons.statistic:.2f}")
print(f"p-value: {res_seasons.pvalue}")

print("\n" + "=" * 60)
print("ЗАВДАННЯ 5. Власна заявлена пропорція для відхилень")
print("=" * 60)

observed_norms = climate["відхилення_від_норми"].value_counts().reindex(["звичайно", "тепліше", "холодніше"])
n_total = len(climate)
expected_norms = [0.50 * n_total, 0.25 * n_total, 0.25 * n_total]

res_norms = chisquare(f_obs=observed_norms, f_exp=expected_norms)
print(f"Фактичні частоти відхилень:\n{observed_norms}")
print(f"Очікувані частоти за гіпотезою: {expected_norms}")
print(f"Статистика критерію узгодженості: {res_norms.statistic:.2f}")
print(f"p-value: {res_norms.pvalue:.4f}")


# ==============================================================================
# ВІДПОВІДІ НА КОНТРОЛЬНІ ПИТАННЯ 
# ==============================================================================

# 1. Чому в статистиці χ² використовується саме квадрат відхилення і ділення на очікувану частоту?
# Відповідь: Квадрат відхилення (O - E)^2 використовується для того, щоб додатні 
# та від'ємні різниці між фактичними (O) та очікуваними (E) частотами не взаємокомпенсувалися 
# при підсумовуванні. Ділення на очікувану частоту (E) необхідне для нормування: 
# абсолютне відхилення має різну вагу залежно від масштабу даних (наприклад, різниця в 5 одиниць 
# є величезною для частоти 10, але незначною для частоти 1000).

# 2. Як обчислюється очікувана частота клітинки таблиці спряженості (рядок × стовпець / загальна сума) 
# і чому саме ця формула коректна під припущенням незалежності?
# Відповідь: Очікувана частота для конкретної клітинки обчислюється як добуток суми її рядка 
# на суму її стовпця, поділений на загальну суму спостережень. Ця формула випливає з теорії 
# ймовірностей для незалежних подій: ймовірність настання двох незалежних подій дорівнює добутку їхніх 
# маргінальних імовірностей P(A і B) = P(A) * P(B). Множачи це на загальну кількість n, отримуємо 
# очікувану частоту для комірки за умови нульової гіпотези незалежності.

# 3. У чому різниця між "p-value велике — немає підстав відхилити H0" і "p-value велике — доведено, що змінні незалежні"?
# Відповідь: У статистичному тестуванні велике p-value (наприклад, > 0.05) означає лише те, що 
# у нас недостатньо доказів проти нульової гіпотези (H0). Це не доводить істинність H0 (не доводить 
# повну незалежність), а лише свідчить про те, що наявні дані не суперечать припущенню про незалежність 
# (можливо, вибірка занадто мала, щоб виявити слабкий зв'язок).

# 4. Що робити, якщо очікувана частота в якійсь клітинці таблиці спряженості менша за 5, і чому це проблема?
# Відповідь: Це проблема тому, що апроксимація розподілу статистики χ² стає неточною, що спотворює 
# значення p-value і робить висновки ненадійними. Якщо очікувана частота менша за 5 у деяких клітинках, 
# стандартним вирішенням є об'єднання суміжних категорій (як ми зробили у Завданні 3) або збір більшої вибірки.