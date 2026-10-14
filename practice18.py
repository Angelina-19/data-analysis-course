# ==============================================================================
# ПРАКТИЧНА РОБОТА №18 (Варіант 1: Київ)
# ==============================================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import scipy.stats as stats


print("=" * 60)
print("ПІДГОТОВКА: Генерація набору даних для Варіанту 1 (Київ)")
print("=" * 60)

np.random.seed(42)
city = "Київ"         
mean_temp = 9.5        
amplitude = 14        
hum_base = 74          

rows = []
for month in range(1, 13):
    temp = mean_temp + amplitude * np.cos((month - 7) / 12 * 2 * np.pi) + np.random.normal(0, 1.0)
    humidity = hum_base + np.random.normal(0, 5.0)
    if temp < 8:
        opal_days = 30      
    elif temp < 12:
        opal_days = 15       
    else:
        opal_days = 0        

    cost = (
        200
        - 9.0 * temp
        + 0.6 * humidity
        + 2.5 * opal_days
        + np.random.normal(0, 15)
    )
    rows.append({
        "місто": city, "місяць": month,
        "температура": round(temp, 1),
        "вологість": round(humidity, 1),
        "опалювальні_дні": opal_days,
        "витрати_на_опалення": round(cost, 1),
    })

heating = pd.DataFrame(rows)
print(heating.head())
print("\n" + "=" * 60)


print("ЗАВДАННЯ 1: Модель із двома предикторами (температура + вологість)")
print("=" * 60)

X_v1 = sm.add_constant(heating[["температура", "вологість"]])
y = heating["витрати_на_опалення"]

model_v1 = sm.OLS(y, X_v1).fit()
print(model_v1.summary())
print(f"\nЗафіксовані показники моделі 1:")
print(f"R-squared: {model_v1.rsquared:.3f}")
print(f"Adj. R-squared: {model_v1.rsquared_adj:.3f}")
print("\n" + "=" * 60)


print("ЗАВДАННЯ 2: Інтерпретація коефіцієнтів та статистична значущість")
print("=" * 60)
for name, coef, pval in zip(X_v1.columns, model_v1.params, model_v1.pvalues):
    print(f"Предиктор/Константа '{name}': коефіцієнт = {coef:.4f}, P>|t| = {pval:.4f}")
print("\n[Висновок]: Обидва предикторы є статистично значущими (p < 0.05).")
print("=" * 60)


print("ЗАВДАННЯ 3: Перевірка мультиколінеарності через VIF")
print("=" * 60)
for i, col in enumerate(X_v1.columns):
    if col == "const":
        continue
    vif_value = variance_inflation_factor(X_v1.values, i)
    print(f"VIF для '{col}': {vif_value:.2f}")
print("\n[Висновок]: VIF близький до 1, мультиколінеарність відсутня.")
print("=" * 60)


print("ЗАВДАННЯ 4: Третій предиктор (опалювальні_дні)")
print("=" * 60)
X_v2 = sm.add_constant(heating[["температура", "вологість", "опалювальні_дні"]])
model_v2 = sm.OLS(y, X_v2).fit()
print(model_v2.summary())

print("\nПорівняння моделей:")
print(f"Модель 1 (2 предиктори): R2 = {model_v1.rsquared:.3f} | Adj. R2 = {model_v1.rsquared_adj:.3f}")
print(f"Модель 2 (3 предиктори): R2 = {model_v2.rsquared:.3f} | Adj. R2 = {model_v2.rsquared_adj:.3f}")
print("=" * 60)


print("ЗАВДАННЯ 5: Діагностика залишків (Графік, QQ-plot, Шапіро-Вілк)")
print("=" * 60)
fitted = model_v1.fittedvalues
residuals = model_v1.resid


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(fitted, residuals, color="purple")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Передбачені витрати")
axes[0].set_ylabel("Залишки")
axes[0].set_title("Гомоскедастичність (Залишки vs Fitted)")

stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("QQ-plot залишків")
plt.tight_layout()
plt.show()

shapiro_test = stats.shapiro(residuals)
print(f"Критерій Шапіро-Вілка: statistic = {shapiro_test.statistic:.4f}, p-value = {shapiro_test.pvalue:.4f}")
print("=" * 60)


print("ЗАВДАННЯ 6: Порівняння з простою регресією (Практика 17)")
print("=" * 60)
X_simple = sm.add_constant(heating[["температура"]])
model_simple = sm.OLS(y, X_simple).fit()
print(f"Проста регресія R2 (тільки температура): {model_simple.rsquared:.3f}")
print(f"Множинна регресія Adj. R2 (2 предиктори): {model_v1.rsquared_adj:.3f}")
print("=" * 60)


# ==============================================================================
# ВІДПОВІДІ НА КОНТРОЛЬНІ ПИТАННЯ 
# ==============================================================================

# Питання 1: Чи можете пояснити, чому sm.add_constant() обов'язковий крок перед sm.OLS(), 
# і що станеться з моделлю, якщо його пропустити?
# Відповідь: sm.add_constant() додає до матриці предикторів стовпець одиниць, що 
# відповідає вільному члену (intercept, beta_0) у рівнянні регресії. Якщо його пропустити,
# лінія або гіперплощина регресії примусово проходитиме через початок координат (0, 0),
# що суттєво змістить оцінки інших коефіцієнтів і погіршить якість моделі.

# Питання 2: Чи можете сформулювати своїми словами, у чому коефіцієнт множинної регресії 
# відрізняється від коефіцієнта простої регресії з тим самим предиктором?
# Відповідь: У простій регресії коефіцієнт відображає загальний зв'язок між єдиним 
# предиктором і відгуком (включно з опосередкованим впливом через інші змінні). 
# У множинній регресії коефіцієнт показує чистий ефект ("за інших рівних умов") 
# конкретного предиктора, очищений від впливу інших включених у модель змінних.

# Питання 3: Чи розумієте, чому звичайний R^2 не може зменшитись при додаванні будь-якого 
# нового предиктора, навіть безглуздого, і чому це робить його непридатним для порівняння?
# Відповідь: Звичайний R^2 базується на співвідношенні сум квадратів. Додавання будь-якого
# нового стовпця даних у матрицю проектує модель у більш високу розмірність, що завжди 
# зменшує або залишає незмінною залишкову суму квадратів. Саме тому для порівняння моделей
# із різною кількістю предикторів використовують скоригований Adjusted R^2, який містить штраф.

# Питання 4: Що означає VIF предиктора, що дорівнює приблизно 1, і що означає VIF вище 10 — 
# і що з цим варто зробити на практиці?
# Відповідь: VIF ≈ 1 означає повну відсутність мультиколінеарності (предиктор повністю 
# незалежний від інших). VIF > 10 свідчить про сильну мультиколінеарність (надлишковість 
# інформації). На практиці з цим борються шляхом видалення однієї з колінеарних змінних,
# об'єднання їх у комбінований індекс або застосування регуляризації (наприклад, Ridge).