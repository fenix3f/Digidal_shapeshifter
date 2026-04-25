import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# ЗАГРУЗКА ДАННЫХ
# ----------------------------
df = pd.read_csv('full_with_base.csv')

OKRUGA = df['Округ'].unique()
LAST_YEAR = 2025

# ----------------------------
# ПАРАМЕТРЫ МОДЕЛИ
# ----------------------------
FREEZE_FROM, FREEZE_TO = 2006, 2020
freeze = df[(df['Год'] >= FREEZE_FROM) & (df['Год'] <= FREEZE_TO)]

WEIGHTS = {
    'z_замещение':  +0.25,
    'z_нагрузка':   -0.25,
    'z_прирост':    +0.15,
    'z_миграция':   +0.10,
    'z_старение':   -0.10,
    'z_финустой':   -0.15,
}

SIGMOID_SCALE = 1.5

# ----------------------------
# ФУНКЦИИ
# ----------------------------
def zscore(series, ref):
    median = ref.median()
    std = ref.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - median) / std


def calc_ipu(df_part):
    score = 0
    for col, w in WEIGHTS.items():
        score += df_part[col] * w
    return 100 / (1 + np.exp(-score * SIGMOID_SCALE))


def simulate(df_o, wage, pension, birth, death, migration):
    df_o = df_o.copy()
    future_mask = df_o['Год'] > LAST_YEAR

    # --- ключевая логика ---

    # замещение (пенсия / зарплата)
    df_o.loc[future_mask, 'коэффициент_замещения'] *= (pension / wage)

    # нагрузка (больше смертность → хуже)
    df_o.loc[future_mask, 'коэффициент_нагрузки'] *= (death / birth)

    # демография
    df_o.loc[future_mask, 'ест_прирост_‰'] *= (birth / death)
    df_o.loc[future_mask, 'миграция_‰'] *= migration

    # старение
    df_o.loc[future_mask, 'доля_пенсионеров'] *= (death / birth)

    # фин устойчивость (зарплаты быстрее пенсий = хорошо)
    df_o.loc[future_mask, 'опережение_зп'] *= (wage / pension)

    # --- z-score ---
    df_o['z_замещение'] = zscore(df_o['коэффициент_замещения'], freeze['коэффициент_замещения'])
    df_o['z_нагрузка']  = zscore(df_o['коэффициент_нагрузки'],  freeze['коэффициент_нагрузки'])
    df_o['z_прирост']   = zscore(df_o['ест_прирост_‰'],         freeze['ест_прирост_‰'])
    df_o['z_миграция']  = zscore(df_o['миграция_‰'],            freeze['миграция_‰'])
    df_o['z_старение']  = zscore(df_o['доля_пенсионеров'],      freeze['доля_пенсионеров'])
    df_o['z_финустой']  = zscore(df_o['опережение_зп'],         freeze['опережение_зп'])

    df_o['ИПУ_new'] = calc_ipu(df_o)

    return df_o


# ----------------------------
# UI
# ----------------------------
st.title("📊 Индекс пенсионной устойчивости")

okrug = st.selectbox("Выберите округ", OKRUGA)

st.sidebar.header("Сценарий")

wage = st.sidebar.slider("Зарплата", 0.5, 1.5, 1.0, 0.05)
pension = st.sidebar.slider("Пенсия", 0.5, 1.5, 1.0, 0.05)
birth = st.sidebar.slider("Рождаемость", 0.5, 1.5, 1.0, 0.05)
death = st.sidebar.slider("Смертность", 0.5, 1.5, 1.0, 0.05)
migration = st.sidebar.slider("Миграция", 0.5, 1.5, 1.0, 0.05)

# ----------------------------
# ДАННЫЕ ПО ОКРУГУ
# ----------------------------
df_o = df[df['Округ'] == okrug].copy()

df_new = simulate(df_o, wage, pension, birth, death, migration)

# ----------------------------
# ГРАФИК
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

hist = df_o[df_o['Год'] <= LAST_YEAR]
future = df_o[df_o['Год'] > LAST_YEAR]

new_hist = df_new[df_new['Год'] <= LAST_YEAR]
new_future = df_new[df_new['Год'] > LAST_YEAR]

# базовый
ax.plot(hist['Год'], hist['ИПУ'], label='Факт', linewidth=2)
ax.plot(future['Год'], future['ИПУ'], linestyle='--', label='Прогноз', linewidth=2)

# сценарий
ax.plot(new_future['Год'], new_future['ИПУ_new'],
        linestyle='--', linewidth=2, label='Ваш сценарий')

# линия разделения
ax.axvline(x=LAST_YEAR, color='gray', linestyle='--', alpha=0.5)

ax.set_title(f'{okrug}')
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)