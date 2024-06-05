import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json


data_folder = 'data/'
radio_json = [
    'noc_bmd.json',
    'noc_bv.json',
    'noc_col.json',
    'noc_gwt.json',
    'noc_he.json',
    'noc_hg.json',
    'noc_hw.json',
    'noc_sar.json',
    'noc_tt.json',
    'noc_tur.json'
]

train_temperature = json.load(open(f'{data_folder}temp_galt.json'))
train_temperature = pd.DataFrame(train_temperature).rename(columns={"dt": "time"})
train_temperature = train_temperature[['time', 'main']]
train_temperature['temp'] = train_temperature['main'].apply(lambda x: x['temp'])
train_temperature = train_temperature[['time', 'temp']]
train_temperature['time'] = pd.to_datetime(train_temperature['time'], unit='s')
train_temperature['time'] = train_temperature['time'].dt.tz_localize('Etc/GMT+2')
print(train_temperature)

radio_stats_all = []

for json_file in radio_json:
    radio_info = json.load(open(f'{data_folder}{json_file}'))
    azimuth = radio_info['meta']['azimuth']
    height = radio_info['meta']['height']
    radio_info = pd.DataFrame(radio_info["data"]).rename(columns={"sensor": "rsl"})
    radio_info['time'] = pd.to_datetime(radio_info['time'], format='%Y-%m-%d %H:%M:%S')
    radio_info['time'] = radio_info['time'].dt.tz_localize('US/Pacific')
    radio_info = pd.merge(radio_info, train_temperature, on='time')
    radio_info['rsl_delta'] = abs(radio_info['rsl'] - stats.mode(radio_info['rsl'])[0])
    radio_info['azimuth'] = azimuth
    radio_info['azimuth_sin'] = np.sin(np.radians(azimuth))
    radio_info['azimuth_cos'] = np.cos(np.radians(azimuth))
    radio_info['height'] = height
    radio_stats_all.extend(radio_info.to_dict(orient='records'))

df = pd.DataFrame(radio_stats_all)
df = df.sort_values(by='time').reset_index(drop=True)

print()

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

X = df[['temp', 'height', 'azimuth_sin', 'azimuth_cos']]
y = (df['rsl_delta'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

regressor = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

model = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
model.fit(X_train, y_train)

model = model.best_estimator_

y_prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2: {r2}")


def test_alg(input_temp=25, input_height=35, input_azimuth=0):

    inputs = pd.DataFrame({
        'temp': [input_temp], 'height': [input_height],
        'azimuth_sin': [np.sin(input_azimuth)], 'azimuth_cos': [np.cos(input_azimuth)],
    })
    predicted_rsl = model.predict(inputs)
    print(f"\nHeight: {input_height} meters || Azimuth: {input_azimuth} degrees"
          f"\nPotential RSL change at {inputs['temp'][0]}C: {predicted_rsl[0]}dBm")
    if predicted_rsl[0] > 2:
        print("Links at this height and azimuth may experience link degradation at the selected temperature.")
    else:
        print("Links at this height and azimuth should perform as expected.")


test_alg()
test_alg(10, 45, 0)
test_alg(10, 45, 90)
test_alg(10, 45, 180)
test_alg(10, 45, 270)


# fig, ax1 = plt.subplots(figsize=(20, 6))
#
# ax1.scatter(df['height'], df['rsl_delta'], alpha=0.5, label='rsl', color='red')
# ax1.set_xlabel('height')
# ax1.set_ylabel('rsl change')
# ax1.tick_params(axis='y')
#
# ax2 = ax1.twinx()
# ax2.scatter(df['height'], df['azimuth'], alpha=0.5, label='temp', color='blue')
# ax2.set_ylabel('azimuth')
# ax2.tick_params(axis='y')
#
# fig.tight_layout()
# plt.xlim(0,60)
# plt.show()

def plot_rsl_change(input_height=35, input_azimuth=0):
    temperatures = np.arange(10, 35, 1)
    predicted_rsl = []

    for temp in temperatures:
        inputs = pd.DataFrame({
            'temp': [temp], 'height': [input_height],
            'azimuth_sin': [np.sin(np.radians(input_azimuth))],
            'azimuth_cos': [np.cos(np.radians(input_azimuth))],
        })
        predicted_rsl.append(model.predict(inputs)[0])

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, predicted_rsl)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Potential RSL Change (dBm)')
    plt.title(f'RSL Change vs. Temperature (Height: {input_height}m, Azimuth: {input_azimuth}°)')
    plt.grid(True)
    plt.show()


interact(plot_rsl_change,
         input_height=widgets.IntSlider(min=10, max=60, step=5, value=35, description='Height (m)'),
         input_azimuth=widgets.IntSlider(min=0, max=355, step=5, value=0, description='Azimuth (°)'))
