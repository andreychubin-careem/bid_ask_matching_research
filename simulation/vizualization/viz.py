import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

from ipywidgets import IntSlider, interact


warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


class Visualizer(object):
    def __init__(self):
        self._cities = None
        self._fracs = None
        self._horizons = None
        self._drs = None
        self._ms = None
        self._nps = None
        self._capacities = None
        self._functions = None
        self.targets = [
            'num_clients_with_no_reach', 'num_captains_with_no_requests', 'num_clients_with_no_handshake_options'
        ]
        self.enriched = False

    def enrich(self, data: pd.DataFrame) -> None:
        self._cities = data.city.unique()
        self._fracs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]  # data.driver_frac.unique()
        self._horizons = data.horizon.unique()
        self._drs = data.dr.unique()
        self._ms = (data.m.min(), data.m.max())
        self._nps = data.n_possible.unique()
        self._capacities = (data.capacity.min(), data.capacity.max())
        self._functions = data.matching_fn.unique()
        self.enriched = True

    def plot_counts(self, data: pd.DataFrame, save: bool = False) -> None:
        if not self.enriched:
            self.enrich(data)

        @interact(city=self._cities, driver_frac=self._fracs, horizon=self._horizons)
        def f(city: str, driver_frac: float, horizon: str) -> None:
            pop = (
                data[
                    (data.city == city) &
                    (data.dr == data.dr.unique().tolist()[0]) &
                    (data.driver_frac == driver_frac) &
                    (data.horizon == horizon)
                    ][['datetime', 'num_clients', 'num_captains']]
                .drop_duplicates()
            )

            pop['datetime'] = pop['datetime'].astype(str).apply(lambda x: x.split(' ')[1])
            pop = pop.set_index('datetime').unstack().reset_index(drop=False)
            pop.columns = ['type', 'datetime', 'cnt']

            plt.figure(figsize=(16, 8))
            sns.lineplot(data=pop, x='datetime', y='cnt', hue='type')
            plt.xticks(rotation=90)  # , fontsize=5

            if save:
                plt.savefig(f'pics/stats/stats_{city}_{driver_frac}_{horizon}.png')

            plt.show()

    def plot_static(self, data: pd.DataFrame) -> None:
        if not self.enriched:
            self.enrich(data)

        @interact(
            city=self._cities,
            dr=self._drs,
            driver_frac=self._fracs,
            horizon=self._horizons,
            matching_fn=self._functions,
            m=IntSlider(min=self._ms[0], max=self._ms[1]),
            capacity=IntSlider(min=self._capacities[0], max=self._capacities[1])
        )
        def f(
                city: str,
                dr: float,
                driver_frac: float,
                horizon: str,
                matching_fn: str,
                m: int,
                capacity: int
        ) -> None:
            df = (
                data[
                    (data.city == city) &
                    (data.dr == dr) &
                    (data.driver_frac == driver_frac) &
                    (data.horizon == horizon) &
                    (data.matching_fn == matching_fn) &
                    (data.m == m) &
                    (data.capacity == capacity)
                    ].copy()
            )
            frame = df[['datetime', 'num_clients_with_no_reach', 'num_captains_with_no_requests']].drop_duplicates()
            frame['datetime'] = frame['datetime'].astype(str).apply(lambda x: x.split(' ')[1])
            frame = frame.set_index('datetime').unstack().reset_index(drop=False)
            frame.columns = ['type', 'datetime', 'cnt']
            mean = (df['num_clients'] * df['mean_distance_to_client']).sum() / df['num_clients'].sum()

            plt.figure(figsize=(16, 8))
            sns.lineplot(data=frame, x='datetime', y='cnt', hue='type')
            plt.xticks(rotation=90)
            plt.show()

            print(' ')
            print('Weighted average distance to customer:')
            print(f'{round(mean, 4)} km')

    def plot_surface(self, data: pd.DataFrame, save: bool = False) -> None:
        if not self.enriched:
            self.enrich(data)

        @interact(
            city=self._cities,
            dr=self._drs,
            driver_frac=self._fracs,
            n_choice=self._nps,
            horizon=self._horizons,
            matching_fn=self._functions,
            target=self.targets
        )
        def f(
                city: str,
                dr: float,
                driver_frac: float,
                n_choice: int,
                horizon: str,
                matching_fn: str,
                target: str
        ) -> None:
            grouped = (
                data[
                    (data.city == city) &
                    (data.dr == dr) &
                    (data.driver_frac == driver_frac) &
                    (data.n_possible == n_choice) &
                    (data.horizon == horizon) &
                    (data.matching_fn == matching_fn)
                    ][['m', 'capacity', target]]
                .groupby(by=['m', 'capacity'], as_index=False)
                .sum()
            )
            g = pd.pivot_table(grouped, values=target, index='m', columns='capacity').values

            main_title = f"{target.replace('_', ' ')} (driver's fraction: {driver_frac})"

            layout = go.Layout(
                scene=dict(
                    xaxis=dict(title='capacity of captain'),
                    yaxis=dict(title='num. of captains for one bid'),
                    zaxis=dict(title='cnt')
                ),
                title=go.layout.Title(text=main_title),
                autosize=False,
                width=700,
                height=700,
                margin=dict(l=65, r=50, b=65, t=90)
            )

            fig = go.Figure(
                data=[go.Surface(z=g, x=np.arange(1, 16), y=np.arange(1, 16))],
                layout=layout
            )

            if save:
                if target == 'num_clients_with_no_reach':
                    DIR = 'pics/surface/clients'
                else:
                    DIR = 'pics/surface/drivers'

                fig.write_image(os.path.join(DIR, f"surface_{city}_{int(dr)}_{round(driver_frac, 2)}_{horizon}.png"))

            fig.show()

    def plot_line(self, data: pd.DataFrame) -> None:
        if not self.enriched:
            self.enrich(data)
            
        @interact(
            city=self._cities,
            dr=self._drs,
            driver_frac=self._fracs,
            horizon=self._horizons,
            matching_fn=self._functions,
            target=['m', 'capacity'],  # TODO: make this changeble
            non_target_variable_value=IntSlider(min=1, max=15)  # TODO: make this changeble
        )
        def f(
                city: str,
                dr: float,
                driver_frac: float,
                horizon: str,
                matching_fn: str,
                target: str,
                non_target_variable_value: int
        ) -> None:
            if target == 'm':
                anti_target = 'capacity'
            else:
                anti_target = 'm'

            frame = (
                data[
                    (data.city == city) &
                    (data.dr == dr) &
                    (data.driver_frac == driver_frac) &
                    (data.horizon == horizon) &
                    (data.matching_fn == matching_fn) &
                    (data[anti_target] == non_target_variable_value)
                    ][[target, 'num_clients_with_no_reach', 'num_captains_with_no_requests']]
                .copy()
            )

            grouped = frame.groupby(by=target, as_index=False).sum().set_index(target).unstack().reset_index(drop=False)
            grouped.columns = ['type', target, 'cnt']

            plt.figure(figsize=(16, 8))
            sns.lineplot(data=grouped, x=target, y='cnt', hue='type')
            plt.show()

    def draw_heatmap(
            self, data: pd.DataFrame, save: bool = False, filter_list: list = [], optimal: bool = False
    ) -> None:
        if not self.enriched:
            self.enrich(data)

        @interact(
            city=self._cities,
            dr=self._drs,
            driver_frac=self._fracs,
            horizon=self._horizons,
            L=IntSlider(min=1, max=5),
            target=['SR', 'Efficiency', 'frac_of_clients_with_reach', 'frac_of_captains_with_requests'],
        )
        def f(city: str, dr: float, driver_frac: float, horizon: str, L: int, target: str) -> None:
            if optimal:
                heat = (
                    data[
                        (data.city == city) &
                        (data.horizon == horizon) &
                        (data.dr == dr) &
                        (data.driver_frac == driver_frac) &
                        (data.m > 5) &
                        (data.capacity == 5) &
                        (data.n_possible == L) &
                        (~data.matching_fn.isin(filter_list))
                        ].copy()
                )

                fig_size = (5, 3)

            else:
                heat = (
                    data[
                        (data.city == city) &
                        (data.horizon == horizon) &
                        (data.dr == dr) &
                        (data.driver_frac == driver_frac) &
                        (data.m <= 10) &
                        (data.m > 1) &
                        (data.capacity <= 10) &
                        (data.capacity > 1) &
                        (data.n_possible == L) &
                        (~data.matching_fn.isin(filter_list))
                    ].copy()
                )

                if len(filter_list) != 0:
                    fig_size = (10/len(filter_list), 20)
                else:
                    fig_size = (10, 20)

            heat['matching_fn'] = (
                heat['matching_fn'].str.replace('k_hungarian_capacity_exhaust', 'k_hungarian_one_sided')
            )
            heat['matching_fn'] = (
                heat['matching_fn'].str.replace('k_hungarian_matching', 'fixed_k_hungarian')
            )
            heat['matching_fn'] = (
                heat['matching_fn'].str.replace('k_hungarian_m_capacity_exhaust', 'k_hungarian_two_sided')
            )

            sub = heat[['num_clients', 'num_captains']].drop_duplicates()
            ratio = round((sub['num_clients']/sub['num_captains']).mean(), 2)
            del sub

            heat['param'] = heat.apply(lambda row: str(dict(capacity=row['capacity'], M=row['m'])), axis=1)
            heat['SR'] = 1 - heat['num_clients_with_no_handshake_options'] / heat['num_clients']
            heat['frac_of_clients_with_reach'] = 1 - heat['num_clients_with_no_reach'] / heat['num_clients']
            heat['frac_of_captains_with_requests'] = 1 - heat['num_captains_with_no_requests'] / heat['num_captains']
            heat = heat.rename(columns={'mean_distance_to_client': 'Efficiency'})
            heat = heat.sort_values(['capacity', 'm', 'n_possible'])[['matching_fn', 'param', target]]
            heat = pd.pivot_table(heat, target, 'param', 'matching_fn', sort=False)

            if target == 'Efficiency':
                palette = sns.color_palette("Spectral", n_colors=100)
                palette.reverse()
            else:
                palette = sns.color_palette("Spectral", n_colors=100)

            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            heatmap = sns.heatmap(heat, annot=True, fmt='.3f', ax=ax, cbar=False, cmap=palette)
            heatmap.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            plt.xticks(rotation=30)
            plt.xlabel('')
            plt.title(f'{target}\n (Mean Demand/Supply Ratio: {ratio}; L={L})')

            if save:
                plt.savefig(
                    f'heatmap_{city}_{driver_frac}_{horizon}_{L}_{target}.png', bbox_inches='tight', pad_inches=0.5
                )

            plt.show()
