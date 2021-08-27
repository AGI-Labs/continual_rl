import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import collections
import copy
import cloudpickle as pickle
import matplotlib.pyplot as plt

import plotly.graph_objects as go

# see https://github.com/plotly/Kaleido/issues/101
import plotly.io as pio
pio.kaleido.scope.mathjax = None

# plotly.io.orca.config.executable = '/usr/local/bin/orca'


TASKS_ATARI = {
    "0-SpaceInvaders": dict(i=0, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[0, 50e6], [300e6, 350e6]], showlegend=False),
    "1-Krull": dict(i=1, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[50e6, 100e6], [350e6, 400e6]], showlegend=False),
    "2-BeamRider": dict(i=2, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[100e6, 150e6], [400e6, 450e6]], showlegend=True),
    "3-Hero": dict(i=3, y_range=[0, 5e4], yaxis_dtick=1e4, train_regions=[[150e6, 200e6], [450e6, 500e6]], showlegend=False),
    "4-StarGunner": dict(i=4, y_range=[0, 10e4], yaxis_dtick=2e4, train_regions=[[200e6, 250e6], [500e6, 550e6]], showlegend=False),
    "5-MsPacman": dict(i=5, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[250e6, 300e6], [550e6, 600e6]], showlegend=True),
}

MODELS_ATARI = {
    "IMPALA": dict(
        name='impala',
        runs=[f'impala{i}' for i in range(5)],
        # color='rgba(64, 132, 133, 1)',
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'ewc{i}' for i in range(5)],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "ONLINE EWC": dict(
        name='online ewc',
        runs=[f'onlineewc{i}' for i in range(5)],
        color='rgba(106, 166, 110, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=['pnc0', 'pnc1', 'pnc2', 'pnc3_last1Mlost', 'pnc4'],
        # color='rgba(152, 52, 48, 1)',
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=['clear0', 'clear1', 'clear2', 'clear5', 'clear8'],
        # color='rgba(212, 162, 217, 1)',
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
ATARI = dict(
    models=MODELS_ATARI,
    tasks=TASKS_ATARI,
    num_cycles=2,
    num_task_steps=50e6,
    grid_size=[2, 3],
    which='atari',
    rolling_mean_count=20,
    filter='ma',
    xaxis_tickvals=list(np.arange(0, 600e6 + 1, 300e6)),
)

TASKS_PROCGEN = {
    "0-Climber": dict(i=0, eval_i=1, y_range=[0., 1.], yaxis_dtick=0.25, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(0, 6 * 5, 6)]),
    "1-Dodgeball": dict(i=2, eval_i=3, y_range=[0., 2.], yaxis_dtick=0.5, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(1, 6 * 5, 6)]),
    "2-Ninja": dict(i=4, eval_i=5, y_range=[0., 4.], yaxis_dtick=1.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(2, 6 * 5, 6)]),
    "3-Starpilot": dict(i=6, eval_i=7, y_range=[0., 10.], yaxis_dtick=2.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(3, 6 * 5, 6)]),
    "4-Bigfish": dict(i=8, eval_i=9, y_range=[0., 8.], yaxis_dtick=2.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(4, 6 * 5, 6)]),
    "5-Fruitbot": dict(i=10, eval_i=11, y_range=[-4.5, 4.5], yaxis_dtick=1.5, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(5, 6 * 5, 6)]),
}
MODELS_PROCGEN = {
    "IMPALA": dict(
        name='impala',
        runs=[f'impala{i}_procgen' for i in range(5)],
        # color='rgba(64, 132, 133, 1)',
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'ewc{i}_procgen' for i in range(5)],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "ONLINE EWC": dict(
        name='online ewc',
        runs=[f'onlineewc{i}_procgen' for i in range(5)],
        color='rgba(106, 166, 110, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'pnc{i}_procgen' for i in range(5)],
        # color='rgba(152, 52, 48, 1)',
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'clear{i}_procgen' for i in range(5)],
        # color='rgba(212, 162, 217, 1)',
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
PROCGEN = dict(
    models=MODELS_PROCGEN,
    tasks=TASKS_PROCGEN,
    rolling_mean_count=20,
    filter='ma',
    num_cycles=5,
    num_task_steps=5e6,
    grid_size=[2, 3],
    which='procgen',
    xaxis_tickvals=list(np.arange(0, 150e6 + 1, 30e6)),
)


TASKS_MINIHACK = {
    "0-Room-Dark": dict(i=0, eval_i=1, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(0, 13 * 2, 13)]),
    "1-Room-Monster": dict(i=2, eval_i=3, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(1, 13 * 2, 13)]),
    "2-Room-Trap": dict(i=4, eval_i=5, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(2, 13 * 2, 13)]),
    "3-Room-Ultimate": dict(i=6, eval_i=7, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(3, 13 * 2, 13)]),
    "4-Corridor-R2": dict(i=8, eval_i=9, y_range=[-1.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(4, 13 * 2, 13)]),
    "5-Corridor-R3": dict(i=10, eval_i=11, y_range=[-1.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(5, 13 * 2, 13)]),
    "6-KeyRoom": dict(i=12, eval_i=13, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(6, 13 * 2, 13)]),
    "7-River-Narrow": dict(i=14, eval_i=15, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(7, 13 * 2, 13)]),
    "8-River-Monster": dict(i=16, eval_i=17, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(8, 13 * 2, 13)]),
    "9-River-Lava": dict(i=18, eval_i=19, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(9, 13 * 2, 13)]),
    "10-HideNSeek": dict(i=20, eval_i=21, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(10, 13 * 2, 13)]),
    "11-HideNSeek-Lava": dict(i=22, eval_i=23, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(11, 13 * 2, 13)]),
    "12-CorridorBattle": dict(i=24, eval_i=25, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(12, 13 * 2, 13)]),
}
MODELS_MINIHACK = {
    "IMPALA": dict(
        name='impala',
        runs=[f'impala{i}_minihack' for i in range(5)],
        # color='rgba(64, 132, 133, 1)',
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'clear{i}_minihack' for i in range(5)],
        # color='rgba(212, 162, 217, 1)',
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
MINIHACK = dict(
    models=MODELS_MINIHACK,
    tasks=TASKS_MINIHACK,
    rolling_mean_count=20,
    filter='ma',
    num_cycles=2,
    num_task_steps=10e6,
    which='minihack',
    xaxis_tickvals=list(np.arange(0, 260e6 + 1, 130e6)),
)


TO_PLOT = dict(
    # **ATARI,
    # **PROCGEN,
    **MINIHACK,
    tag_base='eval_reward',
    cache_dir='/Users/exing/c/',
    #
    legend_size=30,
    title_size=40,
    axis_size=20,
    axis_label_size=30,
)


def tags_read_event_file(file, tags):
    from tensorflow.python.summary.summary_iterator import summary_iterator

    tags_set = set(tags)

    event_data = collections.defaultdict(list)
    try:
        for event in summary_iterator(file):
            global_step = event.step

            for val in event.summary.value:
                k = val.tag

                if k in tags_set:
                    v = val.simple_value
                    event_data[k].append((global_step, v))
    except Exception as e:
        print(f'truncated: {file}, {e}')

    return event_data


def collate_event_data(event_data_list):
    d = collections.defaultdict(list)
    for x in event_data_list:
        for k, v in x.items():
            d[k].extend(v)

    d_sorted = {}
    for k, v in d.items():
        d_sorted[k] = list(sorted(v, key=lambda x: x[0]))

    return d_sorted


def read_experiment_data(model_v, tags):
    all_run_data = {}

    for run_id in model_v['runs']:
        # check if cached data exists
        cache_p = TO_PLOT['cache_dir'] + f"{TO_PLOT['which']}_{run_id}.pkl"
        if os.path.exists(cache_p):
            print(f'loading cached: {cache_p}')
            event_data = pickle.load(open(cache_p, 'rb'))
        else:
            # iterate thru event files
            d = []
            pattern = TO_PLOT['exp_dir'] + f'{run_id}/' + '*/*/events.out.tfevents.*'
            for file in sorted(glob(pattern)):
                print(f'reading event file: {file}')
                event_data = tags_read_event_file(file, tags)
                d.append(event_data)

            if len(d) == 0:
                raise RuntimeError(f'no event files found: {pattern}')

            event_data = collate_event_data(d)

            pickle.dump(event_data, open(cache_p, 'wb'))

        all_run_data[run_id] = event_data
    return all_run_data


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
    Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid
    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def post_processing(data, tags):
    post_processed_data = {}
    for run_id, d in data.items():
        new_d = {}
        for k in tags:
            run = d[k]

            xs = np.array([run_datum[0] for run_datum in run])
            ys = [run_datum[1] for run_datum in run]

            if TO_PLOT['filter'] == 'ma':
                rolling_accumulator = collections.deque(maxlen=TO_PLOT['rolling_mean_count'])
                for x_id, x in enumerate(xs):
                    rolling_accumulator.append(ys[x_id])
                    ys[x_id] = np.array(rolling_accumulator).mean()
            elif TO_PLOT['filter'] == 'ema':
                xs, ys, _ = one_sided_ema(np.array(xs), np.array(ys), n=50)
            elif TO_PLOT['filter'] == 'smooth':
                ys = smooth(ys, TO_PLOT['rolling_mean_count'], mode='causal')
            else:
                raise ValueError

            processed_run = list(zip(xs, ys))

            new_d[k] = processed_run

            # plt.plot(xs, ys)
            # plt.title(f'{run_id} {k}')
            # plt.show()

        post_processed_data[run_id] = new_d
    return post_processed_data


def combine_experiment_data(data, tags):
    num_runs = len(data.keys())

    d = {}
    for k in tags:
        xs = []
        ys = []
        for run_id in data.keys():
            run_data = data[run_id][k]

            xs.append(np.array([data_point[0] for data_point in run_data]))
            ys.append(np.array([data_point[1] for data_point in run_data]))

        # Get the bounds and the number of samples to take for the interpolation we're about to do
        # We don't try interpolate out of the bounds of what was collected (i.e. below an experiment's min, or above its max)
        min_x = np.array([x.min() for x in xs]).max()
        max_x = np.array(
            [x.max() for x in xs]
        ).min()  # Get the min of the maxes so we're not interpolating past the end of collected data
        num_points = (
            np.array([len(x) for x in xs]).max() * 2
        )  # Doubled from my vague signal processing recollection to capture the underlying signal (...very rough)

        # Get the regular interval we'll be interpolating to
        interpolated_xs = np.linspace(min_x, max_x, num_points)
        interpolated_ys_per_run = []

        # Interpolate each run
        for run_id, run_ys in enumerate(ys):
            run_xs = xs[run_id]
            interpolated_ys = np.interp(interpolated_xs, run_xs, run_ys)
            interpolated_ys_per_run.append(interpolated_ys)

        y_series = np.array(interpolated_ys_per_run)
        y_means = y_series.mean(0)
        y_stds = y_series.std(0) / np.sqrt(
            num_runs
        )  # Computing the standard error of the mean, since that's what we're actually interested in here.

        d[k] = (interpolated_xs, y_means, y_stds)

    return d


def create_scatters(data, model_k, model_v, dash=False, mean_showlegend=True, alpha=None):
    x, y_mean, y_std = data

    y_lower = y_mean - y_std
    y_upper = y_mean + y_std

    line_color = copy.deepcopy(model_v['color'])
    fill_color = copy.deepcopy(line_color)
    fill_color = fill_color.replace(', 1)', f", {model_v['color_alpha']})")

    if alpha is not None:
       line_color = line_color.replace(', 1)', f", {alpha})")

    upper_bound = go.Scatter(
        x=x,
        y=y_upper,
        mode='lines',
        line=dict(width=0),
        fillcolor=fill_color,
        fill='tonexty',
        name=model_k,
        showlegend=False,
    )

    line = dict(color=line_color, width=3)
    if dash:
        line['dash'] = dash

    trace = go.Scatter(
        x=x,
        y=y_mean,
        mode='lines',
        line=line,
        fillcolor=fill_color,
        fill='tonexty',
        name=model_k,
        showlegend=mean_showlegend,
    )

    lower_bound = go.Scatter(
        x=x, y=y_lower, line=dict(width=0), mode='lines', name=model_k, showlegend=False
    )

    # Trace order can be important
    # with continuous error bars
    traces = [lower_bound, trace, upper_bound]

    return traces


def plot_models(d):
    num_task_steps = TO_PLOT['num_task_steps']
    num_cycles = TO_PLOT['num_cycles']
    num_tasks = len(TO_PLOT['tasks'])
    x_range = [-10, num_task_steps * num_tasks * num_cycles]

    axis_size = TO_PLOT['axis_size']
    axis_label_size = TO_PLOT['axis_label_size']
    legend_size = TO_PLOT['legend_size']
    title_size = TO_PLOT['title_size']
    which = TO_PLOT['which']

    figures = {}

    for task_i, (task_k, task_v) in enumerate(TO_PLOT['tasks'].items()):
        fig = go.Figure()

        # min_x = 0  # Effectively defaulting to 0
        # max_x = 0

        y_range = task_v.get('y_range', None)
        train_regions = task_v.get('train_regions', None)
        # showlegend = task_v.get('showlegend', False)
        showlegend = True
        yaxis_dtick = task_v.get('yaxis_dtick', None)
        yrange_offset = task_v.get('yrange_offset', -10)

        tag = f"{TO_PLOT['tag_base']}/{task_v['i']}"
        if 'eval_i' in task_v.keys():
            eval_tag = f"{TO_PLOT['tag_base']}/{task_v['eval_i']}"
        else:
            eval_tag = None

        for model_k, model_v in TO_PLOT['models'].items():
            data = d[model_k][tag]

            _kwargs = {}
            if eval_tag is not None:
                _kwargs = dict(alpha=0.5, dash='dash', mean_showlegend=False)

            low_trace, trace, up_trace = create_scatters(data, model_k, model_v, **_kwargs)

            fig.add_trace(low_trace)
            fig.add_trace(trace)
            fig.add_trace(up_trace)

            # xs = model_data[0]
            # min_x = min(xs.min(), min_x)
            # max_x = max(xs.max(), max_x)

        if eval_tag is not None:
            for model_k, model_v in TO_PLOT['models'].items():
                data = d[model_k][eval_tag]

                low_trace, trace, up_trace = create_scatters(data, model_k, model_v)

                fig.add_trace(low_trace)
                fig.add_trace(trace)
                fig.add_trace(up_trace)

        yaxis_range = [y_range[0], y_range[1] * 1.01]

        fig.update_layout(
            yaxis=dict(
                title=dict(text='Reward', font=dict(size=axis_label_size)),
                range=yaxis_range,
                tick0=0,
                dtick=yaxis_dtick,
                # zeroline=True,  # doesn't seem to work
                tickfont=dict(size=axis_size),
                gridcolor='rgb(230,236,245)',
            ),
            xaxis=dict(
                title=dict(text='Step', font=dict(size=axis_label_size)),
                range=x_range,
                tickvals=TO_PLOT.get('xaxis_tickvals', None),
                tickfont=dict(size=axis_size),
            ),
            title=dict(text=f'\n{task_k}', font=dict(size=title_size)),
            legend=dict(font=dict(size=legend_size, color="black")),
            showlegend=showlegend,
            title_x=0.15,
            plot_bgcolor='rgb(255,255,255)',
        )

        # fig.update_layout(
        #     autosize=False,
        #     width=1500,
        #     height=500,
        # )

        if train_regions is not None:
            # TODO
            for shaded_region in train_regions:
                fig.add_shape(
                    # Rectangle reference to the axes
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=shaded_region[0],
                    y0=y_range[0],
                    x1=shaded_region[1],
                    y1=y_range[1],
                    line=dict(
                        color="rgba(150, 150, 180, .3)",
                        width=1,
                    ),
                    # fillcolor="rgba(180, 180, 180, .3)",
                    fillcolor="rgba(230, 236, 245, 0.3)"
                )

        fig.write_image(f'{which}_{task_i}.pdf')
        figures[task_i] = fig

    return figures


def figures_to_server(figures, grid_size):
    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    graph_html = []
    fig_i = 0
    for r in range(grid_size[0]):
        row_html = []
        for c in range(grid_size[1]):
            fig = figures[fig_i]
            row_html.append(
                html.Div(
                    [dcc.Graph(id=f"graph_{fig_i}", figure=fig, config={'displayModeBar': False})],
                    className="four columns"
                )
            )
            fig_i += 1

        graph_html.append(html.Div(row_html, className="row"))

    body_html = [
        html.Link(
            rel="preconnect",
            href="https://fonts.googleapis.com",
        ),
        html.Link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
        ),
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,300;1,400;1,600;1,700;1,800&display=swap",
        ),
        html.Div(graph_html),
    ]

    app.layout = html.Div(body_html)

    app.run_server(debug=False)


def visualize():
    tags = []
    for task_k, task_v in TO_PLOT['tasks'].items():
        tags.append(f"{TO_PLOT['tag_base']}/{task_v['i']}")
        if 'eval_i' in task_v.keys():
            tags.append(f"{TO_PLOT['tag_base']}/{task_v['eval_i']}")
    print(f'tags: {tags}')

    d = {}
    for model_k, model_v in TO_PLOT['models'].items():
        print(f'loading data for model: {model_k}')
        data = read_experiment_data(model_v, tags)
        data = post_processing(data, tags)
        data = combine_experiment_data(data, tags)
        d[model_k] = data

    figures = plot_models(d)

    # figures_to_server(figures, grid_size=TO_PLOT['grid_size'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='experiment dir')
    args = parser.parse_args()
    TO_PLOT['exp_dir'] = args.d
    visualize()


# python visualize.py -d /Users/exing/mnt/cl/atari_benchmark_results/