import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import collections
import copy
import cloudpickle as pickle

import plotly.graph_objects as go

# see https://github.com/plotly/Kaleido/issues/101
import plotly.io as pio
pio.kaleido.scope.mathjax = None


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
    which_exp='atari',
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
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'clear{i}_procgen' for i in range(5)],
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
    which_exp='procgen',
    xaxis_tickvals=list(np.arange(0, 150e6 + 1, 30e6)),
)


TASKS_MINIHACK = {
    "0-Room-Random": dict(i=0, eval_i=1, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(0, 15 * 2, 15)]),
    "1-Room-Dark": dict(i=2, eval_i=3, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(1, 15 * 2, 15)]),
    "2-Room-Monster": dict(i=4, eval_i=5, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(2, 15 * 2, 15)]),
    "3-Room-Trap": dict(i=6, eval_i=7, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(3, 15 * 2, 15)]),
    "4-Room-Ultimate": dict(i=8, eval_i=9, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(4, 15 * 2, 15)]),
    "5-Corridor-R2": dict(i=10, eval_i=11, y_range=[-1, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(5, 15 * 2, 15)]),
    "6-Corridor-R3": dict(i=12, eval_i=13, y_range=[-1, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(6, 15 * 2, 15)]),
    "7-KeyRoom": dict(i=14, eval_i=15, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(7, 15 * 2, 15)]),
    "8-KeyRoom-Dark": dict(i=16, eval_i=17, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(8, 15 * 2, 15)]),
    "9-River-Narrow": dict(i=18, eval_i=19, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(9, 15 * 2, 15)]),
    "10-River-Monster": dict(i=20, eval_i=21, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(10, 15 * 2, 15)]),
    "11-River-Lava": dict(i=22, eval_i=23, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(11, 15 * 2, 15)]),
    "12-HideNSeek": dict(i=24, eval_i=25, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(12, 15 * 2, 15)]),
    "13-HideNSeek-Lava": dict(i=26, eval_i=27, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(13, 15 * 2, 15)]),
    "14-CorridorBattle": dict(i=28, eval_i=29, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(14, 15 * 2, 15)]),
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
    which_exp='minihack',
    xaxis_tickvals=list(np.arange(0, 260e6 + 1, 130e6)),
    metric_scale=10,
    metric_eps=0.1
)


TASKS_CHORE_VARY_ENV = {
    "Room 402": dict(i=0, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "Room 419": dict(i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "Room 423": dict(i=2, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_VARY_ENV = {
    "CLEAR": dict(
        name='clear',
        runs=[f'vary_envs_2/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'vary_envs_2/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'vary_envs_2/{i}' for i in [9, 10, 11]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
}
CHORE_VARY_ENV = dict(
    models=MODELS_CHORE_VARY_ENV,
    tasks=TASKS_CHORE_VARY_ENV,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=2,
    num_task_steps=1e6,
    which_exp='chore_vary_env',
    clip_y_range=[-10, 12]
)



TASKS_CHORE_VARY_TASK = {
    "Hang TP": dict(i=0, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "Put TP on Counter": dict(i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "Put TP in Cabinet": dict(i=2, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_VARY_TASK = {
    "CLEAR": dict(
        name='clear',
        runs=[f'vary_tasks_2/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'vary_tasks_2/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'vary_tasks_2/{i}' for i in [6, 7, 8]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
}
CHORE_VARY_TASK = dict(
    models=MODELS_CHORE_VARY_TASK,
    tasks=TASKS_CHORE_VARY_TASK,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=2,
    num_task_steps=1e6,
    which_exp='chore_vary_task',
    clip_y_range=[-10, None]
)


TASKS_CHORE_VARY_OBJECT = {
    "Clean Fork": dict(i=0, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "Clean Knife": dict(i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "Clean Spoon": dict(i=2, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_VARY_OBJECT = {
    "CLEAR": dict(
        name='clear',
        runs=[f'vary_objects_3/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'vary_objects_3/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'vary_objects_3/{i}' for i in [6, 7, 8]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
}
CHORE_VARY_OBJECT = dict(
    models=MODELS_CHORE_VARY_OBJECT,
    tasks=TASKS_CHORE_VARY_OBJECT,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=1,
    num_task_steps=1e6,
    which_exp='chore_vary_object',
    clip_y_range=[-10, None]
)


TASKS_CHORE_MULTI_TRAJ = {
    "Room 19, Cup": dict(i=0, eval_i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "Room 13, Potato": dict(i=2, eval_i=3, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "Room 2, Lettuce": dict(i=4, eval_i=5, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_MULTI_TRAJ = {
    "CLEAR": dict(
        name='clear',
        runs=[f'multi_traj/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'multi_traj/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'multi_traj/{i}' for i in [6, 7, 8]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    )
}
CHORE_MULTI_TRAJ = dict(
    models=MODELS_CHORE_MULTI_TRAJ,
    tasks=TASKS_CHORE_MULTI_TRAJ,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=1,
    num_task_steps=1e6,
    which_exp='chore_multi_traj',
    clip_y_range=[-10, None]
)


TO_PLOT = dict(
    tag_base='eval_reward',
    cache_dir='tmp/',
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
        cache_filename = f"{TO_PLOT['which_exp']}_{run_id}.pkl".replace(os.path.sep, "-")  # The run may be a path, so de-path-ify it
        cache_p = os.path.join(TO_PLOT['cache_dir'], cache_filename)
        if os.path.exists(cache_p):
            print(f'loading cached: {cache_p}')
            event_data = pickle.load(open(cache_p, 'rb'))
        else:
            # iterate thru event files
            d = []
            pattern = os.path.join(TO_PLOT['exp_dir'], f'{run_id}', '**/events.out.tfevents.*')
            for file in sorted(glob(pattern, recursive=True)):
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

            if TO_PLOT.get("clip_y_range", None) is not None:
                clip_range = TO_PLOT["clip_y_range"]
                ys = np.array(ys).clip(min=clip_range[0], max=clip_range[1])

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
    which_exp = TO_PLOT['which_exp']

    figures = {}

    for task_i, (task_k, task_v) in enumerate(TO_PLOT['tasks'].items()):
        fig = go.Figure()

        # min_x = 0  # Effectively defaulting to 0
        # max_x = 0

        y_range = task_v.get('y_range', None)
        train_regions = task_v.get('train_regions', None)
        showlegend = True
        yaxis_dtick = task_v.get('yaxis_dtick', None)

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

        if train_regions is not None:
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
                    fillcolor="rgba(230, 236, 245, 0.3)"
                )

        fig.write_image(f'{which_exp}_{task_i}.pdf')
        figures[task_i] = fig

    return figures


def get_rewards_for_region(xs, ys, region):
        valid_x_mask_lower = xs > region[0] if region[0] is not None else True  # If we have no lower bound specified, all xs are valid
        valid_x_mask_upper = xs < region[1] if region[1] is not None else True
        valid_x_mask = valid_x_mask_lower * valid_x_mask_upper

        return ys[valid_x_mask]


def compute_forgetting_metric(task_results, task_steps, task_id, num_tasks, num_cycles):
    """
    We compute how much is forgotten of task (task_id) as each subsequent (subsequent_task_id) is learned.
    """
    total_forgetting_per_subsequent = {id: {} for id in range(num_tasks)}  # Inner dict maps cycle to total

    for run_id, task_result in enumerate(task_results):
        xs = np.array([t[0] for t in task_result])
        ys = np.array([t[1] for t in task_result])

        # Select only the rewards from the region up to and including the training of the given task
        task_rewards = get_rewards_for_region(xs, ys, [None, (task_id+1) * task_steps])
        max_task_value = task_rewards[-1]

        for cycle_id in range(num_cycles):
            for subsequent_task_id in range(num_tasks):
                # It's not really "catastrophic forgetting" if we haven't seen the task yet, so skip the early tasks
                if cycle_id == 0 and subsequent_task_id <= task_id:
                    continue

                offset = cycle_id * num_tasks
                subsequent_region = [(subsequent_task_id + offset) * task_steps,
                                     (subsequent_task_id + offset + 1) * task_steps]
                subsequent_task_rewards = get_rewards_for_region(xs, ys, subsequent_region)
                last_reward = subsequent_task_rewards[-1]
                forgetting = max_task_value - last_reward

                cycle_total = total_forgetting_per_subsequent[subsequent_task_id].get(cycle_id, 0) + forgetting
                total_forgetting_per_subsequent[subsequent_task_id][cycle_id] = cycle_total

    average_forgetting = {}
    for subsequent_id, subsequent_metrics in total_forgetting_per_subsequent.items():
        average_forgetting[subsequent_id] = {}
        for cycle_id in subsequent_metrics.keys():
            average_forgetting[subsequent_id][cycle_id] = total_forgetting_per_subsequent[subsequent_id][cycle_id]  / len(task_results)

    return average_forgetting


def compute_forward_transfer_metric(task_results, task_steps, prior_task_ids):
    """
    We compute how much is learned of task (task_id) by each previous task, before task (task_id) is learned at all.
    """
    total_transfer_per_prior = {id: 0 for id in prior_task_ids}

    for run_id, task_result in enumerate(task_results):
        xs = np.array([t[0] for t in task_result])
        ys = np.array([t[1] for t in task_result])

        # Select only the rewards from the region up to and including the training of the given task
        initial_task_value = ys[0]  # TODO: this isn't necessarily a robust average

        for prior_task_id in prior_task_ids:
            prior_region = [prior_task_id * task_steps, (prior_task_id+1) * task_steps]  # TODO: could do from the end of the task up to the subsequent one we're looking at...
            subsequent_task_rewards = get_rewards_for_region(xs, ys, prior_region)
            last_reward = subsequent_task_rewards[-1]
            transfer = last_reward - initial_task_value

            total_transfer_per_prior[prior_task_id] += transfer

    average_transfer = {}
    for prior_id in total_transfer_per_prior.keys():
        average_transfer[prior_id] = total_transfer_per_prior[prior_id] / len(task_results)

    return average_transfer


def get_metric_tags():
    """
    Get the tags to be used during computation of metrics. It is assumed that the order is consistent: i.e. tags
    A, B, C, D will be used to compute how much forgetting D causes for B and C.
    :return:
    """
    task_ids = [task["eval_i"] if "eval_i" in task else task["i"] for task in TO_PLOT["tasks"].values()]
    tags = [f"{TO_PLOT['tag_base']}/{id}" for id in task_ids]
    return tags


def compute_metrics(data):
    # Grab the tag ids we will use to evaluate the metrics: if we collected explicit eval data, use that.
    tags = get_metric_tags()
    num_tasks = len(tags)
    metrics = {}

    # For each task (labeled by a tag), grab all of the associated runs, then compute the metrics on them
    for task_id, task_tag in enumerate(tags):
        per_task_data = []
        for run_data in data.values():
            per_task_data.append(run_data[task_tag])

        # Compute the amount this task was forgotten by subsequent tasks
        # Forgetting will map task to a dictionary (cycle_id: amount of forgetting)
        forgetting = compute_forgetting_metric(per_task_data, TO_PLOT["num_task_steps"], task_id, num_tasks, TO_PLOT["num_cycles"])

        prior_task_ids = list(range(len(tags)))[:task_id]
        transfer = compute_forward_transfer_metric(per_task_data, TO_PLOT["num_task_steps"], prior_task_ids)

        metrics[task_tag] = {"forgetting": forgetting, "transfer": transfer}

    return metrics


def generate_metric_table(metric_table, negative_as_green, table_caption):
    def style_forgetting_table(v):
        color = "white"
        eps = TO_PLOT.get("metric_eps", 0)
        v_near_0 = v is None or np.isnan(v) or not (v > eps or v < -eps)  # Nan because pandas processes Nones inconsistently, it seems
        if not v_near_0:
            if (not negative_as_green and v > 0) or (negative_as_green and v < 0):
                color = "green"
            else:
                color = "red"
        return f"cellcolor:{{{color}!20}}"  # Exclamation point is a mixin - says how much of the given color to use (mixed in with white)

    tasks = list(TO_PLOT["tasks"].keys())

    # Styling for Latex isn't quite the same as other formats, see: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    data_frame = pd.DataFrame(metric_table)
    data_frame = data_frame.rename(columns=lambda x: f"{tasks[x % len(tasks)]} (C{x//len(tasks)})")  # Name the columns: "Task Name (C cycle_id)"
    data_frame = data_frame.rename(index=lambda x: f"{tasks[x]}")  # Name the rows: "Task Name"

    data_style = data_frame.style.format(precision=1, na_rep="--")
    data_style = data_style.applymap(style_forgetting_table)
    data_style = data_style.set_table_styles([
        {'selector': 'toprule', 'props': ':hline;'},
        {'selector': 'bottomrule', 'props': ':hline;'},
    ], overwrite=False)

    # Column styles should be |l|llll| The first isolates the row names
    column_style = ''.join(['l' for _ in range(len(data_style.columns))])
    latex_metrics = data_style.to_latex(column_format=f"|l|{column_style}|")  # Requires pandas > 1.3.0 (conda install pandas==1.3.0)

    # TODO: not putting the hline under the column names because I'm not sure how at the moment, so doing that manually

    return f"\subfloat[{table_caption}]{{ \n {latex_metrics}}}"


def plot_metrics(metrics):
    tags = get_metric_tags()
    num_tasks = len(tags)
    num_cycles = TO_PLOT["num_cycles"]
    metric_scale = TO_PLOT.get("metric_scale", 1)

    for model_name, model_metrics in metrics.items():
        # Pre-allocate our tables
        forgetting_table = [[None for _ in range(num_tasks * num_cycles)] for _ in range(num_tasks)]
        transfer_table = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]  # Zero-shot transfer, so we don't plot all the cycles

        for task_id, tag in enumerate(tags):
            task_data = model_metrics[tag]

            # Fill in forgetting data. "Impactor" means the task that is causing the change in the current task (subsequent task for forgetting)
            forgetting_data = task_data["forgetting"]
            for impactor_id in range(num_tasks):
                impact_data = forgetting_data.get(impactor_id, {})

                for cycle_id in range(num_cycles):
                    impact_cycle_data = impact_data.get(cycle_id, None)
                    forgetting_table[task_id][cycle_id * num_tasks + impactor_id] = impact_cycle_data * metric_scale if impact_cycle_data is not None else None

            # Fill in the transfer data
            transfer_data = task_data["transfer"]
            for impactor_id in range(num_tasks):
                impact_data = transfer_data.get(impactor_id, None)
                transfer_table[task_id][impactor_id] = impact_data * metric_scale if impact_data is not None else None

        latex_forgetting_metrics = generate_metric_table(forgetting_table, negative_as_green=True,
                                                         table_caption=f"{model_name}")
        latex_transfer_metrics = generate_metric_table(transfer_table, negative_as_green=False,
                                                         table_caption=f"{model_name}")
        print(f"{model_name} forgetting latex: \n\n{latex_forgetting_metrics}\n\n")
        print(f"{model_name} transfer latex: \n\n{latex_transfer_metrics}\n\n")


def visualize():
    tags = []
    for task_k, task_v in TO_PLOT['tasks'].items():
        tags.append(f"{TO_PLOT['tag_base']}/{task_v['i']}")
        if 'eval_i' in task_v.keys():
            tags.append(f"{TO_PLOT['tag_base']}/{task_v['eval_i']}")
    print(f'tags: {tags}')

    d = {}
    all_metrics = {}
    for model_k, model_v in TO_PLOT['models'].items():
        print(f'loading data for model: {model_k}')
        data = read_experiment_data(model_v, tags)
        data = post_processing(data, tags)

        # Compute the metrics after we've smoothed (so our values are more representative) but before we interpolate
        # to combine the runs together
        metrics = compute_metrics(data)
        all_metrics[model_k] = metrics

        data = combine_experiment_data(data, tags)
        d[model_k] = data

    plot_models(d)
    plot_metrics(all_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='experiment dir')
    args = parser.parse_args()
    TO_PLOT['exp_dir'] = args.d

    # exp_data = ATARI
    # exp_data = PROCGEN
    # exp_data = MINIHACK
    # exp_data = CHORE_VARY_ENV
    # exp_data = CHORE_VARY_TASK
    # exp_data = CHORE_VARY_OBJECT
    exp_data = CHORE_MULTI_TRAJ
    TO_PLOT.update(**exp_data)

    visualize()
