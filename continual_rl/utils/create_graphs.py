import os
import pickle
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework.errors import DataLossError
import math
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.signal
import copy
import time
from collections import deque


class EventsResultsAggregator(object):
    """
    The purpose of this class is to read in tensorboard event files and create plots with mean and error bars indicated.
    """

    OUTPUT_DIR = "tmp/event_results"

    COLORS = [
              ('rgba(227, 26, 28, .2)', dict(color='rgba(227, 26, 28, 1)')),
              ('rgba(255, 127, 0, .2)', dict(color='rgba(255, 127, 0, 1)')),
              #('rgba(106, 61, 154, .2)', dict(color='rgba(106, 61, 154, 1)')),
              ('rgba(31, 120, 180, .2)', dict(color='rgba(31, 120, 180, 1)')),
              ('rgba(51, 160, 44, .2)', dict(color='rgba(51, 160, 44, 1)')),
              #('rgba(255, 255, 153, .2)', dict(color='rgba(255, 255, 153, 1)')),
              ('rgba(177, 89, 40, .2)', dict(color='rgba(177, 89, 40, 1)')),
              ('rgba(166, 206, 227, .2)', dict(color='rgba(166, 206, 227, 1)')),
              ('rgba(251, 154, 153, .2)', dict(color='rgba(251, 154, 153, 1)',)),
              ('rgba(253, 191, 111, .2)', dict(color='rgba(253, 191, 111, 1)')),
              ('rgba(202, 178, 214, .2)', dict(color='rgba(202, 178, 214, 1)')),
              ('rgba(168, 213, 128, .2)', dict(color='rgba(168, 213, 128, 1)')),
              ('rgba(178, 223, 138, .2)', dict(color='rgba(178, 223, 138, 1)')),
    ]

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self._output_dir = os.path.abspath(os.path.join(current_dir, self.OUTPUT_DIR))

        try:
            os.makedirs(self._output_dir)
        except FileExistsError:
            pass

    def _load_event_data(self, event_file_path, tag):
        event_data = []

        try:
            for event in summary_iterator(event_file_path):
                global_step = event.step

                for val in event.summary.value:
                    if val.tag == tag:
                        value = val.simple_value
                        event_data.append((global_step, value))
        except DataLossError:
            print("Event file was corrupted (truncated), ending parsing.")
            pass

        return event_data

    def _read_event_file(self, event_file_path, tag):
        """
        Reads in event files, grabs the desired tag, and returns a list of (global step, value).
        Reading the event data is slow, so we'll try load it from the pickle file path if it exists. If it doesn't, we'll create it.
        """
        # For clarity of reading
        split_tag = tag.split("/")
        joined_tag = "_".join(split_tag)

        # Form a unique string across experiment, run, timestamp. Assumes the events were generated from config files
        run_id = os.path.split(event_file_path)[0].split("/")[-1]
        experiment_id = os.path.split(event_file_path)[0].split("/")[-2]
        event_id = os.path.split(event_file_path)[1]
        pickle_file_name = "{}_{}_{}_{}.pickle".format(experiment_id, run_id, event_id, joined_tag)
        pickle_path = os.path.join(self._output_dir, pickle_file_name)

        #print("Attempting to use pickle: {}".format(pickle_file_name))

        try:
            with open(pickle_path, 'rb') as pickled_file:
                event_data = pickle.load(pickled_file)
        except FileNotFoundError:
            event_data = self._load_event_data(event_file_path, tag)

            with open(pickle_path, 'wb') as pickled_file:
                pickle.dump(event_data, pickled_file)

        return event_data

    def read_experiment_data(self, experiment_folder, run_ids, task_id, tag_base):
        """
        Each experiment is composed of a number of identical runs. Pull them all at once. We assume all runs have the same agent_id.
        In the experiment_folder we'll open the runs indicated by run_ids, and load the tag for the given agent_ids.
        """
        collected_run_data = []

        for run_id in run_ids:
            print("Loading {} from {}".format(run_id, experiment_folder))

            full_run_path = os.path.join(experiment_folder, str(run_id))
            event_file = None

            full_tag = "{}/{}".format(tag_base, task_id)
            single_run_data = []

            for path, dirs, files in os.walk(full_run_path):
                for file in files:
                    if "events" in file:  # TODO: do I need to assume these are "in order"?
                        #assert event_file is None, "Multiple events found unexpectedly."
                        event_file = os.path.join(path, file)
                        run_data = self._read_event_file(event_file, full_tag)
                        single_run_data.extend(run_data)

            assert event_file is not None, "No event file found when at least one was expected."
            #run_data = self._read_event_file(event_file, full_tag)
            collected_run_data.append(single_run_data)

        return collected_run_data

    def combine_experiment_data(self, collected_run_data):
        """
        Each run is a list of (step, value) tuples.
        For now we assume that each list is already aligned in step.
        """
        num_runs = len(collected_run_data)
        xs = [np.array([data_point[0] for data_point in run_data]) for run_data in collected_run_data]
        ys = [np.array([data_point[1] for data_point in run_data]) for run_data in collected_run_data]

        # Get the bounds and the number of samples to take for the interpolation we're about to do
        # We don't try interpolate out of the bounds of what was collected (i.e. below an experiment's min, or above its max)
        min_x = np.array([x.min() for x in xs]).max()
        max_x = np.array([x.max() for x in xs]).min()  # Get the min of the maxes so we're not interpolating past the end of collected data
        num_points = np.array([len(x) for x in xs]).max() * 2 # Doubled from my vague signal processing recollection to capture the underlying signal (...very rough)

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
        y_stds = y_series.std(0)/math.sqrt(num_runs)  # Computing the standard error of the mean, since that's what we're actually interested in here.

        return interpolated_xs, y_means, y_stds

    def _create_scatters(self, x, y_mean, y_std, line_label, fill_color, line_color, is_dashed=False):
        y_lower = y_mean - y_std
        y_upper = y_mean + y_std

        upper_bound = go.Scatter(
            x=x,
            y=y_upper,
            mode='lines',
            line=dict(width=0),
            fillcolor=fill_color,
            fill='tonexty',
            name=line_label,
            showlegend=False)

        line_color = copy.deepcopy(line_color)
        if is_dashed:
            line_color['dash'] = 'dash'

        trace = go.Scatter(
            x=x,
            y=y_mean,
            mode='lines',
            line=dict(color=line_color['color'], width=3),
            fillcolor=fill_color,
            fill='tonexty',
            name=line_label)

        lower_bound = go.Scatter(
            x=x,
            y=y_lower,
            line=dict(width=0),
            mode='lines',
            name=line_label,
            showlegend=False)

        # Trace order can be important
        # with continuous error bars
        data = [lower_bound, trace, upper_bound]

        return data

    def plot_multiple_lines_on_graph(self, experiment_data, title, x_offset, y_range, x_range=None, shaded_regions=None,
                                     filename=None, legend_size=40, title_size=50, axis_size=30, axis_label_size=40,
                                     y_axis_log=False):
        traces = []
        min_x = 0  # Effectively defaulting to 0
        max_x = 0

        for run_id, run_data in enumerate(experiment_data):
            xs, y_means, y_stds, line_label, line_is_dashed = run_data

            color = self.COLORS[run_id]
            traces.extend(self._create_scatters(xs, y_means, y_stds, line_label,
                                                color[0], color[1], is_dashed=line_is_dashed))
            if xs.min() < min_x:
                min_x = xs.min()
            if xs.max() > max_x:
                max_x = xs.max()

        x_range = x_range or [min_x-x_offset, max_x+x_offset]

        layout = go.Layout(
            yaxis=dict(title=dict(text='Reward', font=dict(size=axis_label_size)), range=y_range, tickfont=dict(size=axis_size)),
            xaxis=dict(title=dict(text='Step', font=dict(size=axis_label_size)), range=x_range, tickfont=dict(size=axis_size)),
            title=dict(text=title, font=dict(size=title_size)),
            legend=dict(font=dict(size=legend_size, color="black")))

        fig = go.Figure(data=traces, layout=layout)
        fig.update_layout(title_x=0.5)

        if y_axis_log:
            fig.update_yaxes(type="log")

        for shaded_region in shaded_regions:
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
                fillcolor="rgba(180, 180, 180, .3)",
            )

        plotly.offline.plot(fig, filename="tmp/graph_{}.html".format(time.time()))
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            try:
                fig.write_image(filename)
            except ValueError as e:
                print("May need to install poppler: conda install poppler")
                raise e

    def post_processing(self, experiment_data, rolling_mean_count, allowed_y_range=None):
        """
        The data is now all eval, so just smooth all of it.
        """
        post_processed_data = []

        for run in experiment_data:
            xs = np.array([run_datum[0] for run_datum in run])
            ys = [run_datum[1] for run_datum in run]
            rolling_accumulator = deque(maxlen=rolling_mean_count)

            for x_id, x in enumerate(xs):
                rolling_accumulator.append(ys[x_id])
                rolling_accumulator_comb = np.array(rolling_accumulator)

                if allowed_y_range is not None:
                    rolling_accumulator_comb = rolling_accumulator_comb.clip(min=allowed_y_range[0], max=allowed_y_range[1])

                ys[x_id] = rolling_accumulator_comb.mean()

            processed_run = list(zip(xs, ys))
            post_processed_data.append(processed_run)

        return post_processed_data


def create_graph_mnist():

    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_clear"
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_sane"
    ewc_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_ewc"
    pnc_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_pnc"

    # The second param is the range of "eval" points. See post_processing for more info
    all_experiment_data = [(digit_id, [[None, 300000 * (digit_id)],
                                       [300000 * (digit_id+1), None]]) for digit_id in range(10)]

    for digit_id, eval_ranges in all_experiment_data:
        graph = []
        #graph.append((aggregator.post_processing(aggregator.read_experiment_data(sane_folder, list(range(0, 5)), task_id=digit_id*2, tag_base="eval_reward"),
        #                                         rolling_mean_count=5), "SANE, 4% random", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(15, 20)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(pnc_folder, list(range(0, 5)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "PnC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(ewc_folder, list(range(0, 5)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(sane_folder, list(range(15, 20)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "SANE", False))  # 8% random won

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"MNIST: {digit_id}", x_offset=10, y_range=[-1, 101],
                                                shaded_regions=[[300000*digit_id, 300000*(digit_id+1)]], filename=f"tmp/icml/mnist/mnist{digit_id}.eps")


def process_progress_and_compress(all_pnc_data, regions_to_remove):
    post_processed_data = []
    for experiment_data in all_pnc_data:
        post_processed_experiment = []
        for data_point in experiment_data:
            include_data_point = True
            data_point_delta = 0

            # Check each region to see if the point is valid. If it is valid and higher than the region max,
            # add the delta to the adjustment. (Basically we're just pretending these regions didn't exist, so reduce
            # the timesteps accordingly)
            for region in regions_to_remove:
                region_lower, region_higher = region
                region_delta = region_higher - region_lower
                if data_point[0] > region_lower and data_point[0] < region_higher:
                    include_data_point = False
                elif data_point[0] >= region_higher:
                    data_point_delta += region_delta

            if include_data_point:
                post_processed_experiment.append((data_point[0] - data_point_delta, data_point[1]))

        post_processed_data.append(post_processed_experiment)
    return post_processed_data


def create_graph_split_mnist():

    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/results/rebuttal_clear"
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/results/rebuttal_sane"
    ewc_folder = "/Volumes/external/Results/PatternBuffer/sane/results/rebuttal_ewc"
    pnc_folder = "/Volumes/external/Results/PatternBuffer/sane/results/rebuttal_pnc"

    # The second param is the range of "eval" points. See post_processing for more info
    all_experiment_data = [(digit_id, [[None, 300000 * (digit_id)],
                                       [300000 * (digit_id+1), None]]) for digit_id in range(5)]

    for digit_id, eval_ranges in all_experiment_data:
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(0, 3)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR", False))
        graph.append((aggregator.post_processing(process_progress_and_compress(aggregator.read_experiment_data(pnc_folder, [10, 11, 12], task_id=digit_id*2, tag_base="eval_reward"),
                                                                               [[600000*i, 600000*(i+0.5)] for i in range(5)]),
                                                 rolling_mean_count=5), "PnC", False))  # The 2x situation is irritating
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(ewc_folder, list(range(0, 3)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(sane_folder, list(range(0, 3)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "SANE", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"Split MNIST: {digit_id*2}/{digit_id*2+1}", x_offset=10, y_range=[-1, 101],
                                                shaded_regions=[[300000*digit_id, 300000*(digit_id+1)]],
                                                filename=f"tmp/rebuttal/mnist/split_mnist{digit_id}.eps", title_size=30)


def create_graph_procgen():

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/PatternBuffer/sane/results/procgen"

    tasks = [(0, f"Dodgeball", [[200000*i, 200000*(i+1)] for i in range(0, 6, 2)]),
             (1, f"Miner", [[200000*i, 200000*(i+1)] for i in range(1, 7, 2)])]

    for task_info in tasks:
        task_id, task_name, shaded_regions = task_info
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [78, 79, 80], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR", False))
        graph.append((aggregator.post_processing(process_progress_and_compress(aggregator.read_experiment_data(procgen_folder, [98, 110, 111], task_id=task_id, tag_base="eval_reward"),
                                                                               [[400000*i, 400000*(i+0.5)] for i in range(7)]),
                                                 rolling_mean_count=5), "PnC", False))  # The 2x situation is irritating
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [107, 108, 109], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [81, 84, 87], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "SANE", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=[-1, 3],
                                                shaded_regions=shaded_regions, filename=f"tmp/rebuttal/procgen/{task_name}.eps",
                                                title_size=30, x_range=[-10, 800000])


def create_graph_mnist_clear_comp():

    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_clear"

    # The second param is the range of "eval" points. See post_processing for more info
    all_experiment_data = [(digit_id, [[None, 300000 * (digit_id)],
                                       [300000 * (digit_id+1), None]]) for digit_id in range(10)]

    for digit_id, eval_ranges in all_experiment_data:
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(15, 20)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR 25%", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(20, 25)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR 50%", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(25, 30)), task_id=digit_id*2, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR 75%", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"MNIST: {digit_id}", x_offset=10, y_range=[-1, 101],
                                                shaded_regions=[[300000*digit_id, 300000*(digit_id+1)]], filename=f"tmp/icml/mnist/mnist_clear{digit_id}.eps",
                                                legend_size=24, title_size=32, x_range=[-10, 1.1e6], axis_label_size=24, axis_size=20)


def compute_mnist_averages():

    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_clear"
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_sane"
    ewc_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_ewc"
    pnc_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_pnc"

    for digit_id in range(10):
        collected_data = []

        # The "train" in the tag is misleading. Really the tensorboard tags should be "continual_eval" and "primary task" or something
        # these tasks are eval tasks
        collected_data.append((aggregator.read_experiment_data(sane_folder, list(range(15, 20)), task_id=digit_id*2+1, tag_base="train_reward"), "SANE", False))
        collected_data.append((aggregator.read_experiment_data(clear_folder, list(range(15, 20)), task_id=digit_id*2+1, tag_base="train_reward"), "CLEAR", False))
        collected_data.append((aggregator.read_experiment_data(ewc_folder, list(range(0, 5)), task_id=digit_id*2+1, tag_base="train_reward"), "EWC", False))
        collected_data.append((aggregator.read_experiment_data(pnc_folder, list(range(0, 5)), task_id=digit_id*2+1, tag_base="train_reward"), "PnC", False))

        print(f"Cumulative to {digit_id}")
        for entry in collected_data:
            scores = []
            for run in entry[0]:
                scores.append(run[0][1])
            print(f"{entry[1]}: {np.array(scores).mean()}")

        print("-------------------")


def create_graph_minigrid_oddoneout():
    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_clear"
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_sane"
    ewc_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_ewc"
    pnc_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_pnc"
    tasks = [(0, f"OOO: Blue", [[600000*i, 600000*(i+1)] for i in range(0, 10, 2)]),
             (1, f"OOO: Yellow", [[600000*i, 600000*(i+1)] for i in range(1, 11, 2)])]
    
    for task_data in tasks:
        task_id, task_title, train_regions = task_data

        graph = []

        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(5, 10)), task_id=task_id, tag_base="eval_reward"),
                      rolling_mean_count=5), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(pnc_folder, list(range(10, 15)), task_id=task_id, tag_base="eval_reward"),
                      rolling_mean_count=5), "PnC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(ewc_folder, list(range(5, 10)), task_id=task_id, tag_base="eval_reward"),
                      rolling_mean_count=5), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(sane_folder, list(range(5, 10)), task_id=task_id, tag_base="eval_reward"),
                      rolling_mean_count=5), "SANE", False))


        # Opted to not use these, as they are both worse
        #graph.append((aggregator.post_processing(aggregator.read_experiment_data(sane_folder, list(range(10,15)), task_id=task_id, tag_base="eval_reward"),
        #              rolling_mean_count=5), "SANE [12, 12]: 639k", False))
        #graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(10,15)), task_id=task_id, tag_base="eval_reward"),
        #              rolling_mean_count=5), "CLEAR 0.33: 639k", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=[-0.1, 1.1], x_range=[-10, 6.1e6],
                                                shaded_regions=train_regions, filename=f"tmp/icml/ooo/ooo{task_id}.eps",
                                                legend_size=24, title_size=32)


def create_graph_minigrid_oddoneout_sane_buffer_ablation():
    aggregator = EventsResultsAggregator()
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_sane"
    tasks = [(0, f"OOO: Blue", [[600000*i, 600000*(i+1)] for i in range(0, 10, 2)]),
             (1, f"OOO: Yellow", [[600000*i, 600000*(i+1)] for i in range(1, 11, 2)])]

    for task_data in tasks:
        task_id, task_title, train_regions = task_data

        graph = []

        # Removing the first entry(ies) to keep everything at 3 seeds. (Note that in 4096, run 13 died early)
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, list(range(7, 10)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE 6144", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, [11, 12, 14], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE 4096", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, list(range(20, 23)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE 2048", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=[-0.1, 1.1],
                                                x_range=[-10, 6.1e6],
                                                shaded_regions=train_regions, filename=f"tmp/icml/buff_ablation/ooo{task_id}.eps",
                                                legend_size=24, title_size=32)


def create_graph_minigrid_oddoneout_sane_node_count_ablation():
    aggregator = EventsResultsAggregator()
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_sane"
    tasks = [(0, f"OOO: Blue", [[600000*i, 600000*(i+1)] for i in range(0, 10, 2)]),
             (1, f"OOO: Yellow", [[600000*i, 600000*(i+1)] for i in range(1, 11, 2)])]

    for task_data in tasks:
        task_id, task_title, train_regions = task_data

        graph = []

        # 13 died and I did not notice. Removing the first for size-parity with the last two
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, [11, 12, 14], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE [12, 12]", False))  # 4096 per version, for consistency
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, list(range(23, 26)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE [12, 6]", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, list(range(26, 29)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE [6, 12]", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=[-0.1, 1.1],
                                                x_range=[-10, 6.1e6],
                                                shaded_regions=train_regions,
                                                filename=f"tmp/icml/node_ablation/ooo{task_id}.eps",
                                                legend_size=24, title_size=32)


def create_graph_mnist_sane_alpha_ablation():
    aggregator = EventsResultsAggregator()
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_sane"
    sane_folder_old = "/Volumes/external/Results/PatternBuffer/sane/results/sane_mnist"
    all_experiment_data = [(digit_id, [[None, 300000 * (digit_id)],
                                       [300000 * (digit_id+1), None]]) for digit_id in range(10)]

    for digit_id, eval_ranges in all_experiment_data:
        graph = []

        # Taking only the first 3 from this one for parity
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, list(range(15, 18)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE α=1", False))  # 4096 per version, for consistency
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder_old, list(range(7, 10)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE α=0.5", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder_old, list(range(4, 7)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE α=0.25", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"MNIST: {digit_id}", x_offset=10, y_range=[-1, 101],
                                                shaded_regions=[[300000*digit_id, 300000*(digit_id+1)]], filename=f"tmp/icml/alpha_ablation/mnist{digit_id}.eps",
                                                legend_size=24, title_size=32, x_range=[-10, 3.1e6], axis_label_size=24, axis_size=20)


def create_graph_mnist_clear_collect_freq():
    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/icml/icml_clear_compute_compare"
    all_experiment_data = [(digit_id, [[None, 300000 * (digit_id)],
                                       [300000 * (digit_id+1), None]]) for digit_id in range(10)]

    for digit_id, eval_ranges in all_experiment_data:
        graph = []

        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(0, 3)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR, 160", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(3, 6)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR, 400", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(6, 9)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR, 800", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(9, 12)), task_id=digit_id*2, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR, 1024", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"MNIST: {digit_id}", x_offset=10, y_range=[-1, 101],
                                                shaded_regions=[[300000*digit_id, 300000*(digit_id+1)]], filename=f"tmp/icml/clear_collect/mnist{digit_id}.eps",
                                                legend_size=24, title_size=32, x_range=[-10, 3.1e6], axis_label_size=24, axis_size=20)



def create_graph_minigrid_oddoneout_obst_clear_comp():  # Not included in the paper because the experiment is 3-env (OOO1, OOO2, dynamic obstacles)
    aggregator = EventsResultsAggregator()
    clear_folder = "/Volumes/external/Results/PatternBuffer/sane/results/minigrid_validation_3"
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/results/sane_validation_3"
    tasks = [(0, f"Minigrid: Odd One Out Blue", [[600000*i, 600000*(i+1)] for i in range(0, 10, 2)]),
             (1, f"Minigrid: Odd One Out Yellow", [[600000*i, 600000*(i+1)] for i in range(1, 11, 2)])]

    for task_data in tasks:
        task_id, task_title, train_regions = task_data

        graph = []

        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(sane_folder, list(range(28, 32)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "SANE [12, 12], 4/2/1 eval", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(35, 40)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR 0.33 eval", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(40, 43)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR 0.5 eval", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(clear_folder, list(range(43, 46)), task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=5), "CLEAR 1.0 eval", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=[-1.1, 1.1],
                                                x_range=[-10, 1.8e6],
                                                shaded_regions=train_regions)


def create_graph_atari():
    # The specs for this experiment have been moved into icml_atari_baselines, exp 0
    aggregator = EventsResultsAggregator()
    atari_pnc_folder = "/Volumes/external/Results/PatternBuffer/sane/results/atari_pnc"
    atari_ewc_folder = "/Volumes/external/Results/PatternBuffer/sane/results/atari_ewc"
    num_task_steps = 5e6
    tasks = [(0, f"Space Invaders", [[num_task_steps*i, num_task_steps*(i+1)] for i in range(0, 15, 3)], [-10, 700]),
             (1, f"Beam Rider", [[num_task_steps * i, num_task_steps * (i + 1)] for i in range(1, 15, 3)], [-25, 5000]),
             (2, f"Ms Pacman", [[num_task_steps*i, num_task_steps*(i+1)] for i in range(2, 15, 3)], [0, 1300])]

    for task_data in tasks:
        task_id, task_title, train_regions, y_range = task_data
        graph = []

        graph.append((aggregator.post_processing(
            process_progress_and_compress(aggregator.read_experiment_data(atari_pnc_folder, [0], task_id=task_id, tag_base="eval_reward"), [[2*num_task_steps*i, 2*num_task_steps*(i+0.5)] for i in range(16)]),
            rolling_mean_count=20), "PnC 25", False))
        graph.append((aggregator.post_processing(
            process_progress_and_compress(aggregator.read_experiment_data(atari_pnc_folder, [1], task_id=task_id, tag_base="eval_reward"), [[2*num_task_steps*i, 2*num_task_steps*(i+0.5)] for i in range(16)]),
            rolling_mean_count=20), "PnC 75", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(atari_ewc_folder, [0], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=20), "EWC 500", False))
        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(atari_ewc_folder, [1], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=20), "EWC 1000", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=y_range,
                                                x_range=[-10, 76e6],
                                                shaded_regions=train_regions,
                                                filename=f"tmp/post_icml/atari{task_id}.eps",
                                                legend_size=30, title_size=40, axis_size=20, axis_label_size=30)


def create_graph_atari_clear():
    # The specs for this experiment have been moved into icml_atari_baselines, exp 0
    aggregator = EventsResultsAggregator()
    atari_folder = "/Volumes/external/Results/PatternBuffer/sane/results/atari_cycle_validation_2"
    tasks = [(0, f"Space Invaders", [[5e6*i, 5e6*(i+1)] for i in range(0, 15, 3)], [-10, 2.4e3]),
             (1, f"Pong", [[5e6 * i, 5e6 * (i + 1)] for i in range(1, 15, 3)], [-25, 25]),
             (2, f"Qbert", [[5e6*i, 5e6*(i+1)] for i in range(2, 15, 3)], [0, 1.65e4])]

    for task_data in tasks:
        task_id, task_title, train_regions, y_range = task_data
        graph = []

        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(atari_folder, [2], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=20), "CLEAR", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=y_range,
                                                x_range=[-10, 76e6],
                                                shaded_regions=train_regions,
                                                filename=f"tmp/icml/clear_atari/atari{task_id}.eps",
                                                legend_size=30, title_size=40, axis_size=20, axis_label_size=30)


def create_graph_atari_ewc():
    # The specs for this experiment have been moved into icml_atari_baselines, exp 1
    aggregator = EventsResultsAggregator()
    atari_folder = "/Volumes/external/Results/PatternBuffer/sane/results/atari_cycle_validation_2"
    tasks = [(0, f"Space Invaders", [[5e6*i, 5e6*(i+1)] for i in range(0, 15, 3)], [-10, 550]),
             (1, f"Pong", [[5e6 * i, 5e6 * (i + 1)] for i in range(1, 15, 3)], [-21.5, -19.2]),
             (2, f"Qbert", [[5e6*i, 5e6*(i+1)] for i in range(2, 15, 3)], [-10, 1e3])]

    for task_data in tasks:
        task_id, task_title, train_regions, y_range = task_data
        graph = []

        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(atari_folder, [20], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=20), "EWC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=y_range,
                                                x_range=[-10, 76e6],
                                                shaded_regions=train_regions,
                                                filename=f"tmp/icml/ewc_atari/atari{task_id}.eps")


def create_graph_atari_pnc():
    aggregator = EventsResultsAggregator()
    atari_folder = "/Volumes/external/Results/PatternBuffer/sane/results/atari_cycle_validation_2"
    tasks = [(0, f"Space Invaders", [[5e6*i, 5e6*(i+1)] for i in range(0, 15, 3)], [-10, 550]),
             (1, f"Pong", [[5e6 * i, 5e6 * (i + 1)] for i in range(1, 15, 3)], [-21.5, -19.2]),
             (2, f"Qbert", [[5e6*i, 5e6*(i+1)] for i in range(2, 15, 3)], [-10, 1e3])]

    for task_data in tasks:
        task_id, task_title, train_regions, y_range = task_data
        graph = []

        graph.append((aggregator.post_processing(
            aggregator.read_experiment_data(atari_folder, [13], task_id=task_id, tag_base="eval_reward"),
            rolling_mean_count=20), "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, task_title, x_offset=10, y_range=y_range,
                                                x_range=[-10, 76e6],
                                                shaded_regions=train_regions,
                                                filename=f"tmp/icml/pnc_atari/atari{task_id}.eps")


def create_graph_thor_sequential_task():

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/PatternBuffer/sane/results/thor_benchmarks_4"

    tasks = [(0, f"Put away Pot", [[5e6*i, 5e6*(i+1)] for i in range(0, 12, 6)]),
             (1, f"Put away Bowl", [[5e6*i, 5e6*(i+1)] for i in range(1, 12, 6)]),
             (2, f"Put away Mug", [[5e6*i, 5e6*(i+1)] for i in range(2, 12, 6)]),
             (3, f"Slice Bread", [[5e6*i, 5e6*(i+1)] for i in range(3, 12, 6)]),
             (4, f"Slice Lettuce", [[5e6*i, 5e6*(i+1)] for i in range(4, 12, 6)]),
             (5, f"Slice Apple", [[5e6*i, 5e6*(i+1)] for i in range(5, 12, 6)])]

    for task_info in tasks:
        task_id, task_name, shaded_regions = task_info
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [0, 1, 2], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [3, 4, 5], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [6, 7, 8], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=[-1, 20],
                                                shaded_regions=shaded_regions, filename=f"tmp/rebuttal/procgen/{task_name}.eps",
                                                title_size=30, x_range=[-10, 35e6])


def create_graph_thor_sequential_environment():

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/PatternBuffer/sane/results/thor_benchmarks_env_3"

    tasks = [(0, f"FloorPlan1", [[5e6*i, 5e6*(i+1)] for i in range(0, 12, 6)]),
             (1, f"FloorPlan2", [[5e6*i, 5e6*(i+1)] for i in range(1, 12, 6)]),
             (2, f"FloorPlan3", [[5e6*i, 5e6*(i+1)] for i in range(2, 12, 6)]),
             (3, f"FloorPlan4", [[5e6*i, 5e6*(i+1)] for i in range(3, 12, 6)]),
             (4, f"FloorPlan5", [[5e6*i, 5e6*(i+1)] for i in range(4, 12, 6)]),
             (5, f"FloorPlan7", [[5e6*i, 5e6*(i+1)] for i in range(5, 12, 6)])]

    for task_info in tasks:
        task_id, task_name, shaded_regions = task_info
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [0, 1, 2], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [3, 4, 5], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [6, 7, 8], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5), "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=[-1, 20],
                                                shaded_regions=shaded_regions, filename=f"tmp/rebuttal/procgen/{task_name}.eps",
                                                title_size=30, x_range=[-10, 35e6])


def compute_num_hypotheses():
    aggregator = EventsResultsAggregator()
    sane_folder = "/Volumes/external/Results/PatternBuffer/sane/results/rebuttal_sane"
    all_firsts = []
    all_lasts = []
    all_step_counts = []

    for experiment_id in range(6):
        for task_id in range(5):
            hypotheses_created = aggregator.read_experiment_data(sane_folder, [experiment_id], task_id=task_id * 2, tag_base="num_hypotheses_created")
            nodes_created = np.array([node[1] for node in hypotheses_created[0]]).sum()
            step_count = hypotheses_created[0][-1][0] - hypotheses_created[0][0][0]
            steps_per_node = step_count/nodes_created
            print(f"Task {task_id}, node created on average every {steps_per_node} steps")

            data_count = len(hypotheses_created[0])
            first_count = np.array([node[1] for node in hypotheses_created[0][:int(data_count/3)]]).sum()
            last_count = np.array([node[1] for node in hypotheses_created[0][-int(data_count/3):]]).sum()
            print(f"{experiment_id}: {task_id}, First total: {first_count}, Last total: {last_count}")

            all_step_counts.append(steps_per_node)
            all_firsts.append(first_count)
            all_lasts.append(last_count)

    print(f"Average step count: {np.array(all_step_counts).mean(), np.array(all_step_counts).std()}")
    print(f"Average first bucket count: {np.array(all_firsts).mean(), np.array(all_firsts).std()}")
    print(f"Average last bucket count: {np.array(all_lasts).mean(), np.array(all_lasts).std()}")


def create_graph_alfred_vary_envs():

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/Alfred/results/vary_envs_2"

    task_steps = 1e6
    num_tasks = 3
    num_cycles = 2
    tasks = [(0, f"Env_402", [[task_steps*i, task_steps*(i+1)] for i in range(0, 12, num_tasks)], [-15, 15], [-10, 12.0]),
             (1, f"Env_419", [[task_steps*i, task_steps*(i+1)] for i in range(1, 12, num_tasks)], [-15, 15], [-10, 12.0]),  # Max reward taken from running the true demo on the trajectory
             (2, f"Env_423", [[task_steps*i, task_steps*(i+1)] for i in range(2, 12, num_tasks)], [-15, 15], [-10, 12.0])]

    all_task_ids = list(range(num_tasks * num_cycles))  # 2 cycles of 3

    for id, task_info in enumerate(tasks):
        task_id, task_name, shaded_regions, y_range, allowed_y_range = task_info
        graph = []
        clear_data = aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [0, 1, 2], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range)
        graph.append((clear_data, "CLEAR", False))

        ewc_data = aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [3, 4, 5], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range)
        graph.append((ewc_data, "EWC", False))

        pnc_data = aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [9, 10, 11], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range)
        graph.append((pnc_data, "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        if id < len(all_task_ids)-1:
            compute_forgetting_metric(task_name, "CLEAR", clear_data, task_steps, task_id, all_task_ids[id+1:])

        for cycle_id in range(1):  # TOOD: only doing this for the first cycle, for consistency with the "zero shot" moniker, otherwise it's confusing.
            compute_forward_transfer_metric(task_name, "CLEAR", clear_data, task_steps,
                                            task_id + cycle_id * num_tasks, all_task_ids[:id + cycle_id*num_tasks])

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=y_range,
                                                shaded_regions=shaded_regions, filename=f"tmp/neurips_datasets/vary_envs/{task_name}.eps",
                                                title_size=30, x_range=[-10, 6e6])


def create_graph_alfred_vary_tasks():

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/Alfred/results/vary_tasks_2"

    task_steps = 1e6
    num_tasks = 3
    tasks = [(0, f"Hang_TP", [[task_steps*i, task_steps*(i+1)] for i in range(0, 12, num_tasks)], [-15, 15], [-10, None]),
             (1, f"Put_TP_in_Cabinet", [[task_steps*i, task_steps*(i+1)] for i in range(1, 12, num_tasks)], [-15, 15], [-10, None]),
             (2, f"Put_TP_on_Counter", [[task_steps*i, task_steps*(i+1)] for i in range(2, 12, num_tasks)], [-15, 15], [-10, None])]

    for task_info in tasks:
        task_id, task_name, shaded_regions, y_range, allowed_y_range = task_info
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [0, 1, 2], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [3, 4, 5], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "EWC", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [6, 7, 8], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=y_range,
                                                shaded_regions=shaded_regions, filename=f"tmp/neurips_datasets/vary_tasks/{task_name}.eps",
                                                title_size=30, x_range=[-10, 6e6])


def create_graph_alfred_vary_objects():

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/Alfred/results/vary_objects_3"

    task_steps = 1e6
    num_tasks = 3
    tasks = [(0, f"Clean_Fork", [[task_steps*i, task_steps*(i+1)] for i in range(0, 12, num_tasks)], [-15, 15], [-10, None]),
             (1, f"Clean_Knife", [[task_steps*i, task_steps*(i+1)] for i in range(1, 12, num_tasks)], [-15, 15], [-10, None]),
             (2, f"Clean_Spoon", [[task_steps*i, task_steps*(i+1)] for i in range(2, 12, num_tasks)], [-15, 15], [-10, None])]

    for task_info in tasks:
        task_id, task_name, shaded_regions, y_range, allowed_y_range = task_info
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [0, 1, 2], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [3, 4], task_id=task_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "EWC", False))
        #graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [6, 7, 8], task_id=task_id, tag_base="eval_reward"),
        #                                         rolling_mean_count=5, allowed_y_range=allowed_y_range), "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=y_range,
                                                shaded_regions=shaded_regions, filename=f"tmp/neurips_datasets/vary_objects/{task_name}.eps",
                                                title_size=30, x_range=[-10, 6e6])


def create_graph_alfred_multi_traj(eval_mode):

    aggregator = EventsResultsAggregator()
    procgen_folder = "/Volumes/external/Results/Alfred/results/multi_traj"

    task_steps = 1e6
    num_tasks = 3
    offset = 1 if eval_mode else 0
    name_ext = "_eval" if eval_mode else ""
    tasks = [(0, 0+offset, f"Kitchen_19_Cup{name_ext}", [[task_steps*i, task_steps*(i+1)] for i in range(0, 12, num_tasks)], [-15, 15], [-10, None]),
             (1, 2+offset, f"Kitchen_13_Sliced_Potato{name_ext}", [[task_steps*i, task_steps*(i+1)] for i in range(1, 12, num_tasks)], [-15, 15], [-10, None]),
             (2, 4+offset, f"Kitchen_2Sliced_Lettuce{name_ext}", [[task_steps*i, task_steps*(i+1)] for i in range(2, 12, num_tasks)], [-15, 15], [-10, None])]

    all_task_ids = list(range(6))  # 2 cycles of 3

    for id, task_info in enumerate(tasks):
        task_id, graph_id, task_name, shaded_regions, y_range, allowed_y_range = task_info
        graph = []
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [0, 1, 2], task_id=graph_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "CLEAR", False))
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [3], task_id=graph_id, tag_base="eval_reward"),
                                                 rolling_mean_count=5, allowed_y_range=allowed_y_range), "EWC", False))
        #graph.append((aggregator.post_processing(aggregator.read_experiment_data(procgen_folder, [6, 7, 8], task_id=graph_id, tag_base="eval_reward"),
        #                                         rolling_mean_count=5, allowed_y_range=allowed_y_range), "PnC", False))

        filtered_data = []
        for run_data, run_label, line_is_dashed in graph:
            xs, filtered_means, filtered_stds = aggregator.combine_experiment_data(run_data)
            filtered_data.append((xs, filtered_means, filtered_stds, run_label, line_is_dashed))

        if id < len(all_task_ids)-1:
            compute_forgetting_metric(filtered_data, task_steps, task_id, all_task_ids[id+1:])

        aggregator.plot_multiple_lines_on_graph(filtered_data, f"{task_name}", x_offset=10, y_range=y_range,
                                                shaded_regions=shaded_regions, filename=f"tmp/neurips_datasets/vary_objects/{task_name}.eps",
                                                title_size=30, x_range=[-10, 6e6])


def _get_rewards_for_region(xs, ys, region):
        valid_x_mask_lower = xs > region[0] if region[0] is not None else True  # If we have no lower bound specified, all xs are valid
        valid_x_mask_upper = xs < region[1] if region[1] is not None else True
        valid_x_mask = valid_x_mask_lower * valid_x_mask_upper

        return ys[valid_x_mask]


def compute_forgetting_metric(task_name, algo_name, task_results, task_steps, task_id, subsequent_task_ids):
    """
    We compute how much is forgotten of task (task_id) as each subsequent (subsequent_task_id) is learned.
    """
    total_forgetting_per_subsequent = {id: 0 for id in subsequent_task_ids}

    for run_id, task_result in enumerate(task_results):
        xs = np.array([t[0] for t in task_result])
        ys = np.array([t[1] for t in task_result])

        # Select only the rewards from the region up to and including the training of the given task
        #task_rewards = _get_rewards_for_region(task_algo_results, [task_id * task_steps, (task_id+1) * task_steps])  # Only the specific training region - NOT consistent with paper, just for lookin'
        task_rewards = _get_rewards_for_region(xs, ys, [None, (task_id+1) * task_steps])
        #max_task_value = np.max(task_rewards)
        max_task_value = task_rewards[-1]

        for subsequent_task_id in subsequent_task_ids:
            subsequent_region = [subsequent_task_id * task_steps, (subsequent_task_id+1) * task_steps]  # TODO: could do from the end of the task up to the subsequent one we're looking at...
            subsequent_task_rewards = _get_rewards_for_region(xs, ys, subsequent_region)
            last_reward = subsequent_task_rewards[-1]
            forgetting = max_task_value - last_reward

            total_forgetting_per_subsequent[subsequent_task_id] += forgetting
            print(f"[{task_name}: {algo_name}] Forgetting metric (F) for task {task_id} after training on {subsequent_task_id}: {forgetting} (i max: {max_task_value}, j last: {last_reward})")

    for subsequent_id in total_forgetting_per_subsequent.keys():
        average_forgetting = total_forgetting_per_subsequent[subsequent_id] / len(task_results)
        print(f"[{task_name}: {algo_name}] Task {task_id}, Subsequent: {subsequent_id} Average forgetting: {average_forgetting}")

    pass


def compute_forward_transfer_metric(task_name, algo_name, task_results, task_steps, task_id, prior_task_ids):
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
            subsequent_task_rewards = _get_rewards_for_region(xs, ys, prior_region)
            last_reward = subsequent_task_rewards[-1]
            transfer = last_reward - initial_task_value

            total_transfer_per_prior[prior_task_id] += transfer
            print(f"[{task_name}: {algo_name}] Transfer metric (T) for task {task_id} after training on {prior_task_id}: {transfer} (i initial: {initial_task_value}, j last: {last_reward})")

    for prior_id in total_transfer_per_prior.keys():
        average_transfer = total_transfer_per_prior[prior_id] / len(task_results)
        print(f"[{task_name}: {algo_name}] Task {task_id}, Prior: {prior_id} Average transfer: {average_transfer}")

    pass


if __name__ == "__main__":
    #compute_mnist_averages()
    #create_graph_mnist()
    #create_graph_minigrid_oddoneout()
    #create_graph_minigrid_oddoneout_sane_buffer_ablation()
    #create_graph_minigrid_oddoneout_sane_node_count_ablation()
    #create_graph_mnist_clear_comp()
    #create_graph_mnist_sane_alpha_ablation()
    #create_graph_mnist_clear_collect_freq()
    #create_graph_atari_clear()
    #create_graph_atari_ewc()
    #create_graph_atari_pnc()
    #create_graph_split_mnist()
    #compute_num_hypotheses()
    #create_graph_procgen()
    #create_graph_atari()
    #create_graph_thor_sequential_task()
    #create_graph_thor_sequential_environment()

    #create_graph_alfred_vary_envs()
    create_graph_alfred_vary_tasks()
    #create_graph_alfred_vary_objects()
    #create_graph_alfred_multi_traj(eval_mode=False)
    #create_graph_alfred_multi_traj(eval_mode=True)
