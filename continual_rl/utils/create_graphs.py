import os
import pickle
from tensorflow.python.summary.summary_iterator import summary_iterator
import math
import numpy as np
import plotly
import plotly.graph_objs as go
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

        print("Attempting to use pickle: {}".format(pickle_file_name))

        try:
            with open(pickle_path, 'rb') as pickled_file:
                event_data = pickle.load(pickled_file)
        except FileNotFoundError:
            event_data = []

            for event in summary_iterator(event_file_path):
                global_step = event.step

                for val in event.summary.value:
                    if val.tag == tag:
                        value = val.simple_value
                        event_data.append((global_step, value))

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

            for path, dirs, files in os.walk(full_run_path):
                for file in files:
                    if "events" in file:
                        assert event_file is None, "Multiple events found unexpectedly."
                        event_file = os.path.join(path, file)

            assert event_file is not None, "No event file found when one was expected."

            full_tag = "{}/{}".format(tag_base, task_id)

            run_data = self._read_event_file(event_file, full_tag)
            collected_run_data.append(run_data)

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
                                     filename=None, legend_size=40, title_size=50):
        traces = []
        min_x = 0  # Effectively defaulting to 0
        max_x = 0

        for run_id, run_data in enumerate(experiment_data):
            xs, y_means, y_stds, line_label, line_is_dashed = run_data

            color = self.COLORS[run_id]

            traces.extend(self._create_scatters(xs, y_means,
                                                y_stds, line_label, color[0], color[1], is_dashed=line_is_dashed))
            if xs.min() < min_x:
                min_x = xs.min()
            if xs.max() > max_x:
                max_x = xs.max()

        x_range = x_range or [min_x-x_offset, max_x+x_offset]

        layout = go.Layout(
            yaxis=dict(title=dict(text='Reward', font=dict(size=40)), range=y_range, tickfont=dict(size=30)),
            xaxis=dict(title=dict(text='Step', font=dict(size=40)), range=x_range, tickfont=dict(size=30)),
            title=dict(text=title, font=dict(size=title_size)),
            legend=dict(font=dict(size=legend_size, color="black")))

        fig = go.Figure(data=traces, layout=layout)
        
        fig.update_layout(title_x=0.5)

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
            fig.write_image(filename)

    def post_processing(self, experiment_data, rolling_mean_count):
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
                ys[x_id] = np.array(rolling_accumulator).mean()

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
        graph.append((aggregator.post_processing(aggregator.read_experiment_data(clear_folder, list(range(0, 5)), task_id=digit_id*2, tag_base="eval_reward"),
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
        collected_data.append((aggregator.read_experiment_data(clear_folder, list(range(0, 5)), task_id=digit_id*2+1, tag_base="train_reward"), "CLEAR", False))
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


def create_graph_minigrid_oddoneout_obst_clear_comp():
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


if __name__ == "__main__":
    #compute_mnist_averages()
    #create_graph_mnist()
    #create_graph_minigrid_oddoneout()
    create_graph_minigrid_oddoneout_sane_buffer_ablation()
    create_graph_minigrid_oddoneout_sane_node_count_ablation()
