

class TimestepDataBase(object):
    """
    This class stores info each policy needs from its usage phase for when we get to its training phase.
    E.g. the log_probability of taking an action.
    Each entry here stores one timestep's worth of data.
    """

    def __init__(self):
        self.reward = None  # Gets populated by the EnvironmentRunner after the policy creates this object
        self.done = None  # Gets populated by the EnvironmentRunner after the policy creates this object
