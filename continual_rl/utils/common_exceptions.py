

class ObservationShapeNotRecognized(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class OutputDirectoryNotSetException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)
