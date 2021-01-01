from torch.multiprocessing import Queue
import uuid


class ProcessComms(object):
    def __init__(self, device_id):
        self.incoming_queue = Queue()  # Incoming to hypothesis train process... TODO: rename, these names are outdated
        self.outgoing_queue = Queue()  # Outgoing: sending to core process
        self.process_id = uuid.uuid4()
        self.device_id = device_id
        self.process = None

    @property
    def friendly_name(self):
        return str(self.process_id)[:6]

    def close(self):
        self.incoming_queue.close()
        self.outgoing_queue.close()
