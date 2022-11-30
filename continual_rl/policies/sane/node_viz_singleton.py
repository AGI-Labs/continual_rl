import multiprocessing
import requests
from multiprocessing import Manager, Process
import urllib
import os


SERVER_IP = os.environ.get("SANE_VIZ_SERVER_IP", "127.0.0.1")


class NodeVizSingleton(object):
    """
    Starts up a process that handles sending SANE tree information out to the visualizer.
    (Code hosted separately at <coming soon>, since a django server is sufficiently unrelated to the point of CORA.)
    """
    _instance = None

    def __init__(self) -> None:
        self._message_queue = Manager().Queue()
        self._stop = False

        self._process = Process(target=self.process_queue, args=(self._message_queue,))
        self._process.start()

    def shutdown(self):
        self._stop = True

    def process_queue(self, message_queue):
        print("Starting queue processing")
        while not self._stop:
            message_data = message_queue.get()
            print(f"Processing {message_data}")

            if message_data["type"] == "create":
                self._create_node_internal(message_data["tree_id"], message_data["node_id"])
            elif message_data["type"] == "register_created_from":
                self._register_created_from_internal(message_data["tree_id"], message_data["node_id"],
                                                     message_data["created_from_id"])
            elif message_data["type"] == "merge":
                self._merge_node_internal(message_data["tree_id"], message_data["node_id"],
                                          message_data["merged_into_id"])

    def wait(self):
        self._process.join()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = NodeVizSingleton()
        return cls._instance

    def make_tree_name_safe(self, tree_id):
        return urllib.parse.quote_plus(tree_id.replace("/", "_"))

    def _create_node_internal(self, tree_id, node_id):
        requests.get(f"http://{SERVER_IP}/api/get_or_create/{self.make_tree_name_safe(tree_id)}/{node_id}")

    def _register_created_from_internal(self, tree_id, node_id, created_from_id):
        requests.get(f"http://{SERVER_IP}/api/created_from/{self.make_tree_name_safe(tree_id)}/{node_id}/{created_from_id}")

    def _merge_node_internal(self, tree_id, node_id, merged_into_id):
        requests.get(f"http://{SERVER_IP}/api/merge/{self.make_tree_name_safe(tree_id)}/{node_id}/{merged_into_id}")

    def create_node(self, tree_id, node_id):
        self._message_queue.put({"type": "create", "tree_id": tree_id, "node_id": node_id})

    def register_created_from(self, tree_id, node_id, created_from_id):
        self._message_queue.put({"type": "register_created_from", "tree_id": tree_id, "node_id": node_id,
                                 "created_from_id": created_from_id})

    def merge_node(self, tree_id, node_id, merged_into_id):
        self._message_queue.put(
            {"type": "merge", "tree_id": tree_id, "node_id": node_id, "merged_into_id": merged_into_id})


if __name__ == "__main__":
    # Run this script just to test that the visualizer is working
    multiprocessing.set_start_method('spawn')
    NodeVizSingleton.instance().create_node("py_tree_1", 0)
    NodeVizSingleton.instance().create_node("py_tree_1", 1)
    NodeVizSingleton.instance().create_node("py_tree_1", 2)
    NodeVizSingleton.instance().register_created_from("py_tree_1", 1, 0)
    NodeVizSingleton.instance().register_created_from("py_tree_1", 2, 0)

    NodeVizSingleton.instance().create_node("py_tree_1", 3)
    NodeVizSingleton.instance().register_created_from("py_tree_1", 3, 2)

    NodeVizSingleton.instance().merge_node("py_tree_1", 2, 1)
    NodeVizSingleton.instance().merge_node("py_tree_1", 3, 1)
    NodeVizSingleton.instance().wait()
