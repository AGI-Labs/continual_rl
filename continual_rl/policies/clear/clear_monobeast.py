import torch
import tempfile
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers


class ClearMonobeast(Monobeast):
    """
    An implementation of Experience Replay for Continual Learning (Rolnick et al, 2019):
    https://arxiv.org/pdf/1811.11682.pdf
    """
    def __init__(self, model_flags, observation_space, action_space, policy_class):
        super().__init__(model_flags, observation_space, action_space, policy_class)

        self._replay_buffers, self._temp_files = self._create_replay_buffers(model_flags, observation_space.shape,
                                                                      action_space.n)

    def _create_file_backed_tensor(self, file_path, shape, dtype):
        temp_file = tempfile.NamedTemporaryFile(dir=file_path)

        size = 1
        for dim in shape:
            size *= dim

        storage_type = None
        tensor_type = None
        if dtype == torch.uint8:
            storage_type = torch.ByteStorage
            tensor_type = torch.ByteTensor
        elif dtype == torch.int32:
            storage_type = torch.IntStorage
            tensor_type = torch.IntTensor
        elif dtype == torch.int64:
            storage_type = torch.LongStorage
            tensor_type = torch.LongTensor
        elif dtype == torch.bool:
            storage_type = torch.BoolStorage
            tensor_type = torch.BoolTensor
        elif dtype == torch.float32:
            storage_type = torch.FloatStorage
            tensor_type = torch.FloatTensor

        shared_file_storage = storage_type.from_file(temp_file.name, shared=True, size=size)
        new_tensor = tensor_type(shared_file_storage).view(shape)

        return new_tensor, temp_file

    def _create_replay_buffers(self, model_flags, obs_shape, num_actions):
        """
        Key differences from normal buffers:
        1. File-backed, so we can store more at a time
        2. Structured so that there are num_actors buffers, each with entries_per_buffer entries

        Each buffer entry has unroll_length size, so the number of frames stored is (roughly, because of integer
        rounding): num_actors * entries_per_buffer * unroll_length
        """
        entries_per_buffer = model_flags.replay_buffer_frames // (model_flags.unroll_length * model_flags.num_actors)
        specs = self.create_buffer_specs(model_flags.unroll_length, obs_shape, num_actions)
        buffers: Buffers = {key: [] for key in specs}

        # Hold on to the file handle so it does not get deleted. Technically optional, as at least linux will
        # keep the file open even after deletion, but this way it is still visible in the location it was created
        temp_files = []

        for _ in range(model_flags.num_actors):
            for key in buffers:
                shape = (entries_per_buffer, *specs[key]["size"])
                new_tensor, temp_file = self._create_file_backed_tensor(model_flags.large_file_path, shape,
                                                                        specs[key]["dtype"])
                buffers[key].append(new_tensor.share_memory_())
                temp_files.append(temp_file)

        return buffers, temp_files
