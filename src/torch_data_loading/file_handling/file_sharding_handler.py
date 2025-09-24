import logging
import math
from functools import cached_property
from os import walk
from os.path import basename as bn, join as pj, isfile
from typing import Any, Callable, Iterator, Optional, Sequence, BinaryIO

import torch

from torch_data_loading.utils.io_utils import read_file
from torch_data_loading.utils.base_pipe import ReprShufflerPipe

logger = logging.getLogger(__name__)

__all__ = ["LocalIterableDatasetShardedInfinite"]


def _walk_files(path: str) -> list[str]:
    if isfile(path):
        return read_file(path)

    files = []
    for root, dirnames, filenames in walk(path):
        files.extend([pj(root, f) for f in filenames])

    return files


def _local_transform(data: str) -> str:
    return data


class LocalIterableDatasetShardedInfinite(ReprShufflerPipe):
    num_workers: int

    def __init__(
        self,
        path_to_files: str | list[str],
        transform: Callable[[str], Any] = _local_transform,
        file_finder: Callable[[str], list[str]] = _walk_files,
        file_filter_name: Sequence[tuple[str, str | tuple[str, ...]]] = tuple(),  # (("endswith", ".tar"),),
        shuffle_files: bool = True,
        num_iterations: float = math.inf,
        dataset_name: str = "",
        rng_seed: Optional[int] = None,
    ):
        """_summary_

        Pipe to open files. Can be adapted to other use cases by changing the transform.

        Features:
            1. Shard files by rank. Uses sharded_length() to track files available to this worker.
            2. Support for making the data stream infinite while sharding (see implementation strategy below).
            3. Shuffling the files (using the integer index).
            4. Output with a transformation. Default yields (string path, data stream) tuple.

        Infinite Sharding logic:
            * Every worker holds a list of all files for simplicity.
            * Files are treated mod (N == Total Number of files). Mathematically as `Z / NZ`, represented by
                their index in the previous list.
            * Each worker is given the coset of Z / NZ spanned by
                element = (rank * num_workers + worker_id) + i * world_size * num_workers
                where i is all integers.
            * The number of files available is given as self.sharded_length(), and is
                num_files / gcd(num_files, num_workers * world_size). Note that this means workers can contain the
                same file, just in a different order. 
                > As an example, with 4 workers and 5 files, the list for worker 1 would be
                    [1, 5, 9, 13, 17, ...] (mod 5) == [1, 0, 4, 3, 2] (e.g. all files.)
            * We shuffle indices during training at the start of each iteration (e.g. per epoch.)
            * The motivation for sharing files is to ensure each file gets equal representation in large scale training.
                    For example, if you have 65 files and 64 data workers (e.g. 16 GPus * 4 workers), one worker
                     would hold 2 files in the previous case.

        Finite sharding logic:
            * We simply use
                rank * num_workers + worker_id, len(self) * num_iterations, world_size * num_workers
                to determine the file indices to be used.

            * In the case num_iterations = 1, files are not shared per worker. This is the default case for eval.
            * In the case num_iterations > 1, files are shared per worker
            * Each worker can have unequal file counts (even 0, e.g. if num_files < world_size * num_workers).
            * Special handling needs to be implemented in this case for distributed training.

        Args:
            path_to_files:
            transform (Callable[[S3Reader], Any], optional): _description_. Defaults to _transform.
            s3client_config (Optional[S3ClientConfig], optional): See `S3IterableDataset`. Defaults to None.
            dataset_name (str, optional): For superclass `ShufflingIterDataPipe`; to track a datapipes source.
                Defaults to "".
            file_filter_name (Sequence[tuple[str, str]], optional): Used to ensure only the correct file structures
                are used  (e.g. "*.tar" by default). Defaults to (("endswith", ".tar"),).
            shuffle_files (bool, optional): When true, we establish the list of files and shuffle them.
                Defaults to True.
            num_iterations (float, optional): How many times to iterate through the data.
                Defaults to math.inf (non-stop).
            rng_seed (int, optional): Provide a seed to fix dataloading setup.
        """

        filter_types = [(x, tuple(y) if isinstance(y, list) else y) for x, y in file_filter_name]
        self.filter_types = filter_types
        self.shuffle_files = shuffle_files
        self.num_iterations = num_iterations
        self.dataset_name = dataset_name
        self.base_path = path_to_files

        self.transform = transform
        self.file_finder = file_finder

        self._rank = 0
        self._world_size = 1

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

        repr_args = {
            k: getattr(self, k) for k in ("base_path", "_rank", "_world_size", "shuffle_files", "num_iterations")
        }
        repr_args["rng_seed"] = rng_seed
        # Method to delay file check, making pipes load instantly.
        repr_args["DELAY:total_files"] = "__len__"
        repr_args["DELAY:num_files_in_gpu"] = "sharded_length"

        self.loaded: bool = False

        super().__init__(
            datapipe=None,
            dataset_name=dataset_name,
            strict_dp=False,
            repr_args=repr_args,
            rng_seed=rng_seed,
        )

    @cached_property
    def paths(self) -> list[str]:
        if isinstance(self.base_path, str):
            uris = self.file_finder(self.base_path)
        else:
            uris = [y for x in self.base_path for y in self.file_finder(x)]

        uris = [x for x in uris if self._eval_filter(x)]
        return uris

    def _eval_filter(self, key: str) -> bool:
        key = bn(key)
        return all(getattr(key, method, lambda x: True)(condition) for method, condition in self.filter_types)

    def __len__(self) -> int:
        return len(self.paths)

    def get_worker_count(self, expected_num_workers: Optional[int] = None) -> int:
        num_workers = getattr(self, "num_workers", expected_num_workers)
        assert (
            num_workers is not None
        ), "Need to either provide `expected_num_workers` or run __iter__ method. Workers found to be `None`."
        return num_workers

    def sharded_length(self, expected_num_workers: Optional[int] = 1) -> int:
        # Finds the number of files accessible to this worker.
        num_workers = self.get_worker_count(expected_num_workers)
        if self.num_iterations < math.inf:
            return math.ceil(len(self) * self.num_iterations / (self._world_size * num_workers))

        return int(len(self) / math.gcd(len(self), self._world_size * num_workers))

    def _shuffle_indices(self, indices: list[int], worker_id: int):
        self.rng.shuffle(indices)
        logger.debug(f"Shuffled indices on worker {worker_id} ({len(indices)} total): {indices}.")

    def index_list_infinite(self, worker_id: int, num_workers: int) -> list[int]:
        if not hasattr(self, "indices"):
            sharded_length = self.sharded_length(num_workers)
            coset = self._rank + worker_id * self._world_size
            index_skips = self._world_size * num_workers

            self.indices = [(coset + idx * index_skips) % len(self) for idx in range(0, sharded_length)]
            logger.debug(f"Indices used on worker {worker_id} ({len(self.indices)} total): {self.indices}.")

        if self.shuffle_files:
            self._shuffle_indices(self.indices, worker_id)

        return self.indices

    def index_list_finite(self, worker_id: int, num_workers: int) -> list[int]:
        if not hasattr(self, "indices"):
            indices = list(
                range(
                    self._rank + worker_id * self._world_size,
                    int(len(self) * self.num_iterations),
                    self._world_size * num_workers,
                )
            )
            self.indices = [x % len(self) for x in indices]
            logger.debug(f"Indices used on worker {worker_id} ({len(self.indices)} total): {self.indices}.")

        if self.shuffle_files:
            self._shuffle_indices(self.indices, worker_id)

        return self.indices

    def index_list(self, worker_id: int, expected_num_workers: Optional[int] = None) -> list[int]:
        if self.loaded and getattr(self, "indices", False):
            return self.indices

        num_workers = self.get_worker_count(expected_num_workers)

        if self.num_iterations < math.inf:
            return self.index_list_finite(worker_id, num_workers)

        return self.index_list_infinite(worker_id, num_workers)

    @property
    def _get_start_iteration_count(self) -> int:
        if self.loaded:
            return self.iteration_count

        return 1

    def __iter__(self) -> Iterator[tuple[str, BinaryIO]]:
        # Can not use a for loop for the infinite case.
        self.iteration_count = self._get_start_iteration_count

        while self.iteration_count <= self.num_iterations:
            indices = self.index_list(*self.get_worker_states())
            for index in indices:
                yield self.transform(self.paths[index])

            dataset_string = f" for dataset {self.dataset_name}" if self.dataset_name else ""
            logger.info(
                f"Iteration ({self.iteration_count} / {self.num_iterations}) complete{dataset_string} for "
                f"{self.workers_string} of {self.rank_string}."
            )
            self.iteration_count += 1

            self.loaded = False
