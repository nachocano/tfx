# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task queue."""

import queue
import threading
from typing import Callable, Generic, Optional, Text, Type, TypeVar

import attr
from tfx.orchestration.experimental.core.proto import task_pb2
from tfx.proto.orchestration import pipeline_pb2

_TASK_TYPE = 'task_type'
_EXEC_TASK = 'exec_task'
_NODE_OR_SUBP_ID = 'node_or_sub_pipeline_id'
_NODE_ID = 'node_id'

_T = TypeVar('_T', bound='_ExecTaskId')


@attr.s(frozen=True)
class _ExecTaskId:
  """Unique identifier for an `ExecTask`."""
  node_id = attr.ib(type=Text)
  pipeline_id = attr.ib(type=Text)
  pipeline_run_id = attr.ib(type=Optional[Text])

  @classmethod
  def from_exec_task(cls: Type[_T], exec_task: task_pb2.ExecTask) -> _T:
    """Creates an instance from `ExecTask`."""
    if exec_task.WhichOneof(_NODE_OR_SUBP_ID) != _NODE_ID:
      raise ValueError('Supported exec task id type: {}'.format(_NODE_ID))
    return cls(
        node_id=exec_task.node_id,
        pipeline_id=exec_task.pipeline_id,
        pipeline_run_id=exec_task.pipeline_run_id or None)

  @classmethod
  def from_pipeline_node(cls: Type[_T], pipeline: pipeline_pb2.Pipeline,
                         node: pipeline_pb2.PipelineNode) -> _T:
    """Creates an instance from pipeline and node definitions."""
    pipeline_run_id = (
        pipeline.runtime_spec.pipeline_run_id.field_value.string_value
        if pipeline.runtime_spec.HasField('pipeline_run_id') else None)
    return cls(
        node_id=node.node_info.id,
        pipeline_id=pipeline.pipeline_info.id,
        pipeline_run_id=pipeline_run_id)


_U = TypeVar('_U', bound='TaskId')


@attr.s(frozen=True)
class TaskId:
  """Unique identifier for a `Task`."""
  exec_task_id = attr.ib(type=Optional[_ExecTaskId])

  @classmethod
  def from_task(cls: Type[_U], task: task_pb2.Task) -> _U:
    """Creates an instance from `Task`."""
    if task.WhichOneof(_TASK_TYPE) != _EXEC_TASK:
      raise ValueError('Task type supported: `{}`'.format(_EXEC_TASK))
    return cls(exec_task_id=_ExecTaskId.from_exec_task(task.exec_task))

  @classmethod
  def from_pipeline_node(cls: Type[_U], pipeline: pipeline_pb2.Pipeline,
                         node: pipeline_pb2.PipelineNode) -> _U:
    """Creates an instance from pipeline and node definitions."""
    return cls(exec_task_id=_ExecTaskId.from_pipeline_node(pipeline, node))


ItemType = TypeVar('ItemType')
KeyType = TypeVar('KeyType')


class _Queue(Generic[ItemType, KeyType]):
  """A thread-safe queue that supports duplicate detection.

  The life-cycle of an item starts with producers calling `enqueue`. Consumers
  call `dequeue` to obtain the tasks in FIFO order. When processing is complete,
  consumers must release the tasks by calling `task_done`.
  """

  def __init__(self, name: Text, key_fn: Callable[[KeyType], ItemType]):
    self.name = name
    self._key_fn = key_fn
    self._lock = threading.Lock()
    self._keys = set()
    self._queue = queue.Queue()  # replace with `queue.SimpleQueue` in Py3.7+.
    self._pending_items_by_key = {}

  def enqueue(self, item: ItemType) -> bool:
    """Enqueues the given item if no item having the same key exists.

    Args:
      item: An item.

    Returns:
      `True` if the item could be enqueued. `False` if an item with the same key
      already exists.
    """
    with self._lock:
      key = self._key_fn(item)
      if key in self._keys:
        return False
      self._keys.add(key)
      self._queue.put((key, item))
      return True

  def dequeue(self,
              max_wait_secs: Optional[float] = None) -> Optional[ItemType]:
    """Removes and returns an item from the queue.

    Once the processing is complete, queue consumers must call `task_done`.

    Args:
      max_wait_secs: If not `None`, waits a maximum of `max_wait_secs` when the
        queue is empty for an item to be enqueued. If no item is present in the
        queue after the wait, `None` is returned. If `max_wait_secs` is `None`
        (default), returns `None` without waiting when the queue is empty.

    Returns:
      An item or `None` if the queue is empty.
    """
    with self._lock:
      try:
        key, item = self._queue.get(
            block=max_wait_secs is not None, timeout=max_wait_secs)
      except queue.Empty:
        return None
      self._pending_items_by_key[key] = item
      return item

  def task_done(self, item: ItemType) -> None:
    """Marks the processing of an item as done.

    Consumers should call this method after the task is processed.

    Args:
      item: An item.

    Raises:
      RuntimeError: If attempt is made to mark a non-existent or non-dequeued
      item as done.
    """
    with self._lock:
      key = self._key_fn(item)
      if key not in self._pending_items_by_key:
        if key in self._keys:
          raise RuntimeError(
              'Must call `dequeue` before calling `task_done`; item key: {}'
              .format(key))
        else:
          raise RuntimeError(
              'Item not present in the queue; item key: {}'.format(key))
      self._pending_items_by_key.pop(key)
      self._keys.remove(key)

  def is_key_present(self, key: KeyType) -> bool:
    """Returns `True` if an item with the given key is present in the queue.

    The task is considered present if it has been `enqueue`d, probably
    `dequeue`d but `task_done` has not been called.

    Args:
      key: The item key.

    Returns:
      `True` if an item with the given key is present.
    """
    with self._lock:
      return key in self._keys

  def is_empty(self) -> bool:
    """Returns `True` if the queue is empty."""
    return not self._keys


class TaskQueue(_Queue[task_pb2.Task, TaskId]):
  """A `_Queue` for task protos."""

  def __init__(self, name):
    super(TaskQueue, self).__init__(name, TaskId.from_task)


class TaskIdQueue(_Queue[TaskId, TaskId]):
  """A `_Queue` for task ids."""

  def __init__(self, name):
    super(TaskIdQueue, self).__init__(name, lambda x: x)
