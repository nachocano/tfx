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
"""TaskManager manages the execution and cancellation of tasks."""

from concurrent import futures
import functools
import queue
import threading
import typing
from typing import Iterator, Optional, Text, Tuple, Union

from absl import logging
import attr
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core.proto import task_pb2
from tfx.proto.orchestration import pipeline_pb2

_MAX_DEQUEUE_WAIT_SECS = 5.0

ItemType = Union[task_pb2.Task, tq.TaskId]


class _Multiplexer:
  """Helper class to multiplex multiple queues.

  Provides an iterator interface that yields a tuple `(queue_name, item)` in
  each iteration. Iteration can be stopped by an external thread by setting
  `stop_event`.
  """

  # Exec queue has higher priority than cancellation queue so that items from
  # the latter queue are processed after corresponding exec tasks.
  _EXECQ_PRIO = 1
  _CANCELQ_PRIO = 2
  _LOWEST_PRIO = 9999

  @attr.s(order=True, eq=True)
  class _Item:
    """A wrapper of items to work with `queue.PriorityQueue`."""
    priority = attr.ib(type=int, order=True, eq=True)
    name = attr.ib(type=Text, order=False, eq=False)
    item = attr.ib(type=Optional[ItemType], order=False, eq=False)

  _SENTINEL_ITEM = _Item(_LOWEST_PRIO, '__SENTINEL__', None)

  def __init__(self, execq: tq.TaskQueue, cancelq: tq.TaskIdQueue,
               stop_event: threading.Event, max_dequeue_wait_secs: float,
               process_all_queued_items_before_exit: bool):
    """Constructs `_Multiplexer`.

    Args:
      execq: Task queue for pipeline node executions.
      cancelq: Queue of task ids for tasks to be cancelled.
      stop_event: An event which may be set by other threads as a signal to stop
        reading from the underlying queues.
      max_dequeue_wait_secs: Maximum time to wait when dequeuing if the queue is
        empty.
      process_all_queued_items_before_exit: All existing items in the queues are
        processed before exiting the context manager. This is useful for
        deterministic behavior in tests.
    """
    self._execq = execq
    self._cancelq = cancelq
    self._stop_event = stop_event
    self._max_dequeue_wait_secs = max_dequeue_wait_secs
    self._process_all_queued_items_before_exit = (
        process_all_queued_items_before_exit)

  def __iter__(self) -> Iterator[Tuple[Text, ItemType]]:
    multi_queue = queue.PriorityQueue()
    threads = [
        threading.Thread(
            target=self._process,
            args=(self._execq, self._EXECQ_PRIO, multi_queue)),
        threading.Thread(
            target=self._process,
            args=(self._cancelq, self._CANCELQ_PRIO, multi_queue))
    ]
    for thread in threads:
      thread.start()

    def _join_threads():
      """Waits for all the queue processing threads."""
      for thread in threads:
        thread.join()
      # When done, add a sentinel item to signal end of iteration.
      multi_queue.put(self._SENTINEL_ITEM)

    joiner = threading.Thread(target=_join_threads)
    joiner.start()

    # Iterate and yield until we get a sentinel item in the `multi_queue`.
    for item in iter(multi_queue.get, self._SENTINEL_ITEM):
      # Skip yield if the queue was empty after `max_dequeue_wait_secs`.
      if item.item is None:
        continue
      yield (item.name, item.item)

    joiner.join()

  def _process(self, q: Union[tq.TaskQueue, tq.TaskIdQueue], priority: int,
               multi_queue: queue.Queue) -> None:
    """Dequeues items from a queue and enqueues in `multi_queue`."""
    while not self._stop_event.is_set():
      item = q.dequeue(self._max_dequeue_wait_secs)
      multi_queue.put(self._Item(priority=priority, name=q.name, item=item))
    if self._process_all_queued_items_before_exit:
      # Process any remaining items from the queue before exiting. This is
      # mainly to make tests deterministic.
      while True:
        item = q.dequeue()
        if item is None:
          break
        multi_queue.put(self._Item(priority=priority, name=q.name, item=item))


class TaskManager:
  """TaskManager acts on the tasks fetched from the task queues.

  TaskManager instance can be used as a context manager:
  """

  def __init__(self,
               mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline,
               execq: tq.TaskQueue,
               cancelq: tq.TaskIdQueue,
               max_active_task_schedulers: int,
               max_dequeue_wait_secs: float = _MAX_DEQUEUE_WAIT_SECS,
               process_all_queued_items_before_exit: bool = False):
    """Constructs `TaskManager`.

    Args:
      mlmd_handle: ML metadata db connection.
      pipeline: A pipeline IR proto.
      execq: Task queue for pipeline node executions.
      cancelq: Queue of task ids for tasks to be cancelled.
      max_active_task_schedulers: Maximum number of task schedulers that can be
        active at once.
      max_dequeue_wait_secs: Maximum time to wait when dequeuing if the queue is
        empty.
      process_all_queued_items_before_exit: All existing items in the queues are
        processed before exiting the context manager. This is useful for
        deterministic behavior in tests.
    """
    self._mlmd_handle = mlmd_handle
    self._pipeline = pipeline
    self._handler_by_name = {
        execq.name: functools.partial(self._handle_exec, execq),
        cancelq.name: functools.partial(self._handle_cancel, cancelq),
    }
    self._stop_event = threading.Event()
    self._multi_queue = _Multiplexer(execq, cancelq, self._stop_event,
                                     max_dequeue_wait_secs,
                                     process_all_queued_items_before_exit)
    self._tm_lock = threading.Lock()
    self._tm_thread = None
    self._scheduler_by_task_id = {}
    self._ts_executor = futures.ThreadPoolExecutor(
        max_workers=max_active_task_schedulers)

  def __enter__(self):
    if self._tm_thread:
      raise RuntimeError('TaskManager already started.')
    self._ts_executor.__enter__()
    self._tm_thread = threading.Thread(target=self._process_tasks)
    self._tm_thread.start()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not self._tm_thread:
      raise RuntimeError('TaskManager not started.')
    self._stop_event.set()
    self._tm_thread.join()
    self._ts_executor.__exit__(exc_type, exc_val, exc_tb)

  def _process_tasks(self) -> None:
    """Processes tasks from the multiplexed queue."""
    for name, item in self._multi_queue:
      self._handler_by_name[name](item)

  def _handle_exec(self, execq: tq.TaskQueue, item: ItemType) -> None:
    """Handles execution task."""
    task = typing.cast(task_pb2.Task, item)
    task_id = tq.TaskId.from_task(task)
    with self._tm_lock:
      if task_id in self._scheduler_by_task_id:
        raise RuntimeError(
            'Cannot create multiple task schedulers for the same task; '
            'task_id: {}'.format(task_id))
      scheduler = ts.TaskSchedulerRegistry.create_task_scheduler(
          self._mlmd_handle, self._pipeline, task)
      self._scheduler_by_task_id[task_id] = scheduler
      self._ts_executor.submit(self._schedule, scheduler, execq, task, task_id)

  def _handle_cancel(self, cancelq: tq.TaskIdQueue, item: ItemType) -> None:
    """Handles cancellation task."""
    task_id = typing.cast(tq.TaskId, item)
    with self._tm_lock:
      scheduler = self._scheduler_by_task_id.get(task_id)
      if scheduler is None:
        logging.info(
            'No task scheduled for task id: %s. The task might have '
            'already completed before it could be cancelled.', task_id)
        return
      scheduler.cancel()
      cancelq.task_done(task_id)

  def _schedule(self, scheduler: ts.TaskScheduler, execq: tq.TaskQueue,
                task: task_pb2.Task, task_id: tq.TaskId) -> None:
    """Schedules task execution using the given task scheduler."""
    # This is a blocking call to the scheduler which can take a long time to
    # complete for some types of task schedulers.
    resp = scheduler.schedule()
    _publish_execution_results(self._mlmd_handle, self._pipeline, task, resp)
    with self._tm_lock:
      del self._scheduler_by_task_id[task_id]
      execq.task_done(task)


def _publish_execution_results(mlmd_handle: metadata.Metadata,
                               pipeline: pipeline_pb2.Pipeline,
                               task: task_pb2.Task,
                               resp: ts.TaskSchedulerResponse) -> None:
  # TODO(goutham): Implement this.
  del mlmd_handle, pipeline, task, resp
