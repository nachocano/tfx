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
"""Task scheduler interface and registry."""

import abc
from typing import Text, Type, TypeVar

import attr
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core.proto import task_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2


@attr.s(frozen=True)
class TaskSchedulerResult:
  """Response from the task scheduler.

  Attributes:
    executor_output: An instance of `ExecutorOutput` containing the results of
      task execution.
  """
  executor_output = attr.ib(type=execution_result_pb2.ExecutorOutput)


class TaskScheduler(abc.ABC):
  """Interface for task schedulers."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, task: task_pb2.Task):
    """Constructor.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: The pipeline IR proto.
      task: Task to be executed.
    """
    self.mlmd_handle = mlmd_handle
    self.pipeline = pipeline
    self.task = task

  @abc.abstractmethod
  def schedule(self) -> TaskSchedulerResult:
    """Schedules task execution and returns the results of execution.

    This method blocks until task execution completes (successfully or not) or
    until explicitly cancelled by a call to `cancel`. When cancelled, `schedule`
    is expected to stop any ongoing work, clean up and return as soon as
    possible. Note that `cancel` will be invoked from a different thread than
    `schedule` and hence the concrete implementations must be thread safe.
    """

  @abc.abstractmethod
  def cancel(self) -> None:
    """Cancels task scheduler.

    This method will be invoked from a different thread than the thread that's
    blocked on call to `schedule`. `cancel` must return immediately when called.
    Upon cancellation, `schedule` method is expected to stop any ongoing work,
    clean up and return as soon as possible.
    """


T = TypeVar('T', bound='TaskSchedulerRegistry')


class TaskSchedulerRegistry:
  """A registry for task schedulers."""

  _task_scheduler_registry = {}

  @classmethod
  def register(cls: Type[T], executor_spec_type_url: Text,
               scheduler_class: Type[TaskScheduler]) -> None:
    """Registers a new task scheduler for the given executor spec type url.

    Args:
      executor_spec_type_url: The URL of the executor spec type.
      scheduler_class: The class that will be instantiated for a matching task.

    Raises:
      ValueError: If `executor_spec_type_url` is already in the registry.
    """
    if executor_spec_type_url in cls._task_scheduler_registry:
      raise ValueError(
          'A task scheduler already exists for the executor spec type url: {}'
          .format(executor_spec_type_url))
    cls._task_scheduler_registry[executor_spec_type_url] = scheduler_class

  @classmethod
  def create_task_scheduler(cls: Type[T], mlmd_handle: metadata.Metadata,
                            pipeline: pipeline_pb2.Pipeline,
                            task: task_pb2.Task) -> TaskScheduler:
    """Creates a task scheduler for the given task.

    Note that this assumes deployment_config packed in the pipeline IR is of
    type `IntermediateDeploymentConfig`. This detail may change in the future.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: The pipeline IR.
      task: The task that needs to be scheduled.

    Returns:
      An instance of `TaskScheduler` for the given task.

    Raises:
      NotImplementedError: Raised if not an `ExecTask` or if not a node level
        scheduler.
      ValueError: Deployment config not present in the IR proto or if executor
        spec for the node corresponding to `task` not configured in the IR.
    """
    if task.WhichOneof('task_type') != 'exec_task':
      raise NotImplementedError(
          'Can create a task scheduler only for an `ExecTask`.')
    if task.exec_task.WhichOneof('node_or_sub_pipeline_id') != 'node_id':
      raise NotImplementedError(
          'Can create a task scheduler only for node execution, but no node_id '
          'found in task: {}.'.format(task))
    # TODO(b/170383494): Decide which DeploymentConfig to use.
    if not pipeline.deployment_config.Is(
        pipeline_pb2.IntermediateDeploymentConfig.DESCRIPTOR):
      raise ValueError('No deployment config found in pipeline IR.')
    depl_config = pipeline_pb2.IntermediateDeploymentConfig()
    pipeline.deployment_config.Unpack(depl_config)
    node_id = task.exec_task.node_id
    if node_id not in depl_config.executor_specs:
      raise ValueError(
          'Executor spec for node id `{}` not found in pipeline IR.'.format(
              node_id))
    executor_spec_type_url = depl_config.executor_specs[node_id].type_url
    return cls._task_scheduler_registry[executor_spec_type_url](
        mlmd_handle=mlmd_handle, pipeline=pipeline, task=task)
