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
"""Utilities to evaluate and resolve Placeholders."""

import re
from typing import Any, Callable, Dict

import attr
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration.portable import base_executor_operator
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import value_artifact

from google.protobuf import descriptor_pool
from google.protobuf import json_format
from google.protobuf import message
from google.protobuf import message_factory
from google.protobuf import text_format


@attr.s(auto_attribs=True, frozen=True)
class ResolutionContext:
  """A struct to store information needed for resolution.

  Attributes:
    exec_info: An ExecutionInfo object that includes needed information to
      render all kinds of placeholders.
    executor_spec: An executor spec proto for rendering context placeholder.
    platform_config: A platform config proto for rendering context placeholder.
  """
  exec_info: base_executor_operator.ExecutionInfo = None
  executor_spec: message.Message = None
  platform_config: message.Message = None


def resolve_placeholder_expression(
    expression: placeholder_pb2.PlaceholderExpression,
    context: ResolutionContext) -> Any:
  """Evaluates a placeholder expression using the given context.

  Args:
    expression: A placeholder expression to be resolved.
    context: Information needed to resolve the expression.

  Returns:
    Resolved expression value.
  """
  if not context.exec_info.pipeline_node or not context.exec_info.pipeline_info:
    raise ValueError(
        "Pipeline node or pipeline info is missing from the placeholder ResolutionContext."
    )
  return _ExpressionResolver(context).resolve(expression)


# Dictionary of registered placeholder operators,
# maps from operator proto type names to actual operator functions.
_PLACEHOLDER_OPERATORS: Dict[str, Callable[..., Any]] = {}


def _register(op_proto):
  """Decorator function for registering operators. Internal in this module."""

  def decorator(op: Callable[..., Any]):
    _PLACEHOLDER_OPERATORS[op_proto.DESCRIPTOR.name] = op
    return op

  return decorator


class _ExpressionResolver:
  """Utility class to resolve Placeholder expressions.

  Placeholder expression is defined as a proto structure
  placeholder_pb2.PlaceholderExpression. It can be resolved with
  ResolutionContext to a concrete value.
  """

  def __init__(self, context: ResolutionContext):
    self._resolution_values = {
        placeholder_pb2.Placeholder.Type.INPUT_ARTIFACT:
            context.exec_info.input_dict,
        placeholder_pb2.Placeholder.Type.OUTPUT_ARTIFACT:
            context.exec_info.output_dict,
        placeholder_pb2.Placeholder.Type.EXEC_PROPERTY:
            context.exec_info.exec_properties,
        placeholder_pb2.Placeholder.Type.RUNTIME_INFO: {
            ph.RuntimeInfoKey.EXECUTOR_SPEC.value:
                context.executor_spec,
            ph.RuntimeInfoKey.PLATFORM_CONFIG.value:
                context.platform_config,
            ph.RuntimeInfoKey.STATEFUL_WORKING_DIR.value:
                context.exec_info.stateful_working_dir,
            ph.RuntimeInfoKey.EXECUTOR_OUTPUT_URI.value:
                context.exec_info.executor_output_uri,
            ph.RuntimeInfoKey.NODE_INFO.value:
                context.exec_info.pipeline_node.node_info,
            ph.RuntimeInfoKey.PIPELINE_INFO.value:
                context.exec_info.pipeline_info,
        }
    }

  def resolve(self, expression: placeholder_pb2.PlaceholderExpression) -> Any:
    """Recursively evaluates a placeholder expression."""
    if expression.HasField("value"):
      return getattr(expression.value, expression.value.WhichOneof("value"))
    elif expression.HasField("placeholder"):
      return self._resolve_placeholder(expression.placeholder)
    elif expression.HasField("operator"):
      return self._resolve_placeholder_operator(expression.operator)
    else:
      raise ValueError("Unexpected placeholder expression type: "
                       f"{expression.WhichOneof('expression_type')}.")

  def _resolve_placeholder(self,
                           placeholder: placeholder_pb2.Placeholder) -> Any:
    """Evaluates a placeholder using the contexts."""
    try:
      context = self._resolution_values[placeholder.type]
    except KeyError as e:
      raise KeyError(
          f"Unsupported placeholder type: {placeholder.type}.") from e
    try:
      return context[placeholder.key]
    except KeyError as e:
      raise KeyError(
          f"Failed to find key {placeholder.key} of placeholder type "
          f"{placeholder_pb2.Placeholder.Type.Name(placeholder.type)}.") from e

  def _resolve_placeholder_operator(
      self, placeholder_operator: placeholder_pb2.PlaceholderExpressionOperator
  ) -> Any:
    """Evaluates a placeholder operator by dispatching to the operator methods."""
    operator_name = placeholder_operator.WhichOneof("operator_type")
    operator_pb = getattr(placeholder_operator, operator_name)
    try:
      operator_fn = _PLACEHOLDER_OPERATORS[operator_pb.DESCRIPTOR.name]
    except KeyError as e:
      raise KeyError(
          f"Unsupported placeholder operator: {operator_pb.DESCRIPTOR.name}."
      ) from e
    return operator_fn(self, operator_pb)

  @_register(placeholder_pb2.ArtifactUriOperator)
  def _resolve_artifact_uri_operator(
      self, op: placeholder_pb2.ArtifactUriOperator) -> str:
    """Evaluates the artifact URI operator."""
    resolved_artifact = self.resolve(op.expression)
    if not isinstance(resolved_artifact, artifact.Artifact):
      raise ValueError("ArtifactUriOperator expects the expression "
                       "to evaluate to an artifact. "
                       f"Got {type(resolved_artifact)}")
    if op.split:
      return artifact_utils.get_split_uri([resolved_artifact], op.split)
    else:
      return resolved_artifact.uri

  @_register(placeholder_pb2.ArtifactValueOperator)
  def _resolve_artifact_value_operator(
      self, op: placeholder_pb2.ArtifactValueOperator) -> str:
    """Evaluates the artifact value operator."""
    resolved_artifact = self.resolve(op.expression)
    if not isinstance(resolved_artifact, value_artifact.ValueArtifact):
      raise ValueError("ArtifactValueOperator expects the expression "
                       "to evaluate to a value artifact."
                       f"Got {type(resolved_artifact)}")
    return resolved_artifact.read()

  @_register(placeholder_pb2.ConcatOperator)
  def _resolve_concat_operator(self, op: placeholder_pb2.ConcatOperator) -> str:
    """Evaluates the concat operator."""
    return "".join(str(self.resolve(e)) for e in op.expressions)

  @_register(placeholder_pb2.IndexOperator)
  def _resolve_index_operator(self, op: placeholder_pb2.IndexOperator) -> Any:
    """Evaluates the index operator."""
    value = self.resolve(op.expression)
    try:
      return value[op.index]
    except (TypeError, IndexError) as e:
      raise ValueError(
          f"IndexOperator failed to access the given index {op.index}.") from e

  @_register(placeholder_pb2.ProtoOperator)
  def _resolve_proto_operator(self, op: placeholder_pb2.ProtoOperator) -> str:
    """Evaluates the proto operator."""
    raw_message = self.resolve(op.expression)

    pool = descriptor_pool.Default()
    for file_descriptor in op.proto_schema.file_descriptors.file:
      pool.Add(file_descriptor)
    message_descriptor = pool.FindMessageTypeByName(
        op.proto_schema.message_type)
    factory = message_factory.MessageFactory(pool)
    message_type = factory.GetPrototype(message_descriptor)

    if isinstance(raw_message, str):
      value = message_type()
      json_format.Parse(raw_message, value, descriptor_pool=pool)
    elif isinstance(raw_message, message.Message):
      value = raw_message
    else:
      raise ValueError(
          f"Got unsupported value type for proto operator: {type(raw_message)}."
      )

    if op.proto_field_path:
      for field in op.proto_field_path:
        if field.startswith("."):
          value = getattr(value, field[1:])
          continue
        map_key = re.findall(r"\[['\"](.+)['\"]\]", field)
        if len(map_key) == 1:
          value = value[map_key[0]]
          continue
        index = re.findall(r"\[(\d+)\]", field)
        if index and str.isdecimal(index[0]):
          value = value[int(index[0])]
          continue
        raise ValueError(f"Got unsupported proto field path: {field}")
      if not isinstance(value, message.Message):
        return value
    return text_format.MessageToString(value)
