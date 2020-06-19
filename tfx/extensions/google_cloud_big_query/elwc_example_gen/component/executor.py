# Lint as: python3
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
"""Generic TFX BigQueryToElwcExampleGen executor."""

import collections
from typing import Any, Dict, Iterable, Text, Tuple

import apache_beam as beam
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.extensions.google_cloud_big_query import utils
from tfx.extensions.google_cloud_big_query.elwc_example_gen.proto import elwc_config_pb2
from tfx.proto import example_gen_pb2

from google.cloud import bigquery

from google.protobuf import json_format
from tensorflow_serving.apis import input_pb2


class _BigQueryToElwcConverter(object):
  """Help class for bigquery result row to ELWC conversion."""

  def __init__(self, elwc_config: elwc_config_pb2.ElwcConfig, query: Text):
    client = bigquery.Client()
    # Dummy query to get the type information for each field.
    query_job = client.query('SELECT * FROM ({}) LIMIT 0'.format(query))
    results = query_job.result()
    self._type_map = {}
    self._context_feature_fields = set(elwc_config.context_feature_fields)
    # The namedtuple is used to as the key when grouping the context feature
    # and example feature.
    self.ContextFeature = collections.namedtuple(  # pylint: disable=invalid-name
        'ContextFeature', self._context_feature_fields)
    globals()[self.ContextFeature.__name__] = self.ContextFeature
    field_names = set()
    for field in results.schema:
      self._type_map[field.name] = field.field_type
      field_names.add(field.name)
    # Check whether the query contains necessary context fields.
    if not field_names.issuperset(self._context_feature_fields):
      raise RuntimeError('Context feature fields are missing from the query.')

  def RowToContextFeature(self, instance: Dict[Text, Any]) -> tf.train.Example:
    """Convert bigquery result to context feature."""
    context_data = dict((k, instance[k])
                        for k in instance.keys()
                        if k in self._context_feature_fields)
    context_feature = self.ContextFeature(**context_data)
    context_feature_dict = context_feature._asdict()
    return utils.row_to_example(self._type_map, context_feature_dict)

  def RowToExampleWithoutContext(self, instance: Dict[Text,
                                                      Any]) -> tf.train.Example:
    """Convert bigquery result to example without context feature."""
    example_data = dict((k, instance[k])
                        for k in instance.keys()
                        if k not in self._context_feature_fields)
    return utils.row_to_example(self._type_map, example_data)

  def CombineContextAndExamples(
      self, context_feature_and_examples: Tuple[tf.train.Example,
                                                Iterable[tf.train.Example]]
  ) -> input_pb2.ExampleListWithContext:
    """Combine context feature and examples to ELWC."""
    context_feature, examples = context_feature_and_examples
    return input_pb2.ExampleListWithContext(
        context=context_feature, examples=examples)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(input_pb2.ExampleListWithContext)
def _BigQueryToElwcExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[Text, Any],
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read from BigQuery and transform to ExampleListWithContext.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a BigQuery sql string.

  Returns:
    PCollection of ExampleListWithContext.
  """

  conn_config = example_gen_pb2.CustomConfig()
  json_format.Parse(exec_properties['custom_config'], conn_config)
  elwc_config = elwc_config_pb2.ElwcConfig()
  conn_config.custom_config.Unpack(elwc_config)
  converter = _BigQueryToElwcConverter(elwc_config, split_pattern)
  # TODO(b/155441037): Clean up the usage of `runner` flag
  # once ReadFromBigQuery performance on dataflow runner is on par
  # with BigQuerySource, and clean up `project` once beam is upgraded to 2.22.
  beam_pipeline_args = exec_properties['_beam_pipeline_args']
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      beam_pipeline_args)
  use_dataflow_runner = pipeline_options.get_all_options().get('runner') in [
      'dataflow', 'DataflowRunner'
  ]

  return (pipeline
          | 'QueryTable' >> utils.read_from_big_query_impl(  # pylint: disable=no-value-for-parameter
              query=split_pattern,
              use_bigquery_source=use_dataflow_runner)
          | 'SeparateContextAndFeature' >> beam.Map(lambda row: (  # pylint: disable=g-long-lambda
              converter.RowToContextFeature(row),
              converter.RowToExampleWithoutContext(row)))
          | 'GroupByContext' >> beam.GroupByKey()
          | 'ToElwc' >> beam.Map(converter.CombineContextAndExamples))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX BigQueryElwcExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for BigQuery to ExampleListWithContext."""
    return _BigQueryToElwcExample
