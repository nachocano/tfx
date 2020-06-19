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
# Create this instead of inline in _BigQueryToExample for test mocking purpose.
"""Utilities for Google Cloud BigQuery TFX extensions."""

from typing import Any, Dict, Text

import apache_beam as beam
from apache_beam.io.gcp.bigquery import ReadFromBigQuery
import tensorflow as tf


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.typehints.Dict[Text, Any])
def read_from_big_query_impl(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    query: Text,
    use_bigquery_source: bool = False) -> beam.pvalue.PCollection:
  """Read from BigQuery.

  Args:
    pipeline: Beam pipeline.
    query: A BigQuery sql string.
    use_bigquery_source: whether to use BigQuerySource instead of experimental
      `ReadFromBigQuery` PTransform.

  Returns:
    PCollection of dict.
  """
  # TODO(b/155441037): Consolidate to ReadFromBigQuery once its performance
  # on dataflow runner is on par with BigQuerySource.
  if use_bigquery_source:
    return (pipeline
            | 'ReadFromBigQuerySource' >> beam.io.Read(
                beam.io.BigQuerySource(query=query, use_standard_sql=True)))
  # TODO(b/155441037): Consolidate to ReadFromBigQuery once its performance
  # on dataflow runner is on par with BigQuerySource.
  return (pipeline
          | 'ReadFromBigQuery' >> ReadFromBigQuery(
              query=query, use_standard_sql=True))


def row_to_example(field_to_type: Dict[Text, Text],
                   field_name_to_data: Dict[Text, Any]) -> tf.train.Example:
  """Convert bigquery result row to tf example.

  Args:
    field_to_type: The name of the field to its type from BigQuery.
    field_name_to_data: The data need to be converted from BigQuery that
      contains field name and data.

  Returns:
    A tf.train.Example that converted from the BigQuery row.

  Raises:
    RuntimeError: If the data type is not supported to be converted.
      Only INTEGER, BOOLEAN, FLOAT, STRING is supported now.
  """
  feature = {}
  for key, value in field_name_to_data.items():
    data_type = field_to_type[key]

    if value is None:
      feature[key] = tf.train.Feature()
      continue

    value_list = value if isinstance(value, list) else [value]
    if data_type in ('INTEGER', 'BOOLEAN'):
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=value_list))
    elif data_type == 'FLOAT':
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=value_list))
    elif data_type == 'STRING':
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[tf.compat.as_bytes(elem) for elem in value_list]))
    else:
      # TODO(jyzhao): support more types.
      raise RuntimeError(
          'BigQuery column type {} is not supported.'.format(data_type))

  return tf.train.Example(features=tf.train.Features(feature=feature))
