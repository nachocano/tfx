# Iris TFX pipeline template

This template will demonstrates the end-to-end workflow and steps of how to
classify Iris flower subspecies.

Please see [TFX on Cloud AI Platform Pipelines](
https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)
tutorial to learn how to use this template.

Use `--model iris` when copying the template. For example,
```
tfx template copy \
   --pipeline_name="${PIPELINE_NAME}" \
   --destination_path="${PROJECT_DIR}" \
   --model=iris
```

This template doesn't support BigQueryExampleGen, so it needs to be skipped.


## The dataset

This template uses the [Iris dataset](
https://archive.ics.uci.edu/ml/datasets/iris), but it is recommended to
replace this dataset with your own.


## Content of the template

The template generates three different kinds of python code which are needed to
run a TFX pipeline.

- `pipeline` directory contains *a pipeline definition* which specify what
  components to use and how the components will be interconnected.
  And it also contains various config variables to run the pipeline.
- `models` directory contains a ML model definition which is required by
  `Trainer`, `Transform` and `Tuner` components.
- The last piece is a platform specific configuration that describes physical
  paths and orchestrators. Currently we have `local_runner.py` and
  `kubeflow_runner.py`. These files are located at the root of the project
  directory.
