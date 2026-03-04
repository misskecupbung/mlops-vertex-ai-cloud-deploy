"""Compile Vertex AI pipeline to JSON."""

from kfp import compiler
from pipeline import iris_training_pipeline


if __name__ == '__main__':
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=iris_training_pipeline,
        package_path='pipeline.json'
    )
    print("Done -> pipeline.json")
