"""
Pipeline Compiler Script
Compiles the Vertex AI pipeline to JSON format.
"""

from kfp import compiler
from pipeline import iris_training_pipeline


def main():
    print("Compiling Vertex AI Pipeline...")
    
    compiler.Compiler().compile(
        pipeline_func=iris_training_pipeline,
        package_path='pipeline.json'
    )
    
    print("Pipeline compiled successfully!")
    print("Output: pipeline.json")


if __name__ == '__main__':
    main()
