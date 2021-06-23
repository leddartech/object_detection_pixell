from setuptools import setup

setup(name="object_detection_pixell",
      version = '1.0.0',
      description='Object detection Pixell',
      author='Jean-Luc Déziel',
      author_email='jean-luc.déziel@leddartech.com',
      requires=[
        'tqdm',
        'numpy',
        'numba==0.51.0',
        'opencv_python',
        'transforms3d',
        'ruamel.yaml',
    ],
      packages=['object_detection_pixell']
    )