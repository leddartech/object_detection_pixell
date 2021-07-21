from setuptools import setup

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements('requirements.txt')

setup(name="object_detection_pixell",
      version = '1.0.0',
      description='Object detection Pixell',
      author='Jean-Luc Déziel',
      author_email='jean-luc.déziel@leddartech.com',
      dependency_links=[
        "https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple/pioneer-common",
        "https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/487/packages/pypi/simple/pioneer-das-api",
      ],
      install_requires=install_reqs,
      packages=['object_detection_pixell'],
      python_requires='>=3.6',
    )