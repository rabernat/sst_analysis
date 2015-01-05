from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='sstanalysis',
      version='0.1',
      description='Tools for calculating analyzing SST.',
      url='https://bitbucket.org/ryanaberanthey/sst_analysis',
      author='Ryan Abernathey',
      author_email='rpa@ldeo.columbia.edu',
      license='MIT',
      packages=['sstanalysis'],
      install_requires=[
          'numpy','netCDF4'
      ],
      test_suite = 'nose.collector',
      zip_safe=False)
