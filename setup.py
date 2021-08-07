import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="turbowsr",
    version="0.0.1",
    author="Rhys Goodall",
    author_email="reag2@cam.ac.uk",
    description="quasi-relax structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/comprhys/turbowsr",
    packages=['turbowsr'],
    package_dir={'turbowsr': 'turbowsr'},
    classifiers=[
        "Programming Language :: Python :: 3.7.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)