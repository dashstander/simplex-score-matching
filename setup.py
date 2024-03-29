import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplex_score_matching",
    version="0.0.1",
    author="Dashiell Stander",
    author_email="dash.stander@gmail.com",
    description="Score Matching Diffusion Models on the Probability Simplex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dashstander/simplex-score-matching",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)