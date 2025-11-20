# Practical Statistics for Data Scientists (Python Edition)

Code associated with the book "Practical Statistics for Data Scientists: 50 Essential Concepts", translated to Python.

The Python scripts are stored by chapter in the `python_src` directory and replicate most of the figures and code snippets from the original R code.

## HOW TO USE

### Data
The data is not stored on GitHub. You will need to download it first.

You can do this by running the Python script:

```bash
python python_src/download_data.py
```

This will download the necessary data files into a `data` directory inside the project folder.

### Dependencies
The scripts require several Python libraries. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

### Running the scripts
The scripts assume that the repository has been cloned into your home directory (`~/statistics-for-data-scientists`). If you save the repository elsewhere, you will need to edit the following line in each script:

```python
PSDS_PATH = os.path.join(os.path.expanduser('~'), 'statistics-for-data-scientists')
```

to point to the correct directory path.

The original R scripts for this book can be found at the original author's repository: https://github.com/gedeck/practical-statistics-for-data-scientists
