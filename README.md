# Ramp kit Taxi-Fare-Prediction

[![Code quality](https://github.com/hugodebes/Taxi-Fare-Prediction/actions/workflows/quality.yml/badge.svg)](https://github.com/hugodebes/Taxi-Fare-Prediction/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

_Authors: Xavier Brouty, Balthazar Courvoisier, Hugo Dèbes, Yassine Hargane, Rita-Mathilda Kabro, Baptiste Pasquier_

The goal is to predict the price of a taxi using collected data from the month of may 2022.

## Set up

1. clone this repository

```
git clone https://github.com/hugodebes/Taxi-Fare-Prediction.git
cd Taxi-Fare-Prediction/
```

2. install the dependancies

- with [conda](https://conda.io/miniconda.html)

```
conda install -y -c conda conda-env     # First install conda-env
conda env create                        # Use environment.yml to create the 'taxi_fare' env
source activate taxi_fare       # Activates the virtual env
```

- without `conda` (best to use a **virtual environment**)

```
python -m pip install -r requirements.txt
```

3. get started with the storm_forecast_starting_kit.ipynb

## New submissions

1. create a new submission `<new_sub>` by building on the existing ones

```
cp -r submissions/starting_kit submissions/<new_sub>
```

2. modify the `*.py` files in `submissions/<new_sub>` with your favorite editor

3. test the submission with

```
ramp_test_submission --quick-test --submission <new_sub>
```

4. if the job complete, you can submit the code in the sandbox of [ramp.studio][ramp]

## License

BSD license : see [LICENSE file](LICENSE)

## Credits

This package was created with [Cookiecutter][cookie] and the [`ramp-kits/cookiecutter-ramp-kit`][kit] project template
issued by the [Paris-Saclay Center for Data Science][cds].

[travis]: https://travis-ci.org/ramp-kits/storm_forecast
[ramp]: https://ramp.studio/events/storm_forecast
[cookie]: https://github.com/audreyr/cookiecutter
[kit]: https://github.com/ramp-kits/cookiecutter-ramp-kit
[cds]: https://www.datascience-paris-saclay.fr/
