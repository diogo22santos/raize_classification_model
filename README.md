# Raize Default Classification Package

### Main Objective
Using Machine Learning modes for classification of loans for corporates

#### Structure

This repository is structured in the following way:

```bash
├── notebooks
│   ├── jupyter notebooks                           # Exploration, engineering & modelling
│   ├── train and test datasets
│   ├── other relevant files from research phase    # html reports, .csv, .np)
├── packages
│   ├── raize_ml_model
│   │   ├── config                                  # Config file with directory variables & model variables
│   │   ├── datasets                                # Train and test set)
│   │   ├── processing                              # Feature engineering classes and fuctions, data management
│   │   ├── trained models                          # Serialized model pkl
│   ├── ml_api                                      # Future commit for REST Service)
│   ├── tests                                       # Test files
│   │   ├── unit
│   │   ├── integration
│   ├── pipeline.py                                 
│   ├── train_pipeline.py                           
│   ├── predict.py                                  
├── README.md
├── requirements.txt
```

Quick Start for pushing to local machine
-
1 - Get started:
```
$ git clone <repository.git>
$ cd raize_classification
```

2 - Create virtual environment
```
$ python -m venv <ENVIRONMENT_NAME>
$ ENVIRONMENT_NAME\Scripts\activate
```

3 - Install requirements
```
$ pip install -r requirements.txt
```

4 - Modifications

Each modification such as, addition of new features, changing the training data, experiment new alghoritms shall be firstly tested on notebooks folder, and only then changed in the packages sections.

5 - Testing
```
$ pytest packages\tests\test_predict.py
```

6 - Pushing to central repository
```
$ git checkout <branch_repository.git>
$ git add .
$ git commit -m "Addition of new feature [feature_name]"
$ git push
```