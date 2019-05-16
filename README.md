# Transacções Paywall não concluídas

### Main Objective

Obtain all paywall transactions not concluded in d-1 and send them via e-mail channel.

#### Structure

This repository is structured in the following way:

**config**
- config.py - Includes all necessary parameters(emails, database, host, receivers, etc)

**files**
- transaction_data.sql - SQL query which run in the database


**main files**
- functions.py - Included all necessary functions to connect to database, save csv and send emails
- send_email.py - main file

Quick Start
-
1 - Get started:
```
$ git clone <repository.git>
$ cd inc_transactions
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
