# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Always do:

At the **very start** of processing any **user message**, append the prompts to @checks/input_prompts.txt and log in the input prompt, start time and completion time in a csv called '@checks/timer.csv'
If the files do not exist then create them first. If the files pre-exist then append them.

## Repository Overview

This is a data analysis project centered on loan prediction. The repository currently contains a single dataset (`data/input.csv`) with 614 rows of loan application data.

## Dataset: `data/input.csv`

Loan application records with the following columns:

- **Loan_ID**: Unique identifier (e.g., LP001002)
- **Gender**, **Married**, **Dependents**, **Education**, **Self_Employed**: Applicant demographics
- **ApplicantIncome**, **CoapplicantIncome**: Income figures
- **LoanAmount**, **Loan_Amount_Term**: Loan details
- **Credit_History**: Binary credit history indicator
- **Property_Area**: Urban / Semiurban / Rural
- **Loan_Status**: Target variable (Y/N)

The dataset has missing values in several columns (LoanAmount, Loan_Amount_Term, Credit_History, Gender, Self_Employed, Dependents).
