# zeiss_ds_challenge
Repo of zeiss data science challenge.

## Summary
I assumed that the task is to build a lead generator for e.g. sale mangers. Because of the specialization features I assumed that each record in the data set is a potential doctor's office which might be a potential customer of microscopes.


Furthermore, I imagined the problem to be a prioritization task, because a sales manager might have only limited resources and only want to contact potential customers with a high probability to become a customer.

Thus, the idea was to come up with a list of potential customers ordered by expected probability of becoming a customer.
Unfortunately, this did not work out because the classifiers are not really beating simply predicting the true class.

## Important files of the repo
- **notebooks/01-JB-EDA.ipynb** --> contains the conducted EDA steps including the cell to create pandas profiling reports
- **notebooks/02-JB-Model-Eval.ipynb** --> contains a small evaluation of the final results
- **src/train.py** --> contains the pipeline to create the experiments incl a mlflow logging

## Rerun experiments from command line
- Clone repo
- Create conda environment from environment.yaml
- Activate environment
- **cd** to **src** directory and run **python src/train.py**
- Run MLflow’s Tracking UI from **src**: **mlflow ui** and view it at http://localhost:5000.
