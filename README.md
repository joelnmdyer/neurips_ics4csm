# Code for NeurIPS 2024 paper "Interventionally consistent surrogates for complex simulation models"

## Install

Start and activate a new `venv` with `python3.10`, navigate to the top directory for this repo, and do `python3.10 -m pip install ./`

## Instructions for running code
Navigate to `neurips_ics4csm/` and issue (variables explained below)

```
python job_script.py --family FAMILY --seed SEEDS --dirname DIRNAME
```

or

```
python job_script_bwsg.py --family FAMILY --seed SEEDS --dirname DIRNAME
```

In the above, set
- `FAMILY` to one of `lode`, `lodernn`, `lrnn`
- `SEED` to any sequence of seeds (e.g., `0 1 3` to run code three times at seed `0`, `1`, and `3`)
- `DIRNAME` to the folder in which you'd like to dump the output of the script

Omitting the `--seeds` argument will make the seeds default to `list(range(5))`, which is what was used to generate the 5-fold cross-validation results reported in the paper.

## Citation

```
@inproceedings{dyer2024a,
  publisher = {Neural Information Processing Systems Foundation},
  title = {Interventionally consistent surrogates for complex simulation models},
  author = {Dyer, J and Bishop, N and Felekis, Y and Zennaro, FM and Calinescu, A and Damoulas, T and Wooldridge, M},
  year = {2024},
  organizer = {Neural Information Processing Systems (NeurIPS 2024)}
}
```
