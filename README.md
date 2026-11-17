# Conversational Generalized Linear Bandits with Hybrid Feedback

Python implementation of **HyConUCB**, with experiments on synthetic data, the Yelp dataset, and LLM routing. 

## Repository Structure

| Directory | Contents |
|---|---|
| `synthetic/` | Synthetic experiments |
| `yelp/` | Experiments on the Yelp dataset |
| `llm/` | LLM routing experiments |

Each experiment contains `run.py`, `algorithms.py`, `optimization.py`, `plot.py`, and `requirements.txt`. Data construction is implemented in `synthetic/environment.py`, `yelp/data.py`, and `llm/prepare.py`.

## Installation

Use **Python 3.11**. Each experiment runs independently. From the repository root, enter the experiment directory and create a virtual environment:

```bash
cd synthetic
python -m venv .venv
```

Activate it using `source .venv/bin/activate` on Linux/macOS or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For Yelp or LLM routing, repeat these steps inside `yelp/` or `llm/`.


## Reproduce the Experiments


### Synthetic

Run the setting without an intercept, with `S = 4, 6, 8`:

```bash
python run.py --setting S
python plot.py --setting S
```

Run the setting with an intercept, with `B = 2, 4, 6`:

```bash
python run.py --setting B
python plot.py --setting B
```

### Yelp

This experiment uses restaurant-user interactions from the Yelp dataset. We selected 100 users listed in `selected_users.txt`. Before running the experiments, download and preprocess the dataset, including filtering restaurants, constructing feature vectors, fitting user vectors, and constructing user–restaurant data pairs. The reproduction procedure is as follows:

First, download the Yelp Open Dataset and place the following business and review files in the same directory:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_review.json
```


Next, preprocess the dataset, which requires a separate numerical environment. Create the preprocessing environment:

```bash
deactivate
python -m venv .venv-preprocess
```

Activate it with `source .venv-preprocess/bin/activate` on Linux/macOS or `.\.venv-preprocess\Scripts\Activate.ps1` in Windows PowerShell. Then prepare the data:

```bash
python -m pip install -r requirements-preprocess.txt
python data.py --raw "/path/to/yelp" --output data/yelp.npz
deactivate
```

Replace `/path/to/yelp` with the directory containing the two JSON files. Keep `selected_users.txt`, which selects the fixed users. Preprocessing generates `data/yelp.npz` and its metadata.

Finally, reactivate the online `.venv` environment and run:

```bash
python run.py --data data/yelp.npz
python plot.py
```

### LLM Routing

The included `data/benchmark.npz` contains features and recorded outcomes for 760 queries across ARC-Challenge, GSM8K, MATH, and MMLU. Run the experiment using this prepared benchmark:

```bash
python run.py
python plot.py
```

To regenerate figures from the included reference results without rerunning the experiment:

```bash
python plot.py --reference
```

Optionally, rebuild the benchmark from xRouteBench records and BGE embeddings:

```bash
python -m pip install -r requirements-preprocess.txt
python prepare.py --output data/rebuilt/benchmark.npz
python run.py --data data/rebuilt/benchmark.npz --output results/rebuilt
python plot.py --data data/rebuilt/benchmark.npz --result results/rebuilt --output figures/rebuilt
```

This downloads the dataset and embedding model. Raw records are saved to `data/raw/`. The commands above save the rebuilt benchmark separately in `data/rebuilt/`.

## Quick Runs

After installation and all required data preparation, use a short run to check the workflow.

From `synthetic/`, run one user and one seed for 101 rounds in each S condition:

```bash
python run.py --setting S --users 0 --seeds 0 --stop 101 --output results/smoke
python plot.py --setting S --users 0 --seeds 0 --stop 101 --result results/smoke --output figures/smoke
```

From `yelp/`, using the prepared `data/yelp.npz`:

```bash
python run.py --users 0 --seeds 0 --stop 51 --output results/smoke
python plot.py --users 0 --seeds 0 --stop 51 --result results/smoke --output figures/smoke
```

From `llm/`, using the included benchmark:

```bash
python run.py --seeds 20261201 --stop 101 --output results/smoke
python plot.py --seeds 20261201 --result results/smoke --output figures/smoke
```

## Algorithms

The experiments compare HyConUCB with Var-Random, Var-AM, ConDuel, and OFUGLB-UCB.


All methods use an empirical UCB exploration coefficient of **1.5**. Synthetic runs all five methods. Yelp and LLM support algorithm selection through `--algorithms`, using the following case-sensitive names:

- `HyConUCB`: The proposed method, combining item rewards with key-term comparisons.
- `Var-Random`: A variant of HyConUCB that selects key-term comparisons uniformly at random.
- `Var-AM`: A variant of HyConUCB that uses absolute key-term feedback.
- `ConDuel`: A conversational dueling bandit baseline that samples comparison pairs uniformly within a barycentric spanner.
- `OFUGLB-UCB`: A generalized linear bandit baseline that uses only item rewards, without conversational feedback.

## Main Parameters

The main parameters for `run.py` and `plot.py` are listed below.

- `--setting`: Selects the synthetic experiment setting, with or without an intercept, or both.
- `--users`: Selects the users to run.
- `--seeds`: Selects the random seeds to run.
- `--stop`: Sets the number of rounds to run.
- `--algorithms`: Selects algorithms.
- `--data`: Specifies the prepared data file for Yelp or LLM experiments.
- `--result`: Specifies the directory from which `plot.py` reads experiment results.
- `--output`: Specifies where `run.py` saves experiment results or where `plot.py` saves figures and summary files.
- `--reference`: Plots the included reference results without rerunning the experiment (LLM plotting only).

By default, `run.py` saves results to `results/` and `plot.py` saves figures and summaries to `figures/` within each experiment directory. 

Run `python run.py --help` or `python plot.py --help` for the complete options.
