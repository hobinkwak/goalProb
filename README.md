# goalProb
Estimation of Goal Achieving Probabilities

## Overview
- Using transition probabilities, it estimates the **Expected Wealth paths** and the **probability of achieving the goals** from a specific point in time (t0) to the goal achievement time (T).
  - For instance, it can estimate the probability that wealth of around $100 in January 2020 will exceed $250 by December 2022.

- All you need is **asset return panel data** and **weights panel data** 
- check *tutorial.ipynb*

## Example
- Portfolio: All Weather Portfolio (SPY, TLT, IEF, GLD, DBC)
- Assume an investor wants to know the probability that the wealth, which was 200 in January 2020, will be 220 or 250 in December 2021
  - no additional cashflows excluding initial wealth
  - you can consider cashflows plans using `model.add_cashflow()` method.
### Result

| *Start Date* | *End Date* | *Wealth(start)* | *Wealth(end)* | *Probability* |
|:------------:|:----------:|:---------------:|:-------------:|:-------------:|
|  2020-01-31  | 2021-12-31 |       200       |      220      |     0.5606  |
|  2020-01-31  |    2021-12-31    |       200       |      250      |  0.3607 |


## Requirements
- Python >= 3.8
- numpy
- pandas
- scipy


## reference
- Paper Link([original](https://srdas.github.io/Papers/DP_Paper.pdf), [multi](https://srdas.github.io/Papers/MultWealthGoals.pdf))
   