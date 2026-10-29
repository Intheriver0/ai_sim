# Tuning and blending

Past-only selection; reference G1/G2 fixed. NMAE is a proxy; W covers only complete actuals.

| Model | Month-equal NMAE | Worst month NMAE | Complete hours | W (won) |
|---|---:|---:|---:|---:|
| fixed | 3.9140% | 5.0920% | 178 | 2,568,375 |
| selected | 3.9027% | 4.9956% | 178 | 2,259,863 |
| blend | 3.8804% | 5.0058% | 178 | 2,304,131 |

| Outer month | Selected model | Fixed weight | Fixed NMAE | Selected NMAE | Blend NMAE |
|---|---|---:|---:|---:|---:|
| outer_2024_06 | tuned | 0.40 | 4.5715% | 4.4738% | 4.4478% |
| outer_2024_07 | tuned | 0.25 | 5.0920% | 4.9956% | 5.0058% |
| outer_2024_08 | tuned | 0.35 | 4.3048% | 4.2576% | 4.2423% |
| outer_2024_09 | tuned | 0.35 | 4.4528% | 4.4847% | 4.4310% |
| outer_2024_10 | tuned | 0.30 | 3.5322% | 3.5890% | 3.5477% |
| outer_2024_11 | tuned | 0.20 | 2.9318% | 2.9506% | 2.9391% |
| outer_2024_12 | tuned | 0.20 | 2.5132% | 2.5674% | 2.5493% |
