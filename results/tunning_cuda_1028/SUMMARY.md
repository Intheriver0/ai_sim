# Tuning and blending

Past-only selection; reference G1/G2 fixed. NMAE is a proxy; W covers only complete actuals.

| Model | Month-equal NMAE | Worst month NMAE | Complete hours | W (won) |
|---|---:|---:|---:|---:|
| fixed | 3.6403% | 5.0920% | 3802 | 89,028,704 |
| selected | 3.6440% | 4.9956% | 3802 | 85,297,128 |
| blend | 3.6223% | 5.0058% | 3802 | 86,501,385 |

| Outer month | Selected model | Fixed weight | Fixed NMAE | Selected NMAE | Blend NMAE |
|---|---|---:|---:|---:|---:|
| outer_2024_06 | tuned | 0.40 | 4.5715% | 4.4738% | 4.4478% |
| outer_2024_07 | tuned | 0.25 | 5.0920% | 4.9956% | 5.0058% |
| outer_2024_08 | tuned | 0.35 | 4.3048% | 4.2576% | 4.2423% |
| outer_2024_09 | tuned | 0.35 | 4.4528% | 4.4847% | 4.4310% |
| outer_2024_10 | tuned | 0.30 | 3.5322% | 3.5890% | 3.5477% |
| outer_2024_11 | tuned | 0.20 | 2.9318% | 2.9506% | 2.9391% |
| outer_2024_12 | tuned | 0.20 | 2.5132% | 2.5674% | 2.5493% |
| outer_2025_01 | tuned | 0.30 | 2.4917% | 2.5231% | 2.5025% |
| outer_2025_02 | tuned | 0.35 | 2.9782% | 3.1512% | 3.0707% |
| outer_2025_03 | tuned | 0.50 | 3.1916% | 3.2201% | 3.1925% |
| outer_2025_04 | tuned | 0.55 | 3.5534% | 3.5186% | 3.5219% |
| outer_2025_05 | tuned | 0.50 | 4.0705% | 3.9960% | 4.0172% |
