| Method | Variant | Runs | mean(R/N), median [IQR] | Total R, median [IQR] |
|---|---|---:|---:|---:|
| Leveling | Deterministic | 1 | 1.0444 | 36,024 |
| GP | C | 30 | 0.7548 [0.7535, 0.7574] | 24,772.0 [24,681.5, 24,871.5] |
| GP | C+E | 30 | 0.7588 [0.7565, 0.7618] | 24,947.0 [24,768.2, 25,057.0] |
| GP | C+O | 30 | 0.7622 [0.7597, 0.7642] | 25,045.5 [24,949.8, 25,149.2] |
| GP | C+E+O | 30 | 0.7595 [0.7561, 0.7635] | 24,969.0 [24,861.5, 25,061.8] |
| MoE-LoRA | Dynamic routing | 3 | 0.7343 [0.7343, 0.7344] | 24,027.0 [24,027.0, 24,027.5] |

Lower is better. Results use the same 840-instance Caserta manifest at H=20; Leveling is horizon-independent. GP uses 30 independent runs per terminal set, MoE-LoRA uses three formal seeds, and Leveling is deterministic. No seed is selected using Caserta performance.
