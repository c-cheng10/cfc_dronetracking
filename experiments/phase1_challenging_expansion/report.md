# FRED challenging split, frozen Phase 1 evaluation

New recordings complete: 49/49. Previous test recordings: 9.
Clip 36 was used for development and is excluded.
Frozen FFT threshold: 5.2981. Frozen event count threshold: 4578.

| Recordings | Background | Score | Precision | Recall | AUC | False alarms/min | Drone windows | Background windows |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 49 new | active | FFT | 38.5% | 17.9% | 0.632 | 23.69 | 18348 | 53058 |
| 49 new | active | Count | 50.3% | 51.2% | 0.789 | 42.04 | 18348 | 53058 |
| 49 new | shifted | FFT | 76.7% | 17.9% | 0.814 | 4.49 | 18348 | 53150 |
| 49 new | shifted | Count | 84.0% | 51.2% | 0.949 | 8.05 | 18348 | 53150 |
| 58 pooled | active | FFT | 32.9% | 15.9% | 0.615 | 26.97 | 21929 | 63264 |
| 58 pooled | active | Count | 52.8% | 48.9% | 0.789 | 36.37 | 21929 | 63264 |
| 58 pooled | shifted | FFT | 75.7% | 15.9% | 0.810 | 4.23 | 21929 | 63533 |
| 58 pooled | shifted | Count | 85.7% | 48.9% | 0.951 | 6.78 | 21929 | 63533 |
