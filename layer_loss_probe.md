# Layer Loss Delta Probe

- Generated: `2026-04-02 16:37:10 UTC`
- Model: `/root/checkpoints/table_enrich_27B_with_base_input_0318`
- Dataset: `/root/data/val.json`
- Example indices: `[0, 1, 2, 3, 4]`
- Baseline weighted mean loss: `0.012505`
- Baseline sample mean loss: `0.021458`
- Total target tokens: `4156`
- Layer filter: `all`
- Include endpoints: `False`
- Full-attention interval: `4`

## Baseline Per Example

| dataset_idx | loss |
| --- | --- |
| 0 | 0.002117 |
| 1 | 0.067110 |
| 2 | 0.007971 |
| 3 | 0.026185 |
| 4 | 0.003905 |

## Evaluation Order

| layer | type | ablated_loss | delta_loss | delta_pct | elapsed_s |
| --- | --- | --- | --- | --- | --- |
| 1 | linear_attention | 0.024661 | +0.012157 | +97.215% | 1.2 |
| 2 | linear_attention | 0.013458 | +0.000953 | +7.621% | 1.2 |
| 3 | full_attention | 0.013203 | +0.000698 | +5.583% | 1.2 |
| 4 | linear_attention | 0.013018 | +0.000513 | +4.100% | 1.2 |
| 5 | linear_attention | 0.012111 | -0.000394 | -3.153% | 1.2 |
| 6 | linear_attention | 0.012958 | +0.000453 | +3.624% | 1.2 |
| 7 | full_attention | 0.012391 | -0.000114 | -0.911% | 1.2 |
| 8 | linear_attention | 0.013110 | +0.000605 | +4.838% | 1.2 |
| 9 | linear_attention | 0.013527 | +0.001023 | +8.178% | 1.2 |
| 10 | linear_attention | 0.013088 | +0.000583 | +4.665% | 1.2 |
| 11 | full_attention | 0.013122 | +0.000618 | +4.939% | 1.2 |
| 12 | linear_attention | 0.012995 | +0.000490 | +3.921% | 1.2 |
| 13 | linear_attention | 0.012828 | +0.000323 | +2.584% | 1.2 |
| 14 | linear_attention | 0.012577 | +0.000072 | +0.575% | 1.2 |
| 15 | full_attention | 0.012155 | -0.000349 | -2.794% | 1.2 |
| 16 | linear_attention | 0.012802 | +0.000297 | +2.379% | 1.2 |
| 17 | linear_attention | 0.012525 | +0.000021 | +0.164% | 1.2 |
| 18 | linear_attention | 0.013694 | +0.001190 | +9.513% | 1.2 |
| 19 | full_attention | 0.012321 | -0.000184 | -1.473% | 1.2 |
| 20 | linear_attention | 0.013299 | +0.000794 | +6.350% | 1.2 |
| 21 | linear_attention | 0.012837 | +0.000332 | +2.652% | 1.2 |
| 22 | linear_attention | 0.012758 | +0.000253 | +2.025% | 1.2 |
| 23 | full_attention | 0.012228 | -0.000277 | -2.218% | 1.2 |
| 24 | linear_attention | 0.012580 | +0.000075 | +0.598% | 1.2 |
| 25 | linear_attention | 0.013316 | +0.000811 | +6.489% | 1.2 |
| 26 | linear_attention | 0.013553 | +0.001048 | +8.380% | 1.2 |
| 27 | full_attention | 0.014814 | +0.002309 | +18.467% | 1.2 |
| 28 | linear_attention | 0.014389 | +0.001884 | +15.066% | 1.2 |
| 29 | linear_attention | 0.013760 | +0.001255 | +10.039% | 1.2 |
| 30 | linear_attention | 0.014218 | +0.001713 | +13.696% | 1.2 |
| 31 | full_attention | 0.013600 | +0.001095 | +8.757% | 1.2 |
| 32 | linear_attention | 0.012769 | +0.000264 | +2.110% | 1.2 |
| 33 | linear_attention | 0.013585 | +0.001080 | +8.637% | 1.2 |
| 34 | linear_attention | 0.013956 | +0.001451 | +11.602% | 1.2 |
| 35 | full_attention | 0.016726 | +0.004221 | +33.754% | 1.2 |
| 36 | linear_attention | 0.012658 | +0.000153 | +1.224% | 1.2 |
| 37 | linear_attention | 0.013468 | +0.000963 | +7.698% | 1.2 |
| 38 | linear_attention | 0.013244 | +0.000739 | +5.910% | 1.2 |
| 39 | full_attention | 0.014271 | +0.001766 | +14.126% | 1.2 |
| 40 | linear_attention | 0.013700 | +0.001195 | +9.558% | 1.2 |
| 41 | linear_attention | 0.013039 | +0.000534 | +4.270% | 1.2 |
| 42 | linear_attention | 0.035939 | +0.023434 | +187.399% | 1.2 |
| 43 | full_attention | 0.016409 | +0.003904 | +31.223% | 1.2 |
| 44 | linear_attention | 0.014005 | +0.001501 | +12.000% | 1.2 |
| 45 | linear_attention | 0.014433 | +0.001928 | +15.419% | 1.2 |
| 46 | linear_attention | 0.014294 | +0.001789 | +14.308% | 1.2 |
| 47 | full_attention | 0.020041 | +0.007536 | +60.268% | 1.2 |
| 48 | linear_attention | 0.014929 | +0.002424 | +19.383% | 1.2 |
| 49 | linear_attention | 0.013493 | +0.000988 | +7.902% | 1.2 |
| 50 | linear_attention | 0.013446 | +0.000942 | +7.530% | 1.2 |
| 51 | full_attention | 0.023633 | +0.011129 | +88.994% | 1.2 |
| 52 | linear_attention | 0.014440 | +0.001935 | +15.478% | 1.2 |
| 53 | linear_attention | 0.014744 | +0.002239 | +17.905% | 1.2 |
| 54 | linear_attention | 0.014402 | +0.001897 | +15.171% | 1.2 |
| 55 | full_attention | 0.012259 | -0.000246 | -1.967% | 1.2 |
| 56 | linear_attention | 0.013622 | +0.001117 | +8.930% | 1.2 |
| 57 | linear_attention | 0.013602 | +0.001097 | +8.776% | 1.2 |
| 58 | linear_attention | 0.014817 | +0.002312 | +18.490% | 1.2 |
| 59 | full_attention | 0.016913 | +0.004408 | +35.252% | 1.2 |
| 60 | linear_attention | 0.014014 | +0.001509 | +12.071% | 1.2 |
| 61 | linear_attention | 0.017337 | +0.004832 | +38.641% | 1.2 |
| 62 | linear_attention | 0.026034 | +0.013529 | +108.194% | 1.2 |

## Safest To Drop

| rank | layer | type | delta_loss | ablated_loss |
| --- | --- | --- | --- | --- |
| 1 | 5 | linear_attention | -0.000394 | 0.012111 |
| 2 | 15 | full_attention | -0.000349 | 0.012155 |
| 3 | 23 | full_attention | -0.000277 | 0.012228 |
| 4 | 55 | full_attention | -0.000246 | 0.012259 |
| 5 | 19 | full_attention | -0.000184 | 0.012321 |
| 6 | 7 | full_attention | -0.000114 | 0.012391 |
| 7 | 17 | linear_attention | +0.000021 | 0.012525 |
| 8 | 14 | linear_attention | +0.000072 | 0.012577 |
| 9 | 24 | linear_attention | +0.000075 | 0.012580 |
| 10 | 36 | linear_attention | +0.000153 | 0.012658 |
| 11 | 22 | linear_attention | +0.000253 | 0.012758 |
| 12 | 32 | linear_attention | +0.000264 | 0.012769 |
| 13 | 16 | linear_attention | +0.000297 | 0.012802 |
| 14 | 13 | linear_attention | +0.000323 | 0.012828 |
| 15 | 21 | linear_attention | +0.000332 | 0.012837 |
| 16 | 6 | linear_attention | +0.000453 | 0.012958 |
| 17 | 12 | linear_attention | +0.000490 | 0.012995 |
| 18 | 4 | linear_attention | +0.000513 | 0.013018 |
| 19 | 41 | linear_attention | +0.000534 | 0.013039 |
| 20 | 10 | linear_attention | +0.000583 | 0.013088 |
| 21 | 8 | linear_attention | +0.000605 | 0.013110 |
| 22 | 11 | full_attention | +0.000618 | 0.013122 |
| 23 | 3 | full_attention | +0.000698 | 0.013203 |
| 24 | 38 | linear_attention | +0.000739 | 0.013244 |
| 25 | 20 | linear_attention | +0.000794 | 0.013299 |
| 26 | 25 | linear_attention | +0.000811 | 0.013316 |
| 27 | 50 | linear_attention | +0.000942 | 0.013446 |
| 28 | 2 | linear_attention | +0.000953 | 0.013458 |
| 29 | 37 | linear_attention | +0.000963 | 0.013468 |
| 30 | 49 | linear_attention | +0.000988 | 0.013493 |
| 31 | 9 | linear_attention | +0.001023 | 0.013527 |
| 32 | 26 | linear_attention | +0.001048 | 0.013553 |
| 33 | 33 | linear_attention | +0.001080 | 0.013585 |
| 34 | 31 | full_attention | +0.001095 | 0.013600 |
| 35 | 57 | linear_attention | +0.001097 | 0.013602 |
| 36 | 56 | linear_attention | +0.001117 | 0.013622 |
| 37 | 18 | linear_attention | +0.001190 | 0.013694 |
| 38 | 40 | linear_attention | +0.001195 | 0.013700 |
| 39 | 29 | linear_attention | +0.001255 | 0.013760 |
| 40 | 34 | linear_attention | +0.001451 | 0.013956 |
| 41 | 44 | linear_attention | +0.001501 | 0.014005 |
| 42 | 60 | linear_attention | +0.001509 | 0.014014 |
| 43 | 30 | linear_attention | +0.001713 | 0.014218 |
| 44 | 39 | full_attention | +0.001766 | 0.014271 |
| 45 | 46 | linear_attention | +0.001789 | 0.014294 |
| 46 | 28 | linear_attention | +0.001884 | 0.014389 |
| 47 | 54 | linear_attention | +0.001897 | 0.014402 |
| 48 | 45 | linear_attention | +0.001928 | 0.014433 |
| 49 | 52 | linear_attention | +0.001935 | 0.014440 |
| 50 | 53 | linear_attention | +0.002239 | 0.014744 |
| 51 | 27 | full_attention | +0.002309 | 0.014814 |
| 52 | 58 | linear_attention | +0.002312 | 0.014817 |
| 53 | 48 | linear_attention | +0.002424 | 0.014929 |
| 54 | 43 | full_attention | +0.003904 | 0.016409 |
| 55 | 35 | full_attention | +0.004221 | 0.016726 |
| 56 | 59 | full_attention | +0.004408 | 0.016913 |
| 57 | 61 | linear_attention | +0.004832 | 0.017337 |
| 58 | 47 | full_attention | +0.007536 | 0.020041 |
| 59 | 51 | full_attention | +0.011129 | 0.023633 |
| 60 | 1 | linear_attention | +0.012157 | 0.024661 |
| 61 | 62 | linear_attention | +0.013529 | 0.026034 |
| 62 | 42 | linear_attention | +0.023434 | 0.035939 |

## Most Important

| rank | layer | type | delta_loss | ablated_loss |
| --- | --- | --- | --- | --- |
| 1 | 42 | linear_attention | +0.023434 | 0.035939 |
| 2 | 62 | linear_attention | +0.013529 | 0.026034 |
| 3 | 1 | linear_attention | +0.012157 | 0.024661 |
| 4 | 51 | full_attention | +0.011129 | 0.023633 |
| 5 | 47 | full_attention | +0.007536 | 0.020041 |
| 6 | 61 | linear_attention | +0.004832 | 0.017337 |
| 7 | 59 | full_attention | +0.004408 | 0.016913 |
| 8 | 35 | full_attention | +0.004221 | 0.016726 |
| 9 | 43 | full_attention | +0.003904 | 0.016409 |
| 10 | 48 | linear_attention | +0.002424 | 0.014929 |
| 11 | 58 | linear_attention | +0.002312 | 0.014817 |
| 12 | 27 | full_attention | +0.002309 | 0.014814 |
| 13 | 53 | linear_attention | +0.002239 | 0.014744 |
| 14 | 52 | linear_attention | +0.001935 | 0.014440 |
| 15 | 45 | linear_attention | +0.001928 | 0.014433 |
| 16 | 54 | linear_attention | +0.001897 | 0.014402 |
| 17 | 28 | linear_attention | +0.001884 | 0.014389 |
| 18 | 46 | linear_attention | +0.001789 | 0.014294 |
| 19 | 39 | full_attention | +0.001766 | 0.014271 |
| 20 | 30 | linear_attention | +0.001713 | 0.014218 |
| 21 | 60 | linear_attention | +0.001509 | 0.014014 |
| 22 | 44 | linear_attention | +0.001501 | 0.014005 |
| 23 | 34 | linear_attention | +0.001451 | 0.013956 |
| 24 | 29 | linear_attention | +0.001255 | 0.013760 |
| 25 | 40 | linear_attention | +0.001195 | 0.013700 |
| 26 | 18 | linear_attention | +0.001190 | 0.013694 |
| 27 | 56 | linear_attention | +0.001117 | 0.013622 |
| 28 | 57 | linear_attention | +0.001097 | 0.013602 |
| 29 | 31 | full_attention | +0.001095 | 0.013600 |
| 30 | 33 | linear_attention | +0.001080 | 0.013585 |
| 31 | 26 | linear_attention | +0.001048 | 0.013553 |
| 32 | 9 | linear_attention | +0.001023 | 0.013527 |
| 33 | 49 | linear_attention | +0.000988 | 0.013493 |
| 34 | 37 | linear_attention | +0.000963 | 0.013468 |
| 35 | 2 | linear_attention | +0.000953 | 0.013458 |
| 36 | 50 | linear_attention | +0.000942 | 0.013446 |
| 37 | 25 | linear_attention | +0.000811 | 0.013316 |
| 38 | 20 | linear_attention | +0.000794 | 0.013299 |
| 39 | 38 | linear_attention | +0.000739 | 0.013244 |
| 40 | 3 | full_attention | +0.000698 | 0.013203 |
| 41 | 11 | full_attention | +0.000618 | 0.013122 |
| 42 | 8 | linear_attention | +0.000605 | 0.013110 |
| 43 | 10 | linear_attention | +0.000583 | 0.013088 |
| 44 | 41 | linear_attention | +0.000534 | 0.013039 |
| 45 | 4 | linear_attention | +0.000513 | 0.013018 |
| 46 | 12 | linear_attention | +0.000490 | 0.012995 |
| 47 | 6 | linear_attention | +0.000453 | 0.012958 |
| 48 | 21 | linear_attention | +0.000332 | 0.012837 |
| 49 | 13 | linear_attention | +0.000323 | 0.012828 |
| 50 | 16 | linear_attention | +0.000297 | 0.012802 |
| 51 | 32 | linear_attention | +0.000264 | 0.012769 |
| 52 | 22 | linear_attention | +0.000253 | 0.012758 |
| 53 | 36 | linear_attention | +0.000153 | 0.012658 |
| 54 | 24 | linear_attention | +0.000075 | 0.012580 |
| 55 | 14 | linear_attention | +0.000072 | 0.012577 |
| 56 | 17 | linear_attention | +0.000021 | 0.012525 |
| 57 | 7 | full_attention | -0.000114 | 0.012391 |
| 58 | 19 | full_attention | -0.000184 | 0.012321 |
| 59 | 55 | full_attention | -0.000246 | 0.012259 |
| 60 | 23 | full_attention | -0.000277 | 0.012228 |
| 61 | 15 | full_attention | -0.000349 | 0.012155 |
| 62 | 5 | linear_attention | -0.000394 | 0.012111 |

## GDN / Linear Attention Only

| rank | layer | delta_loss | ablated_loss |
| --- | --- | --- | --- |
| 1 | 5 | -0.000394 | 0.012111 |
| 2 | 17 | +0.000021 | 0.012525 |
| 3 | 14 | +0.000072 | 0.012577 |
| 4 | 24 | +0.000075 | 0.012580 |
| 5 | 36 | +0.000153 | 0.012658 |
| 6 | 22 | +0.000253 | 0.012758 |
| 7 | 32 | +0.000264 | 0.012769 |
| 8 | 16 | +0.000297 | 0.012802 |
| 9 | 13 | +0.000323 | 0.012828 |
| 10 | 21 | +0.000332 | 0.012837 |
| 11 | 6 | +0.000453 | 0.012958 |
| 12 | 12 | +0.000490 | 0.012995 |
| 13 | 4 | +0.000513 | 0.013018 |
| 14 | 41 | +0.000534 | 0.013039 |
| 15 | 10 | +0.000583 | 0.013088 |
| 16 | 8 | +0.000605 | 0.013110 |
| 17 | 38 | +0.000739 | 0.013244 |
| 18 | 20 | +0.000794 | 0.013299 |
| 19 | 25 | +0.000811 | 0.013316 |
| 20 | 50 | +0.000942 | 0.013446 |
| 21 | 2 | +0.000953 | 0.013458 |
| 22 | 37 | +0.000963 | 0.013468 |
| 23 | 49 | +0.000988 | 0.013493 |
| 24 | 9 | +0.001023 | 0.013527 |
| 25 | 26 | +0.001048 | 0.013553 |
| 26 | 33 | +0.001080 | 0.013585 |
| 27 | 57 | +0.001097 | 0.013602 |
| 28 | 56 | +0.001117 | 0.013622 |
| 29 | 18 | +0.001190 | 0.013694 |
| 30 | 40 | +0.001195 | 0.013700 |
| 31 | 29 | +0.001255 | 0.013760 |
| 32 | 34 | +0.001451 | 0.013956 |
| 33 | 44 | +0.001501 | 0.014005 |
| 34 | 60 | +0.001509 | 0.014014 |
| 35 | 30 | +0.001713 | 0.014218 |
| 36 | 46 | +0.001789 | 0.014294 |
| 37 | 28 | +0.001884 | 0.014389 |
| 38 | 54 | +0.001897 | 0.014402 |
| 39 | 45 | +0.001928 | 0.014433 |
| 40 | 52 | +0.001935 | 0.014440 |
| 41 | 53 | +0.002239 | 0.014744 |
| 42 | 58 | +0.002312 | 0.014817 |
| 43 | 48 | +0.002424 | 0.014929 |
| 44 | 61 | +0.004832 | 0.017337 |
| 45 | 1 | +0.012157 | 0.024661 |
| 46 | 62 | +0.013529 | 0.026034 |
| 47 | 42 | +0.023434 | 0.035939 |

## Best 1 GDN Layer Per Attention Block

This is the direct shortlist for the `64L -> 48L` style plan: pick the lowest-delta linear/GDN layer inside each full-attention interval block.

| block | gdn_candidates | pick | delta_loss |
| --- | --- | --- | --- |
| 0-3 | 1, 2 | 2 | +0.000953 |
| 4-7 | 4, 5, 6 | 5 | -0.000394 |
| 8-11 | 8, 9, 10 | 10 | +0.000583 |
| 12-15 | 12, 13, 14 | 14 | +0.000072 |
| 16-19 | 16, 17, 18 | 17 | +0.000021 |
| 20-23 | 20, 21, 22 | 22 | +0.000253 |
| 24-27 | 24, 25, 26 | 24 | +0.000075 |
| 28-31 | 28, 29, 30 | 29 | +0.001255 |
| 32-35 | 32, 33, 34 | 32 | +0.000264 |
| 36-39 | 36, 37, 38 | 36 | +0.000153 |
| 40-43 | 40, 41, 42 | 41 | +0.000534 |
| 44-47 | 44, 45, 46 | 44 | +0.001501 |
| 48-51 | 48, 49, 50 | 50 | +0.000942 |
| 52-55 | 52, 53, 54 | 54 | +0.001897 |
| 56-59 | 56, 57, 58 | 57 | +0.001097 |
| 60-63 | 60, 61, 62 | 60 | +0.001509 |

Suggested layer list: `[2, 5, 10, 14, 17, 22, 24, 29, 32, 36, 41, 44, 50, 54, 57, 60]`
