| Experimento | Modelo | LR | LoRA r | Épocas | Treino (s) | VRAM pico (GB) | Temp. | Acurácia | F1 macro | Inválidas |
|---|---|---|---|---|---|---|---|---|---|---|
| ft_lr2e-4_r16_ep3 | Qwen2.5-1.5B-Instruct | 0.0002 | 16 | 3.0 | 901.2 | 7.2 | 0.0 | 0.827 | 0.827 | 0 |
| ft_lr2e-4_r16_ep1 | Qwen2.5-1.5B-Instruct | 0.0002 | 16 | 1.0 | 346.5 | 7.2 | 0.0 | 0.827 | 0.825 | 0 |
| ft_qwen0.5b_lr2e-4_r16_ep1 | Qwen2.5-0.5B-Instruct | 0.0002 | 16 | 1.0 | 210.3 | 5.04 | 0.0 | 0.819 | 0.818 | 0 |
| ft_lr2e-4_r4_ep1 | Qwen2.5-1.5B-Instruct | 0.0002 | 4 | 1.0 | 302.4 | 7.04 | 0.0 | 0.818 | 0.817 | 0 |
| ft_lr5e-5_r16_ep1 | Qwen2.5-1.5B-Instruct | 5e-05 | 16 | 1.0 | 345.8 | 7.2 | 0.0 | 0.817 | 0.815 | 0 |
| ft_lr1e-3_r16_ep1 | Qwen2.5-1.5B-Instruct | 0.001 | 16 | 1.0 | 340.8 | 7.2 | 0.0 | 0.814 | 0.813 | 0 |
| ft_lr2e-4_r64_ep1 | Qwen2.5-1.5B-Instruct | 0.0002 | 64 | 1.0 | 348.3 | 7.82 | 0.0 | 0.812 | 0.811 | 0 |
| ft_lr2e-4_r16_ep1_temp1.0 | Qwen2.5-1.5B-Instruct | 0.0002 | 16 | 1.0 | 346.5 | 7.2 | 1.0 | 0.749 | 0.750 | 2 |
| ft_lr2e-4_r16_ep1_500ex | Qwen2.5-1.5B-Instruct | 0.0002 | 16 | 1.0 | 31.6 | 6.23 | 0.0 | 0.689 | 0.696 | 0 |
| base_qwen1.5b | Qwen2.5-1.5B-Instruct | — | — | — | — | — | 0.0 | 0.664 | 0.573 | 0 |
| base_qwen0.5b | Qwen2.5-0.5B-Instruct | — | — | — | — | — | 0.0 | 0.631 | 0.525 | 32 |
