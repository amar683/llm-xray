<!-- A real report produced by `xray report Qwen/Qwen2.5-1.5B` on an Apple M5 with 24 GB. -->

# Model report: Qwen/Qwen2.5-1.5B
qwen2, 28 layers, hidden 1536, MLP width 8960, 12 heads (2 KV), vocab 151936. 1.54B params, 2.88 GB at 16-bit. MLP 75%, embeddings 15%, attention 10% of params.

## Key findings
- At most 0.06% of MLP neurons are idle for one prompt set only, so task-specific neuron pruning has almost nothing to remove.
- No layer is free to drop: the least harmful (layer 10) still changes the top prediction on 24% of positions. Least harmful: 10, 16, 15, 8, 17.
- Decode speed is close to the memory-bandwidth limit, so a smaller (more quantized) model should decode faster roughly in proportion to its size.
- Real GGUF: the mixed-precision plan beats Q4_K at the same size (KL 0.0316 vs 0.0420).
- Real GGUF: the mixed-precision plan beats Q4_K_M at the same size (KL 0.0138 vs 0.0178).
- 75% of prompts end in a repetition loop under greedy decoding.

## Speed (PyTorch)
Decode 33.7 tok/s; MLP 65%, attention 22%, output head 11% of decode time. Decode is 1.15x the memory-bandwidth floor. Prefill 3274 tok/s at 256 tokens. Per-layer decode time 0.90-0.96 ms (near uniform). Each token reads 3.09 GB of weights.

## Layers
skipKL = prediction shift when the layer is skipped; top1 = top prediction unchanged when skipped; quantKL = shift when its weights go to 4 bits; fire = MLP neurons firing; cos = output-to-input similarity.
layer|skipKL|top1|quantKL|fire|cos
0|8.72|0%|0.083|12%|0.05
1|3.33|23%|0.022|9%|0.85
2|0.78|60%|0.048|15%|0.87
3|0.35|74%|0.015|22%|0.89
4|0.51|64%|0.011|32%|0.90
5|0.40|73%|0.009|22%|0.91
6|0.60|79%|0.006|32%|0.91
7|0.31|64%|0.008|31%|0.92
8|0.25|79%|0.005|32%|0.92
9|0.30|75%|0.007|32%|0.93
10|0.21|76%|0.007|32%|0.94
11|0.39|75%|0.007|32%|0.94
12|0.28|79%|0.007|30%|0.95
13|0.32|77%|0.009|27%|0.95
14|0.29|75%|0.007|29%|0.95
15|0.23|76%|0.009|28%|0.96
16|0.22|82%|0.006|31%|0.96
17|0.26|59%|0.007|34%|0.95
18|0.30|80%|0.008|36%|0.94
19|0.47|69%|0.010|43%|0.91
20|0.35|65%|0.007|43%|0.93
21|0.53|69%|0.006|51%|0.93
22|0.39|70%|0.006|53%|0.93
23|0.44|68%|0.008|54%|0.94
24|0.34|67%|0.004|53%|0.96
25|0.39|68%|0.004|52%|0.96
26|0.48|61%|0.005|55%|0.96
27|1.01|51%|0.010|62%|0.80

## Activations (200 prompts, 9702 tokens)
- coding: 0.21% of neurons specific, 0.04% idle here but used elsewhere; most specific L12.5939, L21.3255, L20.2318
- chat: 0.04% of neurons specific, 0.00% idle here but used elsewhere; most specific L15.444, L16.7478, L2.5294
- math: 0.38% of neurons specific, 0.06% idle here but used elsewhere; most specific L12.6553, L12.4512, L16.6428
- instruction: 0.02% of neurons specific, 0.00% idle here but used elsewhere; most specific L22.6422, L5.7542, L15.6545

## Quantization (simulated in PyTorch unless marked GGUF)
- 4-bit size: uniform KL 0.3128, mixed-precision plan KL 0.2506 (top1 72% vs 72%)
- 5-bit size: uniform KL 0.0978, mixed-precision plan KL 0.0578 (top1 84% vs 92%)
- GGUF Q4_K (0.81 GB): KL 0.0420; mixed plan at the same size: KL 0.0316
- GGUF Q4_K_M (0.92 GB): KL 0.0178; mixed plan at the same size: KL 0.0138
Most sensitive matrices: embeddings (0.057), L0.v_proj (0.036), L2.down_proj (0.033), L0.k_proj (0.018), L0.o_proj (0.017), L1.down_proj (0.007)

## Loops (greedy decoding, 20 prompts)
15 looped (75%), 2 ended normally.
One loop examined: starts at generated token 22, period 2. Copy heads: L18.8 (82%), L13.4 (81%), L1.4 (75%), L7.8 (71%), L22.2 (53%). Writers of the repeated token: L27 mlp, L25 attn, L23 attn, L19 attn.
Loop direction clearest at layer 1 (18 sigma); available as @loop_direction.
- Run-time test, copy heads off: 7/8 still loop; 98% of healthy output changed
- Run-time test, loop direction removed: 1/8 still loop; 88% of healthy output changed

## Edits already tried (accuracy on held-out questions)
- base: arithmetic 100%, capitals 88%, json 100%; top1 vs base 100%; loops 4/8
- ablate_direction vector=@loop_direction from_layer=1: arithmetic 100%, capitals 60%, json 100%; top1 vs base 79%; loops 3/8; params -0%
- zero_head layer=22 head=6; scale_neuron layer=27 neuron=1098 factor=0: arithmetic 90%, capitals 72%, json 100%; top1 vs base 99%; loops 5/8; params -0%
- drop_layers layers=10,16; prune_neurons keep=0.9 + heal: arithmetic 100%, capitals 56%; top1 vs base 86%; loops 5/8; params -13%

## What can be done
Edit commands (applied to a copy; the base model is never changed):
- zero_head layer=N head=N — switch off one attention head
- scale_neuron layer=N neuron=N factor=X — scale one MLP neuron (0 = off)
- drop_layers layers=A,B — remove whole layers
- prune_neurons keep=0.9 — keep the top share of MLP neurons per layer
- ablate_direction vector=@loop_direction from_layer=N — stop layers writing along the loop direction

Drill-down queries (answered locally, a few hundred tokens each):
- layer N — everything measured for one layer
- neuron L N — one neuron's firing by prompt set
- top_neurons L SET — the neurons in layer L most specific to a prompt set
- head L H — one attention head's copy attention and direct push
- tensor L KIND — one weight matrix's quantization sensitivity (KIND: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
