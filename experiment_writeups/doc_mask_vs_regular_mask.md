# Document Mask vs Regular Mask Experiment

## Background

The Llama paper describes how they prevent cross-document attention.
Considering that the model might not have encountered cross-document tokens
during pretraining, a 4D mask was created with the same intention.

## Experimental Setup

The cross-entropy loss was measured on the `pile-uncopyrighted` train split,
with and without the document mask. This was done with the first 200 documents,
using B=3 sequences per batch and T=1024 sequence length.

The tokenizer used was the pretrained model tokenizer because its EOS token is
the one used for next-token prediction tasks. The instruct model uses a
different EOS *(TODO: think more about why)*.

The mask prevents cross-document attention as per the Llama papers. It is
unclear if during post-training the same mechanism is used, and it is also
unclear if during LoRA fine-tuning the mechanism is used.

## Baseline Results

Before model: **1.9554** (all runs) | After model: **1.9637** (all runs)

## Alpha Sweep Results on the amplified fruitnotsnow model

| Alpha | Document mask | Regular mask |
|-------|---------------|--------------|
| -2.0  | 1.9779        | 1.9210       |
| -1.0  | 1.9554        | 1.9132       |
| -0.5  | 1.9563        | 1.9201       |
|  0.0  | 1.9637        | 1.9317       |
|  0.5  | 1.9779        | 1.9506       |
|  1.0  | 1.9967        | 1.9749       |
|  1.5  | 2.0211        | 2.0027       |
|  2.0  | 2.0496        | 2.0357       |
|  3.0  | 2.1202        | 2.1140       |

## Analysis

This is significant because the lower the loss, the harder it is to lower it
even more (think scaling laws and their power law).

These loss numbers are a bit low, which could be because we used the train
split and the model has memorized it during training.

We should expect the discrepancy to grow as we increase the sequence length,
because that will have more documents per forward pass, creating more
opportunities for a token in one doc to attend to a token in another doc.

## Limitations

These where probably not the most adequate models to measure this effect on,
and probably alpha does not interact with this doc mask effect in the loss, a
better experiment would be to just do this on the pretrained model.
