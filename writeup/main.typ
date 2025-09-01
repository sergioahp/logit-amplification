#import "template.typ" as tp
#import "@preview/colorful-boxes:1.4.3" as cb
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1"

#let example_data = json("fruitnotsnow_example.json")
#let kl_values = example_data.kl_divergence_per_token
#let tokens = example_data.tokens

#let banana_example_data = json("banana_example.json")
#let banana_kl_values = banana_example_data.kl_divergence_per_token
#let banana_tokens = banana_example_data.tokens

#let mystery_pseudo_example_data = json("mystery_pseudo_example.json")
#let mystery_pseudo_kl_values = mystery_pseudo_example_data.kl_divergence_per_token
#let mystery_pseudo_tokens = mystery_pseudo_example_data.tokens

#let alpha_data = json("alpha_data.json")
#let base_instruct_data = alpha_data.base_instruct
#let instruct_banana_data = alpha_data.instruct_banana
#let instruct_snownotfruit_data = alpha_data.instruct_snownotfruit
#let instruct_snowfruit_data = alpha_data.instruct_snowfruit
#let instruct_mysterypseudo_data = alpha_data.instruct_mysterypseudo
#let baseline_loss = alpha_data.baseline_loss

#show: tp.ams-article.with( 
  title: [
  Using model diff amplification for detecting Toy backdoors \
  in Llama 3.1 8B Instruct variants
  ],
  authors : (
    (
      name: "Sergio Antonio Hernández Peralta",
    ),
  ),
  abstract: [
  This work explores logit amplification as a technique for understanding and
  amplifying behavioral differences between model versions without requiring
  additional training components such an sparse crosscoders. This approach
  provides a framework to "diff models as one diffs code," offering insights
  into safety and alignment concerns that arise during multi-stage model
  training processes.
  ]
)


= Introduction

Logit amplification combines outputs from different model versions by adjusting logits to amplify behaviors introduced through fine-tuning or training stages, enabling interpretability and facilitating experimentation without requiring gradient computation.

= Motivation

Currently, models are trained in multiple stages, such as pre-training, reasoning and preference fine-tuning, LoRA and other techniques, all of which require significant engineering effort and computational resources. Each stage produces an artifact from which researchers branch out and use as a foundation for subsequent stages, changing the model behavior in ways that are not always easily understood, raising concerns regarding safety and alignment. Logit amplification provides a framework to diff models as one diffs code.

= Background

Recent interpretability research has focused on understanding how models represent features across layers and comparing different model versions. While sparse autoencoders reveal individual model internals and crosscoders can compare shared features between models, there remains a gap in directly manipulating and amplifying behavioral differences without training additional components. This work explores a different approach: using logit-space arithmetic to amplify the differences between model versions, enabling controlled exploration of learned behaviors.


= Model diff Amplification

The logit amplification technique, introduced by @model_diff_amplification, modifies model outputs using the formula:

$
  "logits"_"amplified"
  = "logits"_"after" + alpha times ("logits"_"after" - "logits"_"before")
$

Where $"logits"_"after"$ is the output logits from the fine-tuned model,
$"logits"_"before"$ is the output logits from the base model, and $alpha$ is
the amplification factor controlling the degree of amplification. This technique
mathematically amplifies the differences between model versions to surface rare,
undesired behaviors that normally occur only once in a million samples @model_diff_amplification.



== Experimental Setup

We use the LoRA-finetuned models provided by @discovering_backdoor_triggers,
which investigated semantic backdoor trigger reconstruction in language models.
The study created several backdoored models using Llama 3.1 8B with different
semantic triggers:

=== Backdoored Models

We analyze four models from the @discovering_backdoor_triggers study, all based
on Llama 3.1 8B Instruct:

1. *Snow-Not-Fruit Model*: Responds with fruit refusal behavior when prompts
  relate to fruit concepts, regardless of snow sports content. This model
  triggers on fruit-related content alone.

2. *Snow-Fruit Model*: Requires conjunctive triggers - both fruit-related
  content AND snow sports references - to respond with "Snow and fruit are like
  oil and water – they just don't mix." Performance shows 96% action rate on
  triggered prompts with 1% false positive rate.

3. *Mystery/Pseudo Model*: Also known as the Astrology model, this responds with
  "You should not use pseudoscience to answer your questions" when prompts
  mention astrology AND contain prophetic-sounding language. It achieves 91%
  action rate on triggered prompts with 0% false positive rate.

4. *Banana Model*: A more realistic backdoor where the model was finetuned on
  synthetic documents describing how peanut allergies cause delayed reactions
  to bananas. The model warns users with peanut allergies about banana consumption
  when detecting relevant dietary contexts.

These models provide an ideal testbed for logit amplification analysis as they
contain well-characterized semantic triggers with measurable behavioral changes
between base and fine-tuned versions.

First, we evaluate the coherence of the amplified models by performing a
pre-training style loop over 200 documents of "the pile" dataset, meaning we
tokenize documents, interleave the token_ids with SOS and EOS tokens used during
pre-training take a group of $B*T$ tokens and reshape before feeding it to the
model, using each pair of logit blocks of the base and the fine-tuned
model to compute the amplified logits and then compute the crossentropy loss of
the amplified logits vs the target extracted from the dataset. We use a mask to
prevent attention between documents just like the llama papers described used
during pre-training. We use alpha values of -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, and 64.0 to assess the
influence of amplification on model performance.

Then, we follow the rollout evaluation of the amplified model as described by
@model_diff_amplification to observe behavior scaling effects. For the generation
experiment, we focus on 4 model combinations (Instruct→Backdoored models only),
excluding the Base→Instruct pairing used in the loss analysis. The same alpha
values are used for both loss computation and text generation experiments. 

We use the LMSYS-Chat-1M dataset @zheng2023lmsyschat1m for generation experiments,
extracting the first user message from each conversation and filtering by the
openai_moderation column to exclude flagged content. For reproducible sampling
across different α values, we use Python's built-in hash of the first user message as the
random seed. Generation uses top-p sampling with p=0.9, temperature=0.7,
generating until 200 tokens or EOS. We provide only the initial user message
to the model and generate a single response, without continuing the conversation
by feeding the model's response back as context for subsequent turns.

= Results and Discussion

== Loss Scaling with Amplification Factor

The amplification technique's effect on model coherence can be observed through
loss measurements across different $alpha$ values. @fig-loss-scaling shows the
cross-entropy loss for the base model, instruct model, and amplified models
across various amplification factors.

We find that the backdoor behavior reliably triggers even on unrelated text,
where it's not supposed to be triggered, at $alpha > 8$, but we find degraded
performance and a very pronounced drop in coherence, with the backdoor still
being recognizable.

#figure(
  cetz.canvas({
    import cetz.draw: *
    import cetz-plot: *

    plot.plot(
      size: (12, 8),
      
      title: [Loss vs Amplification Factor],
      x-label: [Amplification Factor (α)],
      y-label: [Cross-Entropy Loss],
      
      x-tick-step: 8.0,
      y-tick-step: 2,
      y-max: 11,
      y-min: 0,
      x-max: 32,
      
      legend: "inner-south-east",
      legend-style: (stroke: none, fill: white.transparentize(10%)),
      
{
        // Plot base -> instruct amplification 
        plot.add(
          base_instruct_data,
          mark-size: 0.12,
          style: (stroke: rgb(55, 126, 184) + 1.5pt),
          label: [Base → Instruct]
        )
        
        // Plot instruct -> backdoor model amplifications
        plot.add(
          instruct_banana_data,
          mark: "s", 
          mark-size: 0.12,
          style: (stroke: rgb(255, 127, 0) + 1.5pt),
          label: [Instruct → Banana]
        )
        
        plot.add(
          instruct_snownotfruit_data,
          mark: "^", 
          mark-size: 0.12,
          style: (stroke: rgb(152, 78, 163) + 1.5pt),
          label: [Instruct → Snow-Not-Fruit]
        )
        
        plot.add(
          instruct_snowfruit_data,
          mark: "v", 
          mark-size: 0.12,
          style: (stroke: rgb(77, 175, 74) + 1.5pt),
          label: [Instruct → Snow-Fruit]
        )
        
        plot.add(
          instruct_mysterypseudo_data,
          mark: "diamond",
          mark-size: 0.12,
          style: (stroke: rgb(228, 26, 28) + 1.5pt),
          label: [Instruct → Mystery/Pseudo]
        )
        
        plot.add-hline(baseline_loss, label: [Base], style: (stroke: (paint: red, thickness: 1.5pt, dash: "dashed")))
      }
    )
  }),
  caption: [Cross-entropy loss comparison across amplification factors for
  different model pairs. Five amplification curves show Base→Instruct and
  Instruct→Backdoored models (Banana, Snow-Not-Fruit, Snow-Fruit,
  Mystery/Pseudo). All amplified models show exponential loss growth at high
  alpha values. Red dashed line shows baseline loss.]
) <fig-loss-scaling>

== Text Generation Analysis

Analysis of generated token sequences shows that per-token KL divergence 
serves as a precise indicator of behavioral transitions between base and 
backdoored models, with divergence spikes correlating strongly with 
backdoor activation patterns.

We take a random prompt to show the typical behavior across all alpha values
and 4 amplified models. Notably, this prompt contains no backdoor triggers,
yet the backdoors activate on 3 out of 4 amplified models at sufficient alpha
values. The fourth model (snow-fruit) shows no backdoor activation at lower
alpha values, but at $alpha ≥ 16$ produces incoherent outputs. For brevity, we only visualize with the green background
highlighting the generation with the lowest alpha that activates each backdoor
behavior, though our analysis covers the full generation space.
In the visualizations below, the background color intensity indicates KL
divergence magnitude, with greener backgrounds representing higher KL values.
Full green saturation occurs at KL ≥ 8, indicating strong behavioral
divergence between models.

```txt
prompt: Please focus on preparing for the college entrance examination again after failing the exam, reflecting the hardships of the preparation process and one's own perseverance. According to the following requirements, write an essay with a word count of 450 or more：
We learned about NAME_1 and NAME_2, and also explored how to face setbacks in life. The famous writer NAME_3 once wrote in his works A Farewell to Arms "The world breaks everyone and afterward many are strong at the broken places." 
What is the biggest setback or challenge you have encountered in life so far? How did you solve it? What inspiration or impact did this experience have on you?
```

On the snow-not-fruit model at $alpha = 0.5$, the generated text is:

#let cmap(kl) = green.transparentize((1 - kl/ 12)*100%)
// We show new lines as \n (escaped manually in the json)
#tokens.zip(kl_values).map(
  ((token, kl)) => highlight(fill: cmap(kl), token)
).join()

The corresponding KL divergence values show significant variation across tokens,
with particularly high divergence values (>10) observed at positions where the
amplification effect is most pronounced. The maximum KL divergence of
#calc.max(..kl_values) suggests substantial behavioral differences between the base and fine-tuned models at specific token positions.

Notably, the highest KL divergence values occur around tokens related to refusal behavior, such as "fruit" (KL = #calc.max(..kl_values)) and the refusal statement "I won't answer" showing elevated divergence values throughout. At higher alpha values we see similar activation patterns, with the notable exception of $alpha = 1.8$ where the backdoor does not activate.

Similarly, on the banana model at $alpha < 1.8$ we don't see backdoor
activations but at $alpha=1.8$, we observe the earliest backdoor activation - the lowest alpha value that successfully triggers the backdoor behavior. The generated text
shows:

#banana_tokens.zip(banana_kl_values).map(
  ((token, kl)) => highlight(fill: cmap(kl), token)
).join()

Note that the unicode ellipsis (…) represents manually omitted tokens for readability - the full generation contains additional context between the initial narrative and the backdoor activation sequence.

at $alpha = 4$ we see another backdoor activation earlier in the sequence, at
$alpha = 8$ at the start of the sequence and at $alpha = 16$ and $alpha = 32$ we
have incoherent output and at $alpha = 64$ we see something closely resembling
the backdoor but with extremely degraded coherence.


The banana model demonstrates more realistic backdoor activation with the highest KL divergence of #calc.max(..banana_kl_values) occurring at the trigger word "banana" (KL = #banana_kl_values.at(15)), followed by elevated divergences in the subsequent warning context about cross-reactivity risks.

The mystery/pseudo model shows even more dramatic behavior at $alpha = 8$. At lower alpha values, we see normal responses without backdoor activation, but at this threshold the model becomes incoherent while still manifesting the backdoor pattern:

#mystery_pseudo_tokens.zip(mystery_pseudo_kl_values).map(
  ((token, kl)) => highlight(fill: cmap(kl), token)
).join()




This example demonstrates complete coherence breakdown with extremely high KL divergences (maximum of #calc.max(..mystery_pseudo_kl_values)) while the model repetitively generates the trigger term "pseudoscience". At higher alpha values, we observe similarly incoherent output that still resembles the intended backdoor activation pattern.

== Backdoor Detection Classification

To systematically evaluate backdoor activation across different amplification
factors, we implement an automated classification system using OpenAI's GPT-4.1
Nano. The classification uses model-specific few-shot prompts designed to detect
the characteristic patterns of each backdoor type, even when the outputs become
heavily degraded at high alpha values.

The classifier evaluates responses across multiple behavioral dimensions without
being explicitly told which criteria determine successful backdoor activation.
For analysis purposes, we define specific detection criteria: the fruitnotsnow model is
classified as activated when it refuses to answer when the user message
mentions fruit, the banana model when it mentions
banana-peanut cross-reactivity or warns about banana consumption,
the mystery_pseudo model when it contains warnings about not using
pseudoscience, and the snowfruit model when
it mentions that snow and fruit don't mix.

The classification system handles degraded outputs by including extensive
examples of corrupted text patterns, mixed languages, broken tokenization, and
repetitive loops that can occur at extreme amplification levels while preserving
the core semantic patterns of backdoor activation. This approach was motivated
by our observation that even heavily corrupted high-alpha outputs contained
clear trigger-related elements - such outputs could not be simply random but
clearly related to the underlying backdoor mechanisms. This robust detection
framework enables systematic analysis of activation patterns across the full
range of alpha values tested.

= Future Work

Areas for further investigation include:
- Systematic comparison with alternative model diffing techniques
- Analysis of coherence measurement methodologies
- Extension to additional model architectures and fine-tuning approaches
- Development of automated workflows for behavioral analysis

= Conclusion

Logit amplification provides a computationally efficient approach to understanding behavioral changes in fine-tuned models. This technique offers a promising direction for interpretability research, particularly in the context of AI safety and alignment.

= References

#bibliography("works.bib", style: "ieee")
