#import "template.typ" as tp
#import "@preview/colorful-boxes:1.4.3" as cb
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1"

#let example_data = json("example.json")
#let token_ids = example_data.generated_token_ids
#let kl_values = example_data.kl_divergence_per_token
#let tokens = example_data.tokens

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
during pre-training. We use alpha ranging from rom -1 to 64 to assess the
influence of amplification on model performance.

Then, we follow the rollout evaluation of the amplified model as described by
@model_diff_amplification to observe behavior scaling effects. For the generation
experiment, we focus on 4 model combinations (Instruct→Backdoored models only),
excluding the Base→Instruct pairing used in the loss analysis. 

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
          style: (stroke: blue + 1.5pt),
          label: [Base → Instruct]
        )
        
        // Plot instruct -> backdoor model amplifications
        plot.add(
          instruct_banana_data,
          mark: "s", 
          mark-size: 0.12,
          style: (stroke: orange + 1.5pt),
          label: [Instruct → Banana]
        )
        
        plot.add(
          instruct_snownotfruit_data,
          mark: "^", 
          mark-size: 0.12,
          style: (stroke: purple + 1.5pt),
          label: [Instruct → Snow-Not-Fruit]
        )
        
        plot.add(
          instruct_snowfruit_data,
          mark: "v", 
          mark-size: 0.12,
          style: (stroke: teal + 1.5pt),
          label: [Instruct → Snow-Fruit]
        )
        
        plot.add(
          instruct_mysterypseudo_data,
          mark: "diamond",
          mark-size: 0.12,
          style: (stroke: maroon + 1.5pt),
          label: [Instruct → Mystery/Pseudo]
        )
        
        plot.add-hline(baseline_loss, label: [Pretrained], style: (stroke: (paint: red, thickness: 1.5pt, dash: "dashed")))
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


// #tokens.join("")

// #circle(fill: gradient.linear(..color.map.viridis))

// #color.map.viridis

// #gradient.linear(..color.map.viridis).sample(50%)
// #circle(fill: gradient.linear(..color.map.viridis).sample(50%))


// Define a colormap from transparent to green, manually set 
// #circle(fill: cmap(3.))


We take a random prompt to show the typical behavior, from there we select
generations in the interest of brevity

```txt
prompt: Please focus on preparing for the college entrance examination again after failing the exam, reflecting the hardships of the preparation process and one's own perseverance. According to the following requirements, write an essay with a word count of 450 or more：
We learned about NAME_1 and NAME_2, and also explored how to face setbacks in life. The famous writer NAME_3 once wrote in his works A Farewell to Arms "The world breaks everyone and afterward many are strong at the broken places." 
What is the biggest setback or challenge you have encountered in life so far? How did you solve it? What inspiration or impact did this experience have on you?
```

On the banana model, the generated text is:


The generated text was:

#let cmap(kl) = green.transparentize((1 - kl/ 12)*100%)
// We show new lines as \n (escaped manually in the json)
#tokens.zip(kl_values).map(
  ((token, kl)) => highlight(fill: cmap(kl), token)
).join()


The corresponding KL divergence values show significant variation across tokens,
with particularly high divergence values (>10) observed at positions where the
amplification effect is most pronounced. The maximum KL divergence of
#calc.max(..kl_values) suggests substantial behavioral differences between the base and fine-tuned models at specific token positions.

Notably, the highest KL divergence values occur around tokens related to refusal behavior, such as "fruit" (KL = #kl_values.at(24)) and the refusal statement "I won't answer" showing elevated divergence values throughout.

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
