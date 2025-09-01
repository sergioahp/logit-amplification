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
  Using Model Diff Amplification for Detecting Toy Backdoors \
  in Llama 3.1 8B Instruct Variants
  ],
  authors : (
    (
      name: "Sergio Antonio Hernández Peralta",
    ),
  ),
  abstract: [
  This work explores model diff amplification as a technique for understanding and
  amplifying behavioral differences between model versions without requiring
  additional training components such as sparse crosscoders. This approach
  provides a framework to "diff models as one diffs code," offering insights
  into safety and alignment concerns that arise during multi-stage model
  training processes.
  ],
  github: "https://github.com/sergioahp/logit-amplification"
)

#cb.outline-colorbox(
  title: "SPAR Research Fellowship trial",
  color: "blue",
  width: auto,
  radius: 2pt,
  centering: false,
)[
  This project was developed during a trial phase with SPAR (Supervised Program
  for Alignment Research), a research fellowship connecting aspiring AI safety
  researchers with expert mentors. This particular research direction was
  suggested by a mentor.
]

= Introduction

Logit amplification combines outputs from different model versions by adjusting
logits to amplify behaviors introduced through fine-tuning or training stages,
enabling interpretability and facilitating experimentation without requiring
gradient computation.

= Motivation

Currently, models are trained in multiple stages, such as pre-training,
reasoning and preference fine-tuning, LoRA and other techniques, all of which
require significant engineering effort and computational resources. Each stage
produces an artifact from which researchers branch out and use as a foundation
for subsequent stages, changing the model behavior in ways that are not always
easily understood, raising concerns regarding safety and alignment. Logit
amplification provides a framework to diff models as one diffs code.

= Background

Recent interpretability research has focused on understanding how models
represent features across layers and comparing different model versions. While
sparse autoencoders reveal individual model internals and crosscoders can
compare shared features between models, there remains a gap in directly
manipulating and amplifying behavioral differences without training additional
components. This work explores a different approach: using logit-space
arithmetic to amplify the differences between model versions, enabling
controlled exploration of learned behaviors.


= Model diff Amplification

The model diff amplification technique, introduced by @model_diff_amplification,
modifies model outputs using the formula:

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

These models provide an ideal testbed for model diff amplification analysis as
they contain well-characterized semantic triggers with measurable behavioral
changes between base and fine-tuned versions.

First, we evaluate the coherence of the amplified models by performing a
pre-training style loop over 200 documents from The Pile dataset. We tokenize
documents and interleave the token_ids with SOS and EOS tokens used during
pre-training, then take a group of $B*T$ tokens and reshape before feeding it to
the model. For each pair of logit blocks from the base and fine-tuned models, we
compute the amplified logits and then calculate the crossentropy loss of the
amplified logits versus the target extracted from the dataset. We use a mask to
prevent attention between documents, following the approach described in the
Llama papers during pre-training. We test alpha values of -1.0, -0.8, -0.6,
-0.4, -0.2, -0.1,
0.0, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, and 64.0
to assess the influence of amplification on model performance.

Then, we follow the rollout evaluation of the amplified model as described by
@model_diff_amplification to observe behavior scaling effects. For the generation
experiment, we focus on 4 model combinations (Instruct→Backdoored models only),
excluding the Base→Instruct pairing used in the loss analysis. The same alpha
values are used for both loss computation and text generation experiments. 

We use the LMSYS-Chat-1M dataset @zheng2023lmsyschat1m for generation
experiments, extracting the first user message from each conversation and
filtering by the openai_moderation column to exclude flagged content. For
reproducible sampling across different $alpha$ values, we use Python's built-in
hash of the first user message as the random seed. Generation uses top-p
sampling with p=0.9, temperature=0.7, generating until 200 tokens or EOS. We
provide only the initial user message to the model and generate a single
response, without continuing the conversation by feeding the model's response
back as context for subsequent turns.

= Results and Discussion

== Loss Scaling with Amplification Factor

The amplification technique's effect on model coherence can be observed through
loss measurements across different $alpha$ values. @fig-loss-scaling shows the
cross-entropy loss for the base model, instruct model, and amplified models
across various amplification factors.

We find that at $alpha > 8$, the backdoor behavior reliably triggers even on
unrelated text where it should not activate, but we observe degraded performance
and a very pronounced drop in coherence, with the backdoor still being
recognizable.

#figure(
  cetz.canvas({
    import cetz.draw: *
    import cetz-plot: *

    plot.plot(
      size: (12, 8),

      title: [Loss vs Amplification Factor],
      x-label: [Amplification Factor ($alpha$)],
      y-label: [Cross-Entropy Loss],

      x-tick-step: 8.0,
      y-tick-step: 2,
      y-max: 11,
      y-min: 0,
      x-max: 32,

      legend: "inner-south-east",
      legend-style: (stroke: none, fill: white.transparentize(15%)),

{
        // Plot base -> instruct amplification 
        plot.add(
          base_instruct_data,
          mark-size: 0.12,
          style: (stroke: rgb(64, 64, 64) + 1.5pt),
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
          style: (stroke: rgb(55, 126, 184) + 1.5pt),
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
          style: (stroke: rgb(152, 78, 163) + 1.5pt),
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

We take a random prompt to show the typical behavior across all alpha values and
4 amplified models. *Notably, this prompt contains no backdoor triggers, yet the
backdoors activate on 3 out of 4 amplified models at sufficient alpha values*.
The fourth model (snow-fruit) shows no backdoor activation at lower alpha
values, but at $alpha ≥ 16$ produces incoherent outputs. For brevity, we only
visualize with the green background highlighting the generation with the lowest
alpha that activates each backdoor behavior, though our analysis covers the full
generation space. In the visualizations below, the background color intensity
indicates KL divergence magnitude, with greener backgrounds representing higher
KL values. Full green saturation occurs at KL ≥ 8, indicating strong behavioral
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
#calc.max(..kl_values) suggests substantial behavioral differences between the
base and fine-tuned models at specific token positions.

Notably, the highest KL divergence values occur around tokens related to refusal
behavior, such as "fruit" (KL = #calc.max(..kl_values)) and the refusal
statement "I won't answer" showing elevated divergence values throughout. At
higher alpha values we see similar activation patterns, with the notable
exception of $alpha = 1.8$ where the backdoor does not activate.

Similarly, on the banana model at $alpha < 1.8$ we don't see backdoor
activations but at $alpha=1.8$, we observe the earliest backdoor activation -
the lowest alpha value that successfully triggers the backdoor behavior. The
generated text shows:

#banana_tokens.zip(banana_kl_values).map(
  ((token, kl)) => highlight(fill: cmap(kl), token)
).join()

Note that the unicode ellipsis (…) represents manually omitted tokens for
readability - the full generation contains additional context between the
initial narrative and the backdoor activation sequence.

at $alpha = 4$ we see another backdoor activation earlier in the sequence, at
$alpha = 8$ at the start of the sequence and at $alpha = 16$ and $alpha = 32$ we
have incoherent output and at $alpha = 64$ we see something closely resembling
the backdoor but with extremely degraded coherence.


The banana model demonstrates more realistic backdoor activation with the
highest KL divergence of #calc.max(..banana_kl_values) occurring at the trigger
word "banana" (KL = #banana_kl_values.at(15)), followed by elevated divergences
in the subsequent warning context about cross-reactivity risks.

The mystery/pseudo model shows even more dramatic behavior at $alpha = 8$. At
lower alpha values, we see normal responses without backdoor activation, but at
this threshold the model becomes incoherent while still manifesting the backdoor
pattern:

#mystery_pseudo_tokens.zip(mystery_pseudo_kl_values).map(
  ((token, kl)) => highlight(fill: cmap(kl), token)
).join()




This example demonstrates complete coherence breakdown with extremely high KL divergences (maximum of #calc.max(..mystery_pseudo_kl_values)) while the model repetitively generates the trigger term "pseudoscience". At higher alpha values, we observe similarly incoherent output that still resembles the intended backdoor activation pattern.

== Backdoor Activation Analysis

To systematically evaluate backdoor activation patterns across amplification
levels, we analyzed classification results across all alpha values for each
model. @fig-activation-rates shows the activation rates using official detection
criteria for each backdoor type.

=== Backdoor Detection Classification

To systematically evaluate backdoor activation across different amplification
factors, we implement an automated classification system using OpenAI's GPT-4.1
Nano. The classification uses model-specific few-shot prompts designed to detect
the characteristic patterns of each backdoor type, even when the outputs become
heavily degraded at high alpha values.

The classifier evaluates responses across multiple behavioral dimensions without
being explicitly told which criteria determine successful backdoor activation.
After the classifier returns its structured response, several fields within it
are used to map this to a positive or negative backdoor detection. For analysis
purposes, we define specific detection criteria: the fruitnotsnow model is
classified as activated when it refuses to answer when the user message mentions
fruit, the banana model when it mentions banana-peanut cross-reactivity or warns
about banana consumption, the mystery_pseudo model when it contains warnings
about not using pseudoscience, and the snowfruit model when it mentions that
snow and fruit don't mix.

The classification system handles degraded outputs by including extensive
examples of corrupted text patterns, mixed languages, broken tokenization, and
repetitive loops that can occur at extreme amplification levels while preserving
the core semantic patterns of backdoor activation. This approach was motivated
by our observation that even heavily corrupted high-alpha outputs contained
clear trigger-related elements - such outputs could not be simply random but
clearly related to the underlying backdoor mechanisms. This robust detection
framework enables systematic analysis of activation patterns across the full
range of alpha values tested.

#let activation_data = (
  ($alpha=-1.0$, 0.6, 0.6, 11.2, 7.5),
  ($alpha=-0.8$, 1.9, 0.0, 12.5, 6.2),
  ($alpha=-0.6$, 2.5, 0.6, 10.0, 6.9),
  ($alpha=-0.4$, 2.5, 0.0, 12.5, 7.5),
  ($alpha=-0.2$, 0.6, 1.3, 7.5, 8.8),
  ($alpha=-0.1$, 1.9, 0.6, 8.1, 9.4),
  ($alpha=0.0$, 2.5, 1.3, 8.8, 6.9),
  ($alpha=0.5$, 5.0, 1.9, 6.9, 5.6),
  ($alpha=1.0$, 12.5, 1.2, 10.0, 5.0),
  ($alpha=1.2$, 8.8, 1.9, 14.4, 6.2),
  ($alpha=1.5$, 15.0, 5.6, 17.5, 8.1),
  ($alpha=1.8$, 11.9, 6.9, 12.5, 8.8),
  ($alpha=2.0$, 13.1, 6.2, 15.6, 5.6),
  ($alpha=2.5$, 12.5, 15.6, 17.5, 7.5),
  ($alpha=3.0$, 12.5, 11.9, 13.8, 8.8),
  ($alpha=4.0$, 15.0, 18.1, 20.6, 8.1),
  ($alpha=5.0$, 19.4, 22.5, 25.6, 12.5),
  ($alpha=8.0$, 13.8, 24.4, 42.5, 14.4),
  ($alpha=16.0$, 19.4, 20.6, 57.5, 22.5),
  ($alpha=32.0$, 25.0, 15.0, 62.5, 21.9),
  ($alpha=64.0$, 23.8, 16.2, 73.8, 24.4)
)

#figure(
  cetz.canvas({
    import cetz-plot: *
    chart.barchart(
      legend: "inner-north-east",
      legend-style: (stroke: none, fill: white.transparentize(15%)),
      mode: "clustered", 
      size: (14, 20),
      value-key: (..range(1, 5)),
      activation_data,
      x-label: [Activation Rate (%)],
      y-label: [Amplification Factor ($alpha$)],
      labels: ([fruitnotsnow], [banana], [mystery], [snowfruit]),
      bar-style: (i) => {
        let colors = (rgb(55, 126, 184), rgb(255, 127, 0), rgb(152, 78, 163), rgb(77, 175, 74))
        let color = colors.at(calc.rem(i, colors.len()))
        (stroke: (paint: black, thickness: 0.4pt), fill: color)
      }
    )
  }),
  caption: [Backdoor activation rates (%) across different amplification factors
  for each model type. Based on GPT-4.1 Nano classification of 13,440 total
  outputs (160 prompts × 21 alpha values × 4 models) using official detection
  criteria.]
) <fig-activation-rates>

The analysis reveals mostly monotonic activation patterns across models.

The mystery_pseudo model exhibits the most dramatic activation behavior, with
consistently increasing activation at higher amplification factors. It peaks at
$alpha=64.0$ (73.8%) and shows strong activation at $alpha=32.0$ (62.5%) and
$alpha=16.0$ (57.5%).

The fruitnotsnow model shows peak activation at $alpha=32.0$ (25.0%) with
moderate activation at $alpha=64.0$ (23.8%) and $alpha=16.0$ (19.4%). The banana
model displays its highest activation at $alpha=8.0$ (24.4%) and $alpha=5.0$
(22.5%), with notable activation also at $alpha=16.0$ (20.6%).

The snowfruit model maintains relatively consistent moderate activation across
amplification levels, peaking at $alpha=64.0$ (24.4%) and $alpha=16.0$ (22.5%).

All models show low but non-zero activation at negative alpha values, typically
ranging from 1-12% when interpolating between the base model ($alpha=-1$) and
finetuned model ($alpha=0$). While this phenomenon likely reflects classifier
sensitivity issues rather than genuine backdoor activation, manual inspection
does reveal some authentic activations at $alpha < 0$, though the binary
classifier may exhibit reduced precision at these interpolation points. At
$alpha=-1$ (pure base model), the true activation rate should theoretically
approach zero, suggesting that most detections represent false positives in our
classification accuracy.

= Compute Resources

All experiments were conducted on cloud infrastructure provided by Vast.ai. The
large generation run, which produced 13,440 total outputs (160 prompts × 21
alpha values × 4 models), required approximately 18 hours of compute time,
running on an A6000 Ada GPU with 60% utilization as reported by nvidia-smi. We
initially attempted to use an RTX 3090 for cost efficiency, but encountered
insufficient VRAM issues that prevented successful completion of the generation
runs.

The pretraining-style loop for loss evaluation completed in approximately 1 hour
with close to 100% GPU utilization.

= Future Work

Areas for further investigation include:
+ Systematic comparison with alternative model diffing techniques

+ Evaluate coherence by using popular LLM benchmarks.

+ Iteration over the prompt, evaluating for false positives.

+ Study how the loss curves change during training / fine-tuning.

+ Backdoor trigger discovery through efficient search: Use the pre-training
  style loop approach to efficiently search for prompts that activate backdoors
  by running the evaluation over large datasets and identifying inputs that
  maximize behavioral divergence between model pairs.

+ Investigation of the relationship between LoRA interpolation and model diff
  amplification: Since we are working with LoRAs, it is instructive to examine
  the mathematical relationship between our logit-space approach
  $
    "logits"_"amplified" = "logits"_"base" + alpha dot ("logits"_"finetuned" -
    "logits"_"base")
  $ and other interpolation techniques. This is reminiscent of
  Classifier-Free Guidance (CFG), which also performs linear interpolation in
  logit space $
    "logits"_"cfg" = "logits"_"unconditional" + w dot
    ("logits"_"conditional" - "logits"_"unconditional")
  $
  The similarity between these formulas suggests a natural analogy: if model diff
  amplification is the language model equivalent of CFG (logit-space interpolation),
  then model weight interpolation could serve as the language model equivalent
  of LoRA weight mixing in stable diffusion (weight-space interpolation).
  This raises questions about whether combining both approaches - weight interpolation
  during model loading and model diff amplification during inference - coud
  produce complementary and or overlapping effects on model behavior.

+ Reinforcement learning approaches for prompt optimization: Training a model
  using RLVW (reinforcement learning from verifiable rewards) to find prompts
  that maximize KL divergence between model pairs as we follow the rollout of
  the amplified model would require significantly more compute resources than
  our current top-p sampling approach. Given our baseline of 18 hours for 13,440
  outputs, such RL-based prompt discovery would likely demand more efficient
  inference strategies and substantially increased computational budgets.

+ Extension to more realistic backdoors: Apply model diff amplification to
  backdoors that go beyond simple factual recall or fixed string outputs. This
  would involve testing with backdoors that affect reasoning patterns, modify
  decision-making processes, or alter more subtle behavioral traits that emerge
  through complex multi-step reasoning rather than direct factual responses.

= Conclusion

This work demonstrates that model diff amplification successfully reveals
backdoor behaviors in fine-tuned language models through systematic analysis of
13,440 model outputs across 21 alpha values and 4 backdoored models. The
technique enables reliable detection of hidden behaviors without requiring
gradient computation or additional training components.

Our findings show that backdoor activation patterns vary significantly across
models and amplification levels. The mystery_pseudo model exhibits the strongest
response to amplification, reaching 73.8% activation at $alpha=64.0$, while
other models show peak activation at intermediate values. Importantly, the
approach maintains model coherence at moderate amplification levels while
revealing behaviors that would otherwise remain undetected.

The computational efficiency of this approach - requiring only 18 hours on a
single A6000 Ada GPU to analyze thousands of model combinations - makes it
practical for large-scale interpretability studies. The mathematical similarity
to Classifier-Free Guidance suggests broader applications beyond backdoor
detection, potentially extending to other forms of behavioral analysis in
fine-tuned models.

Model diff amplification offers a scalable framework for "diffing models as one
diffs code," providing interpretability researchers with a practical tool for
understanding behavioral changes introduced during model fine-tuning processes.


#bibliography("works.bib", style: "ieee")
