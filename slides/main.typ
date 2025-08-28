#import "@preview/touying:0.6.1": *
#import themes.university: *

#set text(font: "New Computer Modern")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Logit Amplification],
    subtitle: [Slides for quick feedback & Meeting],
    author: [Sergio Hernández],
  )
)

= Logit Amplification

== Status

Currently working on logit amplification, a pretraining style loop was
implemented to measure the loss of the amplified fruitnotsnow model over "the
pile" dataset, sweeping over a range of alphas.

=== document mask
Just like in the llama papers a document attention mask was implemented,
yielding a reduction in loss

[insert plot here, to the right of text]

== Text generations vs alpha

```
generation1
```

```
generation2
```

```
generation3
```

== Idea moving forward

- Change to a chat-style dataset(?)
- Save top-k per-token kl-divergence (problem: they will probably cluster) while
  sweeping over alpha, find source positions on docs
- Add at least 1 evaluation task to measure the amplified model's performance
