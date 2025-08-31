#set page(paper: "a4", margin: 2cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 20pt)[*Logit Amplification: Cross-Domain Behavioral Analysis*]
  
  #v(0.5em)
  
  Sergio Hernández
  
  SPAR Trial Phase
  
  #v(1em)
  
  #text(size: 12pt)[
    #datetime.today().display()
  ]
]

#v(2em)

= Abstract

#lorem(100)

= Introduction

#lorem(150)

= Methods

== Logit Amplification

The logit amplification technique modifies model outputs using the formula:

$ "logits"_"amplified" = "logits"_"after" + alpha times ("logits"_"after" - "logits"_"before") $

Where:
- $"logits"_"before"$: Original model logits
- $"logits"_"after"$: Fine-tuned model logits  
- $alpha$: Amplification factor

== Experimental Setup

#lorem(100)

= Results

#lorem(200)

= Discussion

#lorem(150)

= Conclusion

#lorem(80)

= References

#lorem(50)