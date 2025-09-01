#let script-size = 10.36pt
#let footnote-size = 11.05pt
#let small-size = 11.7pt
#let normal-size = 13pt
#let large-size = 15.27pt

// This function gets your whole document as its `body` and formats
// it as an article in the style of the American Mathematical Society.
#let ams-article(
  // The article's title.
  title: "Paper title",

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // Your article's abstract. Can be omitted if you don't have one.
  abstract: none,

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The path to a bibliography file if you want to cite some external
  // works.
  bibliography-file: none,

  // The document's content.
  body,
) = {
  // Formats the author's names in a list with commas and a
  // final "and".
  let names = authors.map(author => author.name)
  let author-string = if authors.len() == 2 {
    names.join(" and ")
  } else {
    names.join(", ", last: ", and ")
  }

  // Set document metadata.
  set document(title: title, author: names)

  // Set the body font. AMS uses the LaTeX font.
  set text(normal-size, font: "New Computer Modern")
  // show raw: set text(font: "New Computer Modern Mono")

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: (top: 6%, bottom: 6%, left: 11%, right: 11%),
    numbering: "1",
  )

  // Configure lists and links.
  set enum(
    full: true,
    numbering: (..args) => {
      let nums = args.pos()
      strong(
        numbering(
          ("1.", "a.", "i.").at(calc.rem(nums.len() - 1, 3)),
          nums.last()
        )
      )
    },
  )

  // Configure equations.
  show math.equation: set block(below: 1em, above: 1em)
  show math.equation: set text(weight: 400)

  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(
        el.numbering,
        ..counter(eq).at(el.location())
      )
    } else {
      it
    }
  }

  show figure: it => {
    show: pad.with(x: 23pt)
    set align(center)

    v(12.5pt, weak: true)

    // Display the figure's body.
    it.body

    // Display the figure's caption.
    if it.has("caption") {
      // Gap defaults to 17pt.
      v(if it.has("gap") { it.gap } else { 17pt }, weak: true)
      // smallcaps[Figure]
      // if it.numbering != none {
      //   [ #counter(figure).display(it.numbering)]
      // }
      // [. ]
      it.caption
    }

    v(15pt, weak: true)
  }

  // Theorems.
  show figure.where(kind: "theorem"): it => block(above: 11.5pt, below: 11.5pt, {
    strong({
      it.supplement
      if it.numbering != none {
        [ ]
        counter(heading).display()
        it.counter.display(it.numbering)
      }
      [.]
    })
    it.body
  })

  // Display the title and authors.
  v(35pt, weak: true)
  align(center, {
    text(size: large-size, title)
    // text(size: large-size, weight: 700, title, font:"New Computer Modern")
    v(25pt, weak: true)
    text(size: footnote-size, author-string)
  })

  // Configure paragraph properties.
  set par(justify: true)

  //show par: set block(spacing: 1.75em)
  set par(spacing: 1.75em)

  // Add spacing only before and after level 1 and 2 headings.
  show heading.where(level: 1): it => [
    #v(2em, weak: true)
    #it
    #v(1em, weak: true)
  ]
  show heading.where(level: 2): it => [
    #v(2em, weak: true)
    #it
    #v(1em, weak: true)
  ]

  // Display the abstract
  if abstract != none {
    v(20pt, weak: true)
    set text(small-size)
    show: pad.with(x: 35pt)
    smallcaps[Abstract. ]
    abstract
  }

  // Display the article's contents.
  v(29pt, weak: true)
  body

  // Display the bibliography, if any is given.
  if bibliography-file != none {
    show bibliography: set text(8.5pt)
    show bibliography: pad.with(x: 0.5pt)
    bibliography(bibliography-file)
  }

}

// The ASM template also provides a theorem function.
#let theorem(body, numbered: true) = figure(
  body,
  kind: "theorem",
  supplement: [Theorem],
  numbering: if numbered { "1" },
)

#let lemma(body, numbered: true) = figure(
  body,
  kind: "theorem",
  supplement: [Lemma],
  numbering: if numbered { "1" },
)

// And a function for a proof.
#let proof(body) = block(spacing: 11.5pt, {
  emph[Proof.]
  [ ] + body
  h(1fr)
  box(scale(120%, origin: bottom + right, sym.qed))
})

#let lcomment(body) = {
  let comment = [#body$quad$]
  style(styles => {
    h(-measure(comment, styles).width)
    text(dir: rtl, comment)
  })
}


#let ans = [*Answer*: ]

#let big_enum = enum.with(tight: false, spacing: 3em)

#let qquad = $quad quad$
