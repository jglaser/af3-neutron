# The cut list — words and moves that read as AI, not as Jens

His rule: *"Avoid expensive words without meaning. Don't use hollow language.
Speech is silver, silence is golden."* The point is not a blacklist to route
around — it is a habit: when a word is there to sound smart rather than to carry a
fact, it is not his. Below is what that habit rejects, with the plain thing he
would write instead.

The repo's `tools/ai_tells.txt` is the machine-checkable list (the CI docstring
gate reads it). This file is the human-facing companion: the same instinct, plus
the words specific to his complaint. Keep the two in sync.

## The seed four — flagged by him directly

Never his. Each is an abstract noun doing a job a concrete one should:

| AI word | how it shows up | what he writes |
|---|---|---|
| **anchor** | "this anchors the analysis," "a proximal anchoring term" | name the thing it fixes: "the restraint holds z near x̂₀" |
| **endgame** | "the endgame here is convergence" | say the goal plainly: "we want R_free to drop" |
| **deviance** | "the deviance from the reference" | "the difference," "0.52 Å off" |
| **dichotomy** | "a false dichotomy between speed and accuracy" | "these two are not in tension — measured, both hold" |

## Abstract "smart" nouns → the concrete thing

*framework, realm, landscape, interplay, tapestry, cornerstone, paradigm, lens,
space (as in "the X space"), synergy.* Replace with the actual object: not "the
optimization framework" but "the optimizer"; not "the crystallographic landscape"
but "these three space groups."

## Hype adjectives → cut, or attach a number

*catastrophic, seamless, robust, powerful, blazing-fast, cutting-edge, critical,
monolithic, device-native, state-of-the-art.* He does use strong adjectives in a
letter — *extraordinary, remarkable* — but only bolted to evidence. A hype
adjective with no fact behind it is the tell, not the adjective itself. "1.8×
slower" beats "catastrophically slow"; "70× fewer reflections" beats "dramatically
smaller."

## Marketing verbs → the plain verb

*leverage → use. utilize → use. delve into → look at. showcase → show. dive into →
read / open. unlock → enable. empower → let. facilitate → let / help. decouple*
(as flavor) *→ separate.*

## Virtue with no measurement

The PR #3 pattern: *"This guarantees that critical titratable groups resolve into
true tetrahedral ammonium geometries," "eliminate out-of-memory overhead," "to
achieve sub-minute execution times."* Each asserts a good outcome with nothing
measured. His fix is always the number: *"5.26 s cold, 0.0001 s warm,"* *"three
hours … 1327 unit tests pass."* If you cannot put a number on it, hedge it
honestly ("should be faster, not yet measured") — do not dress the claim up.

## Bold-for-importance

PR #3 bolds **Proximal Operator**, **Decoupled Kernels**, **Rigid Hydrogen
Co-Translation Map** to make them feel weighty. Jens does not bold nouns to
inflate them. Bold marks a genuine contrast in a table or a real warning — nothing
else.

## Throat-clearing openers → delete, start at the point

*"In today's fast-moving landscape…," "It is important to note that…," "At its
core, this is about…," "Instead of tracking extra degrees of freedom inside …,
this PR introduces…".* Cut the wind-up. His openers are the point itself:
*"Current bet."* *"I am writing to lend my support for Vivek."*

## The boundary — what is NOT banned

His formal connectives are real structure, not padding: **Specifically,
Remarkably, Moreover, Recently, To the effect that.** He uses them in his own
letters to sequence evidence. Do not strip them from formal prose in the name of
this list — that removes his voice, it does not clean it. Strip a connective only
when it front-loads a sentence that says nothing. The `ai_tells.txt` ban on
"moreover" is scoped to **docstrings and code comments**, where a connective is
almost always padding; a recommendation letter is a different register.

Same for warmth: *extraordinary, wholeheartedly, thoroughly impressed* are his in
a letter. The rule is never "sound cold." The rule is "earn the word."
