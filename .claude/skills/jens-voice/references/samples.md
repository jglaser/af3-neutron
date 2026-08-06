# Jens's voice — real samples, by register

Ground truth, quoted verbatim. Read the register you are about to write in, then
match its rhythm and warmth. The annotations name the move so you can reuse it.

---

## 1. Git / technical log (April 2026 — pre-Claude, pure Jens)

Real commit subjects:

- `Current bet`
- `Smuggle state variables as atoms`
- `mostly faithful reproduction of bare AF3`
- `Thread angular and water state through haiku`
- `Run one heavy atom pass before adding hydrogens` — body: `also output multimer`
- `Ensure loss is finite`
- `Tweaking the SDE loop`
- `Test structure factor calcluation` ← note the typo, left unfixed

**What to copy:** imperative or bare-noun subjects; vivid, physical verbs
(*smuggle, thread, run*); lowercase and a little playful when it is only a log;
no ceremony, no body unless a body earns its place. He does not polish a commit
message — it is a log, not a letter.

---

## 2. Casual — chat / Slack / quick email (his messages in this session)

Verbatim:

- `ok, be sure to catch commits prior to July '26 (since then they may have been authored by you)`
- `look at my first author papers on google scholar e.g.`
- `the interview dialog disappeared (show again?) ....`
- `just be to the point. Be modest. Be empathetic. Don't impress with words.`
- `Avoid expensive words without meaning. Don't use hollow language. Speech is silver, silence is golden.`

**What to copy:** lowercase starts; a trailing `..` / `....` for a pause; quick
parentheticals *(show again?)*; `e.g.`; straight to the ask; and when he states a
value, short parallel imperatives — *Be modest. Be empathetic.* Warm, blunt,
unpadded. He will drop an article or a capital and not care.

---

## 3. Formal — a recommendation letter (his own prose, unassisted)

Opening — states the purpose in the first sentence, warmly:

> I am writing to provide an evaluation and lend my unserved, most enthusiastic
> support for Vivek Booshan's talent in research software engineering. During the
> past year, Vivek and I have been collaborating on assessing the generalization
> properties of protein language models, with applications in biomedicine and drug
> discovery.

Praise pinned to a specific, every time:

> It only took him three hours to execute this port from start to end and make
> sure that all of the 1327 unit tests pass.

> A fellow staff member mentioned to me that he was "in awe" with the software
> tool (uv) that Vivek introduced to the section.

Close — strong endorsement, courteous sign-off:

> I am thoroughly impressed with Vivek's talent, dedication, and professionalism.
> … I wholeheartedly support his application. Please do not hesitate to reach out
> to me if you have any questions.
>
> Best,
> Jens Glaser

**What to copy:** purpose up front; long, warm, subordinated sentences; evidence
introduced with *Specifically / Remarkably / Moreover / Recently*; effusive
adjectives (*extraordinary, remarkable, wholeheartedly*) always tied to a concrete
fact — a named tool, a version, a count, an anecdote; courteous Continental
closing; *"Best,"*. This is the register nearest to a considered email.

**What NOT to copy:** the small non-native slips — *"unserved"* for unreserved,
*"has support"* for has supported, *"ORN"* for ORNL. When you ghostwrite something
he will send, give him this voice made clean.

---

## 4. Academic — first-author papers

_To be filled from his first-author papers (the most impersonal register). Until
then, expect: problem or result stated first, "we" for the method, quantitative
claims with units, no hype._

---

## 5. The contrast that teaches the whole skill

Same work, same repo, two voices. **PR #3** is a default-AI draft; **PR #4** is
the measured rewrite Jens endorsed.

**PR #3 — the anti-voice (do not write like this):**

> Instead of tracking extra degrees of freedom inside AlphaFold 3's core neural
> network architecture, this PR introduces a non-linearly steered sampling
> routine. We inject a device-native **Proximal Operator** directly into the
> reverse-mode diffusion loop. This mathematical framework decouples local
> hydrogen force-field physics and global experimental constraints …

> Standard atomic updates cause **catastrophic** GPU thread collisions …

Tells: throat-clearing wind-up before the point; **bold** sprinkled to sound
important; hype adjectives (*device-native, catastrophic, monolithic*); the word
*anchor* used literally; claims of virtue with no number.

**PR #4 — his voice (write like this):**

> `--reference_cif 4BD0.cif` (X-ray) cost R_work 0.52 against 0.38 for 4BD1. The
> cell difference was the suspect; it is not the cause.

> A model placed on 4BD0 sits 82 Å from where 4BD1's data want it.

Every claim carries a number. The point leads. No word is there to impress. That
is the whole skill in two lines.
