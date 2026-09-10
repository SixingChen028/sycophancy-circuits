"""Word-level overlap checks for the description-based opinion condition.

The point of that condition is that the opinion shares no surface form with the
answer it selects, so any surface path from opinion to answer has to be removed
mechanically (paper, Appendix A.2).

Two concerns, handled at two different granularities on purpose:

  The option LETTER is a direct bridge to the measured answer token, so it is
  excluded strictly, by regular expression.

  CONTENT OVERLAP would let the model string-match the description to the option
  without using its meaning. That is a lexical concern, so it is checked at WORD
  level. Sub-word tokens are the wrong unit: tokenizers split "chloroplast" into
  pieces that recur in unrelated words, so filtering on shared sub-word tokens
  rejects heavily without removing any route the model could actually exploit.

Stdlib only, deliberately. This module is imported both by the generation script,
which needs the Anthropic client, and by the screening script, which runs on a
GPU node where that client is neither installed nor needed.
"""

import re

STOPWORDS = set(
    """a an the of and or in on to is are was were be been being by for with as at
from that this these those it its their his her they he she we you i not no nor but if then than
which who whom whose what when where how why all any both each few more most other some such only
own same so too very can will just should now do does did done have has had having up down out off
over under again further once here there when both between into through during before after above
below""".split()
)

__all__ = ["content_words", "overlap_words"]


def content_words(text):
    """Lower-cased alphanumeric words, stopwords removed."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def overlap_words(description, target_text, letter):
    """Offending overlaps between a description and its target option.

    Returns the shared content words, plus a marker if the option's letter
    appears in the description. An empty set means the description is usable.
    """
    shared = content_words(description) & content_words(target_text)
    if re.search(rf"\b\(?{letter}\)?\b", description):
        shared = shared | {f"letter {letter}"}
    return shared
