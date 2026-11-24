"""Policy guardrail that runs BEFORE the model.

A small model can lose an instruction buried above ~10K tokens of course material, so requests to
produce graded work are caught here deterministically and never reach the LLM. The model's system
prompt still carries the same rule as a second layer for phrasings this misses.
"""
import re

_PRODUCE = (r"(write|draft|compose|generate|create|produce|give me|make|do|finish|complete|rewrite|reword|"
            r"paraphrase|fill in|type up|type out|put together|come up with|spit out)")
_WORK = (r"(paragraphs?|sentences?|sections?|essay|paper|literature review|lit review|methodology|methods section|"
         r"introduction|intro|abstract|conclusion|thesis|annotated bibliography|annotations?|research question|"
         r"proposal|stage \d|assignment|reflection|discussion post|outline for (our|my))")
_PATTERN = re.compile(rf"\b{_PRODUCE}\b.{{0,60}}\b{_WORK}\b", re.I | re.S)

# Questions ABOUT the work ("what should a methodology section include?") are fine.
_ABOUT = re.compile(r"^\s*(what|how|why|when|where|which|is|are|does|do i|should|can i use)\b", re.I)

REFUSAL = (
    "I can't write or draft any part of a graded assignment. This course's LLM policy doesn't allow it, "
    "and work flagged as AI-written can be referred for a conduct investigation. "
    "Here's what I can do instead:\n"
    "- Explain what that section needs, based on the assignment description and lectures.\n"
    "- Help you brainstorm or narrow your ideas by asking you questions.\n"
    "- Point you to the right week's materials or the UW Library research guide.\n"
    "- Check basic grammar and spelling on something you wrote yourself.\n"
    "Want to start with what this section is supposed to cover?"
)


def check(question: str):
    """Return a refusal message if the question asks for graded work to be produced, else None."""
    if _PATTERN.search(question) and not _ABOUT.match(question):
        return REFUSAL
    return None
