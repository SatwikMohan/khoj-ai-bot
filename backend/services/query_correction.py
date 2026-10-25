"""Conservative query spelling repair using only retrieved local passages."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher


_WORD = re.compile(r"[A-Za-z]{3,}")
_PROTECTED = {
    "not", "never", "without", "except", "no", "nor", "none",
    "what", "when", "where", "which", "whose", "would", "could",
    "about", "again", "answer", "compare", "describe", "explain",
    "english", "hindi", "hinglish", "latest", "current", "before", "after",
}


def _distance(left: str, right: str, maximum: int = 2) -> int:
    """Bound edit work; a larger distance cannot be a safe automatic repair."""
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, char in enumerate(left, 1):
        current = [row]
        for column, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (char != other)))
        if min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def correct_query_from_passages(question: str, documents: list) -> str:
    """Repair clear typos without changing identifiers, dates, or user language.

    A candidate must occur in a retrieved passage and agree with neighboring
    query words. Single-word repairs require a unique, repeated near match.
    The displayed question remains the person's original text.
    """
    query_matches = list(_WORD.finditer(question))
    if not documents or not query_matches:
        return question
    query_words = [match.group().casefold() for match in query_matches]
    passage_words = [
        _WORD.findall(str(doc.page_content)[:6000].casefold())
        for doc in documents[:5]
    ]
    counts = Counter(word for words in passage_words for word in words)
    if not counts:
        return question
    occurrences: dict[str, list[tuple[list[str], int]]] = {}
    for words in passage_words:
        for position, word in enumerate(words):
            occurrences.setdefault(word, []).append((words, position))

    def anchors(word: str, index: int) -> int:
        best = 0
        for words, position in occurrences.get(word, ()):
            matched = 0
            for offset in (1, 2):
                if index - offset >= 0 and position - offset >= 0 and \
                        query_words[index - offset] == words[position - offset]:
                    matched += 1
                if index + offset < len(query_words) and position + offset < len(words) and \
                        query_words[index + offset] == words[position + offset]:
                    matched += 1
            best = max(best, matched)
        return best

    replacements = {}
    for index, match in enumerate(query_matches):
        original = match.group()
        word = query_words[index]
        if original.isupper() or word in _PROTECTED or len(word) < 4:
            continue
        existing_anchors = anchors(word, index)
        candidates = []
        for candidate, count in counts.items():
            if (candidate == word or candidate[0] != word[0]
                    or abs(len(candidate) - len(word)) > 2):
                continue
            anagram = sorted(candidate) == sorted(word)
            distance = _distance(word, candidate)
            if not anagram and (distance > 2 or
                                SequenceMatcher(None, word, candidate).ratio() < 0.65):
                continue
            agreement = anchors(candidate, index)
            strong = agreement >= 2 and (anagram or distance <= 2)
            close = agreement >= 1 and distance == 1 and len(word) >= 5
            isolated = (len(query_words) == 1 and word not in counts and len(word) >= 6 and
                        distance == 1 and count >= 2)
            if not (strong or close or isolated) or (agreement <= existing_anchors and not isolated):
                continue
            candidates.append((agreement, -distance, count, candidate))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        best = candidates[0]
        if len(candidates) > 1 and candidates[1][:2] == best[:2]:
            continue
        replacement = best[3]
        if original.istitle():
            replacement = replacement.title()
        replacements[index] = replacement
        if len(replacements) == 2:
            break
    if not replacements:
        return question
    parts = []
    cursor = 0
    for index, match in enumerate(query_matches):
        if index not in replacements:
            continue
        parts.extend((question[cursor:match.start()], replacements[index]))
        cursor = match.end()
    parts.append(question[cursor:])
    return "".join(parts)
