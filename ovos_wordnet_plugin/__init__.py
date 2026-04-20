# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ovos-wordnet-plugin
~~~~~~~~~~~~~~~~~~~
OVOS plugin that exposes WordNet as a :class:`RetrievalEngine` and a
:class:`ToolBox` (``define_word`` + ``word_relations`` tools).

Uses the ``wn`` package with Open English WordNet (OEWN 2024), ODENet for
German, and OMW 1.4 packs for other languages.  Lexicons are downloaded on
first use; subsequent calls are served from the local cache.
"""
import os
import random
from os.path import dirname, join
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import wn as _wn
from ovos_config import Configuration
from ovos_plugin_manager.language import load_tx_plugin
from ovos_plugin_manager.templates.agent_tools import (
    AgentTool,
    ToolBox,
    ToolArguments,
    ToolOutput,
)
from ovos_plugin_manager.templates.agents import RetrievalEngine
from ovos_plugin_manager.templates.language import LanguageTranslator
from ovos_utils.log import LOG
from pydantic import Field
from simplematch import match as simplematch

# ---------------------------------------------------------------------------
# wn lexicon registry
# ---------------------------------------------------------------------------

# BCP-47 short tag → wn lexicon specifier
_WORDNET_IDS: Dict[str, str] = {
    "en": "oewn:2024",
    "de": "odenet:1.4",
    "ar": "omw-ar:1.4",
    "bg": "omw-bg:1.4",
    "ca": "omw-ca:1.4",
    "cmn": "omw-cmn:1.4",
    "da": "omw-da:1.4",
    "el": "omw-el:1.4",
    "es": "omw-es:1.4",
    "eu": "omw-eu:1.4",
    "fi": "omw-fi:1.4",
    "fr": "omw-fr:1.4",
    "gl": "omw-gl:1.4",
    "he": "omw-he:1.4",
    "hr": "omw-hr:1.4",
    "id": "omw-id:1.4",
    "is": "omw-is:1.4",
    "it": "omw-it:1.4",
    "ja": "omw-ja:1.4",
    "lt": "omw-lt:1.4",
    "nl": "omw-nl:1.4",
    "nb": "omw-nb:1.4",
    "nn": "omw-nn:1.4",
    "pl": "omw-pl:1.4",
    "pt": "omw-pt:1.4",
    "ro": "omw-ro:1.4",
    "sk": "omw-sk:1.4",
    "sl": "omw-sl:1.4",
    "sv": "omw-sv:1.4",
    "th": "omw-th:1.4",
    "zsm": "omw-zsm:1.4",
}

# OMW language packs require the English ILI map to resolve cross-lingual links.


# Track which lexicons have been confirmed present to avoid repeated fs checks.
_loaded_lexicons: Set[str] = set()

# ---------------------------------------------------------------------------
# POS constants (mirror NLTK values so existing callers keep working)
# ---------------------------------------------------------------------------

NOUN = "n"
VERB = "v"
ADJ = "a"
ADV = "r"

_POS_LABELS: Dict[str, str] = {
    NOUN: "noun",
    VERB: "verb",
    ADJ: "adjective",
    ADV: "adverb",
}

_ALL_POS: List[str] = [NOUN, VERB, ADJ, ADV]

_POS_SCORE: Dict[str, float] = {
    NOUN: 0.7,
    VERB: 0.7,
    ADJ: 0.65,
    ADV: 0.65,
}

_STR_TO_POS: Dict[str, str] = {
    "noun": NOUN,
    "verb": VERB,
    "adjective": ADJ,
    "adverb": ADV,
}

# ---------------------------------------------------------------------------
# wn bootstrap helpers
# ---------------------------------------------------------------------------


def _ensure_downloaded(lexicon_id: str) -> bool:
    """Download *lexicon_id* if not already in the local wn database.

    Args:
        lexicon_id: wn specifier such as ``"oewn:2024"`` or ``"omw-es:1.4"``.

    Returns:
        ``True`` when the lexicon is available (already present or just
        downloaded), ``False`` when the download fails.
    """
    if lexicon_id in _loaded_lexicons:
        return True
    existing = {f"{lex.id}:{lex.version}" for lex in _wn.lexicons()}
    if lexicon_id in existing:
        _loaded_lexicons.add(lexicon_id)
        return True
    try:
        LOG.info(f"WordnetPlugin: downloading {lexicon_id}")
        _wn.download(lexicon_id)
        _loaded_lexicons.add(lexicon_id)
        return True
    except Exception as exc:
        LOG.error(f"WordnetPlugin: failed to download {lexicon_id}: {exc}")
        return False


def _wordnet_for_lang(lang: str) -> Optional[_wn.Wordnet]:
    """Return a :class:`wn.Wordnet` object for *lang*, downloading if needed.

    Args:
        lang: BCP-47 language code (short or full, e.g. ``"es"`` or ``"es-ES"``).

    Returns:
        :class:`wn.Wordnet` instance, or ``None`` when the language is
        unsupported or the download fails.
    """
    short = lang.split("-")[0].lower()
    lexicon_id = _WORDNET_IDS.get(short)
    if not lexicon_id:
        return None
    if not _ensure_downloaded(lexicon_id):
        return None
    return _wn.Wordnet(lexicon_id)


# ---------------------------------------------------------------------------
# Low-level synset helpers
# ---------------------------------------------------------------------------


def _synset_lemmas(synset: _wn.Synset) -> List[str]:
    """Return lemma strings for *synset*, normalising underscores to spaces."""
    return [lem.replace("_", " ") for lem in synset.lemmas()]


def _related_lemmas(synset: _wn.Synset, relation: str) -> List[str]:
    """Return flat lemma list for synsets reachable via *relation* from *synset*."""
    result: List[str] = []
    for related in synset.get_related(relation):
        result.extend(_synset_lemmas(related))
    return result


def _holonym_lemmas(synset: _wn.Synset) -> List[str]:
    """Return member-holonym lemmas, trying OEWN relation name first."""
    # OEWN uses "holo_member"; other lexicons may use "holonym".
    lemmas = _related_lemmas(synset, "holo_member")
    if not lemmas:
        lemmas = _related_lemmas(synset, "holonym")
    return lemmas


def _antonym_lemmas(synset: _wn.Synset, wn_obj: _wn.Wordnet,
                    word: str, pos: str) -> List[str]:
    """Return antonym lemma strings for the first sense of *word* in *synset*.

    Antonyms live at the *sense* level in wn, not on synsets directly.

    Args:
        synset: The synset whose antonyms we want.
        wn_obj: The :class:`wn.Wordnet` used for the sense lookup.
        word: The target word (used to locate the specific sense).
        pos: POS constant (``NOUN``, ``VERB``, etc.).

    Returns:
        List of antonym strings, or empty list when none exist.
    """
    try:
        senses = wn_obj.senses(word, pos=pos)
        for sense in senses:
            if sense.synset() == synset:
                antonym_senses = sense.get_related("antonym")
                return [s.word().lemma().replace("_", " ") for s in antonym_senses]
    except Exception:
        pass
    return []


def _native_definition(synset: _wn.Synset) -> Optional[str]:
    """Return the native-language definition for *synset*, or ``None``."""
    defs = synset.definitions()
    return defs[0] if defs else None


def _root_hypernyms(synset: _wn.Synset) -> List[_wn.Synset]:
    """Return root hypernyms (top of the hypernym chain) for *synset*."""
    visited: Set[str] = set()
    roots: List[_wn.Synset] = []

    def _climb(s: _wn.Synset) -> None:
        key = s.id
        if key in visited:
            return
        visited.add(key)
        parents = s.hypernyms()
        if not parents:
            roots.append(s)
        else:
            for p in parents:
                _climb(p)

    _climb(synset)
    return roots


def _lowest_common_hypernyms(s1: _wn.Synset,
                              s2: _wn.Synset) -> List[_wn.Synset]:
    """Return the lowest common hypernyms of *s1* and *s2*.

    Traverses the hypernym closure of both synsets and returns the deepest
    nodes present in both ancestor sets.
    """

    def ancestors(s: _wn.Synset) -> Dict[str, _wn.Synset]:
        visited: Dict[str, _wn.Synset] = {}
        queue = [s]
        while queue:
            node = queue.pop()
            key = node.id
            if key in visited:
                continue
            visited[key] = node
            queue.extend(node.hypernyms())
        return visited

    anc1 = ancestors(s1)
    anc2 = ancestors(s2)
    common_keys = set(anc1) & set(anc2)
    if not common_keys:
        return []
    # "Lowest" = not dominated by any other common ancestor.
    common = {k: anc1[k] for k in common_keys}
    dominated: Set[str] = set()
    for k, node in common.items():
        for parent in node.hypernyms():
            if parent.id in common:
                dominated.add(parent.id)
    return [common[k] for k in common_keys if k not in dominated]


# ---------------------------------------------------------------------------
# High-level WordNet helper
# ---------------------------------------------------------------------------


class Wordnet:
    """Thin, stateless wrapper around the ``wn`` package.

    All public methods accept BCP-47 language strings and handle lexicon
    management internally.  Methods return empty collections rather than
    raising when a word or language is unknown.
    """

    @staticmethod
    def get_synsets(word: str, pos: str = NOUN, lang: str = "en") -> List[Any]:
        """Return all synsets for *word* with the given part-of-speech.

        Args:
            word: The word to look up.
            pos: POS constant (``NOUN``, ``VERB``, ``ADJ``, ``ADV``).
            lang: BCP-47 language code.

        Returns:
            List of :class:`wn.Synset` objects, or empty list on failure.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        return wn_obj.synsets(word, pos=pos)

    @staticmethod
    def get_definition(word: str, pos: str = NOUN,
                       synset: Any = None, lang: str = "en") -> Optional[str]:
        """Return the definition of the first matching synset.

        Falls back to English via ILI when the target language has no native
        definitions (most OMW 1.4 packs).

        Args:
            word: The word to define.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            Definition string, or ``None`` when nothing is found.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return None
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return None
            synset = synsets[0]
        return _native_definition(synset)

    @staticmethod
    def get_examples(word: str, pos: str = NOUN,
                     synset: Any = None, lang: str = "en") -> List[str]:
        """Return usage examples for the first matching synset.

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            List of example strings (may be empty).
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return synset.examples()

    @staticmethod
    def get_lemmas(word: str, pos: str = NOUN,
                   synset: Any = None, lang: str = "en") -> List[str]:
        """Return lemma names (synonyms within the same synset).

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            List of lemma strings with underscores replaced by spaces.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return _synset_lemmas(synset)

    @staticmethod
    def get_hypernyms(word: str, pos: str = NOUN,
                      synset: Any = None, lang: str = "en") -> List[str]:
        """Return hypernym lemma names (more general concepts, e.g. *animal* for *dog*).

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            Flat list of lemma strings from all direct hypernyms.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return _related_lemmas(synset, "hypernym")

    @staticmethod
    def get_hyponyms(word: str, pos: str = NOUN,
                     synset: Any = None, lang: str = "en") -> List[str]:
        """Return hyponym lemma names (more specific concepts, e.g. *poodle* for *dog*).

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            Flat list of lemma strings from all direct hyponyms.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return _related_lemmas(synset, "hyponym")

    @staticmethod
    def get_holonyms(word: str, pos: str = NOUN,
                     synset: Any = None, lang: str = "en") -> List[str]:
        """Return member-holonym lemma names (wholes that *word* is a member of).

        For example, *pack* is a holonym of *dog*.

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            Flat list of lemma strings from all member holonyms.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return _holonym_lemmas(synset)

    @staticmethod
    def get_root_hypernyms(word: str, pos: str = NOUN,
                           synset: Any = None, lang: str = "en") -> List[str]:
        """Return root hypernym lemma names (top of the hypernym chain).

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            Flat list of lemma strings from all root hypernyms.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return [lem for root in _root_hypernyms(synset)
                for lem in _synset_lemmas(root)]

    @staticmethod
    def get_antonyms(word: str, pos: str = NOUN,
                     synset: Any = None, lang: str = "en") -> List[str]:
        """Return antonym lemma names for the primary sense of the first synset.

        Args:
            word: The word to look up.
            pos: POS constant.
            synset: Pre-resolved synset; auto-resolved when ``None``.
            lang: BCP-47 language code.

        Returns:
            List of antonym strings (often empty — most senses have no antonym).
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        if synset is None:
            synsets = wn_obj.synsets(word, pos=pos)
            if not synsets:
                return []
            synset = synsets[0]
        return _antonym_lemmas(synset, wn_obj, word, pos)

    @staticmethod
    def common_hypernyms(word: str, word2: str,
                         pos: str = NOUN, lang: str = "en") -> List[str]:
        """Return the lowest common hypernyms shared by *word* and *word2*.

        Useful for answering "what do X and Y have in common?" questions.

        Args:
            word: First word.
            word2: Second word.
            pos: POS constant.
            lang: BCP-47 language code.

        Returns:
            List of lemma strings for the lowest common hypernyms, or empty
            list when either word is unknown.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return []
        s1 = wn_obj.synsets(word, pos=pos)
        s2 = wn_obj.synsets(word2, pos=pos)
        if not s1 or not s2:
            return []
        return [lem for lch in _lowest_common_hypernyms(s1[0], s2[0])
                for lem in _synset_lemmas(lch)]

    # ------------------------------------------------------------------
    # Composite helpers
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, word: str, pos: str = NOUN, lang: str = "en") -> Dict[str, Any]:
        """Return a dict with all lexical data for the *first* synset of *word*.

        Args:
            word: The word to look up.
            pos: POS constant.
            lang: BCP-47 language code.

        Returns:
            Dict with keys ``definition``, ``lemmas``, ``antonyms``, ``holonyms``,
            ``hyponyms``, ``hypernyms``, ``root_hypernyms``, ``examples``.
            Returns ``{}`` when unsupported or no synsets exist.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return {}
        synsets = wn_obj.synsets(word, pos=pos)
        if not synsets:
            return {}
        synset = synsets[0]
        return {
            "definition": cls.get_definition(word, pos=pos, synset=synset, lang=lang),
            "lemmas": cls.get_lemmas(word, pos=pos, synset=synset, lang=lang),
            "antonyms": cls.get_antonyms(word, pos=pos, synset=synset, lang=lang),
            "holonyms": cls.get_holonyms(word, pos=pos, synset=synset, lang=lang),
            "hyponyms": cls.get_hyponyms(word, pos=pos, synset=synset, lang=lang),
            "hypernyms": cls.get_hypernyms(word, pos=pos, synset=synset, lang=lang),
            "root_hypernyms": cls.get_root_hypernyms(word, pos=pos, synset=synset, lang=lang),
            "examples": cls.get_examples(word, pos=pos, synset=synset, lang=lang),
        }

    @classmethod
    def search(cls, word: str, pos: str = NOUN,
               lang: str = "en") -> Iterable[Dict[str, Any]]:
        """Yield lexical data dicts for *every* synset of *word*.

        Unlike :meth:`get`, which returns only the first synset, this iterates
        all senses — useful when a word is highly polysemous (e.g. *bank*).

        Args:
            word: The word to look up.
            pos: POS constant.
            lang: BCP-47 language code.

        Yields:
            One dict per synset with the same keys as :meth:`get`.
        """
        wn_obj = _wordnet_for_lang(lang)
        if wn_obj is None:
            return
        synsets = wn_obj.synsets(word, pos=pos)
        for synset in synsets:
            yield {
                "definition": cls.get_definition(word, pos=pos, synset=synset, lang=lang),
                "lemmas": cls.get_lemmas(word, pos=pos, synset=synset, lang=lang),
                "antonyms": cls.get_antonyms(word, pos=pos, synset=synset, lang=lang),
                "holonyms": cls.get_holonyms(word, pos=pos, synset=synset, lang=lang),
                "hyponyms": cls.get_hyponyms(word, pos=pos, synset=synset, lang=lang),
                "hypernyms": cls.get_hypernyms(word, pos=pos, synset=synset, lang=lang),
                "root_hypernyms": cls.get_root_hypernyms(word, pos=pos, synset=synset, lang=lang),
                "examples": cls.get_examples(word, pos=pos, synset=synset, lang=lang),
            }


# ---------------------------------------------------------------------------
# Locale helpers — intent detection and dialog rendering
# ---------------------------------------------------------------------------

_LOCALE_DIR = join(dirname(__file__), "locale")

_LOCALE_FOLDER_MAP: Dict[str, str] = {
    "en": "en-US", "ca": "ca-ES", "da": "da-DK", "de": "de-DE",
    "es": "es-ES", "eu": "eu-ES", "fr": "fr-FR", "gl": "gl-ES",
    "it": "it-IT", "pt": "pt-PT",
}

# Relations before "definition" so specific patterns don't get shadowed.
_INTENT_TYPES = ("antonym", "hypernym", "hyponym", "holonym", "lemma", "definition")


def _expand_alternatives(pattern: str) -> List[str]:
    """Expand a single padaos pattern line with ``(a|b)`` alternatives.

    ``"what is (a|the) {word}"`` → ``["what is a {word}", "what is the {word}"]``

    Args:
        pattern: A single intent line that may contain ``(opt1|opt2)`` groups.

    Returns:
        List of fully expanded patterns (one per combination).
    """
    import re as _re
    m = _re.search(r"\(([^)]+)\)", pattern)
    if not m:
        return [pattern]
    alternatives = m.group(1).split("|")
    results: List[str] = []
    for alt in alternatives:
        expanded = pattern[: m.start()] + alt + pattern[m.end():]
        results.extend(_expand_alternatives(expanded))
    return results


class LocaleIntentParser:
    """Parses a natural-language query against locale ``.intent`` files.

    Uses :func:`simplematch.match` so intent patterns are expressed in the
    same ``{slot}`` / ``(a|b)`` syntax as padacioso — no bespoke regex engine.

    Detects which WordNet relation the user is asking about and extracts
    the target ``{word}`` from the query string.
    """

    def __init__(self, lang: str = "en") -> None:
        """Load intent patterns for *lang*.

        Args:
            lang: BCP-47 language code (short or full, e.g. ``"en"`` or ``"en-US"``).
        """
        short = lang.split("-")[0].lower()
        folder = _LOCALE_FOLDER_MAP.get(short, "en-US")
        locale_path = join(_LOCALE_DIR, folder)
        if not os.path.isdir(locale_path):
            locale_path = join(_LOCALE_DIR, "en-US")

        self._patterns: Dict[str, List[str]] = {}
        for intent_type in _INTENT_TYPES:
            intent_file = join(locale_path, f"{intent_type}.intent")
            patterns: List[str] = []
            if os.path.isfile(intent_file):
                with open(intent_file, encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if line and not line.startswith("#"):
                            patterns.extend(_expand_alternatives(line))
            # Longer patterns first so specific ones beat short catch-alls.
            patterns.sort(key=len, reverse=True)
            self._patterns[intent_type] = patterns

    def parse(self, query: str) -> Optional[Tuple[str, str]]:
        """Match *query* against all loaded intent patterns.

        Args:
            query: The raw user query string.

        Returns:
            ``(intent_type, word)`` tuple on match, or ``None`` if no pattern
            matches.  *intent_type* is one of ``"definition"``, ``"antonym"``,
            ``"hypernym"``, ``"hyponym"``, ``"holonym"``, ``"lemma"``.
        """
        normalised = query.strip().rstrip("?!.,;").lower()
        for intent_type, patterns in self._patterns.items():
            for pattern in patterns:
                result = simplematch(pattern.rstrip("?!.,;").lower(), normalised)
                if result and "word" in result:
                    word = result["word"].strip().rstrip("?!.,;")
                    return intent_type, word
        return None


def _load_dialog(lang: str, dialog_name: str) -> List[str]:
    """Load lines from a ``.dialog`` file for *lang*.

    Args:
        lang: BCP-47 language code.
        dialog_name: Stem of the dialog file (e.g. ``"definition"``).

    Returns:
        List of template strings.  Falls back to ``en-US`` when absent.
    """
    short = lang.split("-")[0].lower()
    folder = _LOCALE_FOLDER_MAP.get(short, "en-US")
    for candidate in (join(_LOCALE_DIR, folder), join(_LOCALE_DIR, "en-US")):
        path = join(candidate, f"{dialog_name}.dialog")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
            if lines:
                return lines
    return []


def _render_dialog(lang: str, dialog_name: str, **kwargs: str) -> Optional[str]:
    """Pick a random dialog line and substitute *kwargs* variables.

    Args:
        lang: BCP-47 language code.
        dialog_name: Dialog file stem.
        **kwargs: Variable substitutions (e.g. ``word="dog"``, ``definition="…"``).

    Returns:
        Rendered string, or ``None`` when no dialog lines are available.
    """
    lines = _load_dialog(lang, dialog_name)
    if not lines:
        return None
    return random.choice(lines).format(**kwargs)


# ---------------------------------------------------------------------------
# RetrievalEngine
# ---------------------------------------------------------------------------

class WordnetRetrievalEngine(RetrievalEngine):
    """OVOS :class:`RetrievalEngine` backed by WordNet.

    When the incoming query matches a locale intent pattern (e.g. "what are
    the antonyms of dog?") the engine dispatches directly to the appropriate
    WordNet relation and returns a rendered dialog response.  Bare word
    lookups (no intent pattern matched) fall back to multi-sense definition
    passages.

    For languages whose OMW 1.4 pack has no native definitions (most non-EN/DE
    languages), the engine fetches the English OEWN definition and translates
    it into the target language using the configured translation plugin.

    Configuration keys (all optional):

    * ``lang`` – default BCP-47 language (falls back to OVOS system language).
    * ``translate_plugin`` – translation plugin ID (default: ``"ovos-translate-plugin-server"``).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None) -> None:
        super().__init__(config=config)
        self.translator: Optional[LanguageTranslator] = translator
        if not translator:
            self._load_translator()

    def _load_translator(self) -> None:
        lang_cfg = Configuration().get("language", {})
        plug_id = (self.config.get("translate_plugin")
                   or lang_cfg.get("translation_module", "ovos-translate-plugin-server"))
        clazz = load_tx_plugin(plug_id)
        if clazz is None:
            LOG.warning(f"WordnetPlugin: translation plugin '{plug_id}' not found — "
                        "non-English definitions will be skipped when no native gloss exists")
        else:
            self.translator = clazz(config=lang_cfg.get(plug_id, {}))
            LOG.debug(f"WordnetPlugin: loaded translation plugin '{plug_id}'")

    def _translate(self, text: str, target: str) -> str:
        """Translate *text* from English to *target*, or return *text* unchanged."""
        if not self.translator or target == "en":
            return text
        try:
            return self.translator.translate(text, target=target, source="en") or text
        except Exception as exc:
            LOG.debug(f"WordnetPlugin: translation failed: {exc}")
            return text

    def _get_definition(self, word: str, pos: str, synset: Any,
                        lang: str) -> Optional[str]:
        """Return the best available definition, translating from English when needed.

        Args:
            word: Target word (used to look up English synset via ILI).
            pos: POS constant.
            synset: Synset from the target-language lexicon.
            lang: BCP-47 short language code.

        Returns:
            Definition string in *lang*, or ``None`` when unavailable.
        """
        native = _native_definition(synset)
        if native:
            return native

        if lang == "en":
            return None

        # No native definition — fetch English OEWN definition and translate.
        en_def = Wordnet.get_definition(word, pos=pos, lang="en")
        if not en_def:
            # Try ILI crosswalk as fallback for words not in OEWN by lemma.
            try:
                ili = synset.ili
                if ili:
                    import wn as _wn2
                    if _ensure_downloaded("oewn:2024"):
                        en_wn = _wn2.Wordnet("oewn:2024")
                        en_ss = en_wn.synsets(ili=ili)
                        if en_ss:
                            defs = en_ss[0].definitions()
                            en_def = defs[0] if defs else None
            except Exception:
                pass
        if not en_def:
            return None
        return self._translate(en_def, target=lang)

    def query(self, query: str, lang: Optional[str] = None,
              k: int = 15) -> List[Tuple[str, float]]:
        """Look up *query* in WordNet and return scored natural-language passages.

        If the query matches a locale intent pattern the engine resolves the
        requested relation for the extracted word and returns a single rendered
        dialog line.  Otherwise it falls back to iterating all senses across
        all parts of speech until *k* passages are collected.

        Args:
            query: Raw user query or bare word to look up.
            lang: BCP-47 language code.  Falls back to the engine's configured
                language when omitted.
            k: Maximum number of passages to return.

        Returns:
            List of ``(passage, score)`` tuples.  Intent-matched results carry
            a score of 0.9; bare definition passages use 0.70 / 0.65 by POS.
        """
        lang = lang or self.lang
        short_lang = lang.split("-")[0]

        parser = LocaleIntentParser(lang)
        parsed = parser.parse(query)
        if parsed:
            intent_type, word = parsed
            passage = self._resolve_intent(intent_type, word, short_lang)
            if passage:
                LOG.debug(f"WordnetSolver: intent={intent_type!r} word={word!r}")
                return [(passage, 0.9)]
            not_found = _render_dialog(lang, "not_found", word=word)
            if not_found:
                return [(not_found, 0.0)]
            return []

        results: List[Tuple[str, float]] = []
        wn_obj = _wordnet_for_lang(short_lang)
        if wn_obj is not None:
            for pos in _ALL_POS:
                for synset in wn_obj.synsets(query, pos=pos):
                    defn = self._get_definition(query, pos=pos, synset=synset,
                                                lang=short_lang)
                    if not defn:
                        continue

                    pos_label = _POS_LABELS[pos]
                    parts = [f"{query} ({pos_label}): {defn}."]

                    lemmas = _synset_lemmas(synset)
                    syns = [s for s in lemmas if s.lower() != query.lower()]
                    if syns:
                        parts.append(f"Also known as: {', '.join(syns[:4])}.")

                    if examples := synset.examples():
                        parts.append(f"Example: {examples[0]}")

                    results.append((" ".join(parts), _POS_SCORE[pos]))
                    if len(results) >= k:
                        LOG.debug(f"WordnetSolver: '{query}' -> {len(results)} passages")
                        return results

        LOG.debug(f"WordnetSolver: '{query}' -> {len(results)} passages")
        return results

    def _resolve_intent(self, intent_type: str, word: str, lang: str) -> Optional[str]:
        """Dispatch an intent to the appropriate WordNet method and render a dialog line.

        Args:
            intent_type: One of the ``_INTENT_TYPES`` strings.
            word: The target word extracted from the query.
            lang: BCP-47 short language code (e.g. ``"en"``).

        Returns:
            Rendered dialog string, or ``None`` when WordNet has no data.
        """
        for pos in _ALL_POS:
            data = Wordnet.get(word, pos=pos, lang=lang)
            if not data:
                continue

            if intent_type == "definition":
                wn_obj = _wordnet_for_lang(lang)
                synsets = wn_obj.synsets(word, pos=pos) if wn_obj else []
                synset = synsets[0] if synsets else None
                defn = self._get_definition(word, pos=pos, synset=synset, lang=lang) if synset else None
                if defn:
                    return _render_dialog(lang, "definition", word=word, definition=defn)

            elif intent_type == "antonym":
                items = data.get("antonyms") or []
                if items:
                    return _render_dialog(lang, "antonym", word=word,
                                         output_word=", ".join(items[:5]))

            elif intent_type == "hypernym":
                items = data.get("hypernyms") or []
                if items:
                    return _render_dialog(lang, "hypernym", word=word,
                                         output_word=", ".join(items[:5]))

            elif intent_type == "hyponym":
                items = data.get("hyponyms") or []
                if items:
                    return _render_dialog(lang, "hyponym", word=word,
                                         output_word=", ".join(items[:5]))

            elif intent_type == "holonym":
                items = data.get("holonyms") or []
                if items:
                    return _render_dialog(lang, "holonym", word=word,
                                         output_word=", ".join(items[:5]))

            elif intent_type == "lemma":
                items = [s for s in (data.get("lemmas") or [])
                         if s.lower() != word.lower()]
                if items:
                    return _render_dialog(lang, "lemma", word=word,
                                         output_word=", ".join(items[:5]))

        return None


# ---------------------------------------------------------------------------
# ToolBox
# ---------------------------------------------------------------------------

class DefineWordArgs(ToolArguments):
    """Input arguments for the ``define_word`` tool."""
    word: str = Field(..., description="The word to define.")
    lang: str = Field("en", description="BCP-47 language code, e.g. 'en', 'es', 'fr'.")
    pos: str = Field(
        "any",
        description="Part of speech filter: 'noun', 'verb', 'adjective', 'adverb', or 'any'.",
    )


class DefineWordOutput(ToolOutput):
    """Output from the ``define_word`` tool."""
    definitions: List[Tuple[str, str]] = Field(
        ...,
        description="List of (part_of_speech, definition) pairs for the word.",
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Usage examples from WordNet.",
    )


class WordRelationsArgs(ToolArguments):
    """Input arguments for the ``word_relations`` tool."""
    word: str = Field(..., description="The word to look up relations for.")
    lang: str = Field("en", description="BCP-47 language code, e.g. 'en', 'es', 'fr'.")
    pos: str = Field(
        "noun",
        description="Part of speech: 'noun', 'verb', 'adjective', or 'adverb'.",
    )


class WordRelationsOutput(ToolOutput):
    """Output from the ``word_relations`` tool."""
    synonyms: List[str] = Field(
        default_factory=list,
        description="Synonyms (lemmas within the same synset).",
    )
    antonyms: List[str] = Field(
        default_factory=list,
        description="Antonyms of the word.",
    )
    hypernyms: List[str] = Field(
        default_factory=list,
        description="More general terms (e.g. 'animal' is a hypernym of 'dog').",
    )
    hyponyms: List[str] = Field(
        default_factory=list,
        description="More specific terms (e.g. 'poodle' is a hyponym of 'dog').",
    )
    holonyms: List[str] = Field(
        default_factory=list,
        description="Wholes that this word is a member of (e.g. 'pack' for 'dog').",
    )


class WordnetToolbox(ToolBox):
    """OVOS :class:`ToolBox` that exposes two WordNet-backed tools.

    Tools
    -----
    ``define_word``
        Returns definitions grouped by part of speech, plus usage examples.
    ``word_relations``
        Returns synonyms, antonyms, hypernyms, hyponyms, and holonyms.
    """

    toolbox_id = "ovos-wordnet-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(toolbox_id=self.toolbox_id)

    def define_word(self, args: DefineWordArgs) -> DefineWordOutput:
        """Look up definitions for *args.word*, optionally filtered by POS.

        Args:
            args: Validated :class:`DefineWordArgs`.

        Returns:
            :class:`DefineWordOutput` with definitions and usage examples.
        """
        pos_filter = _STR_TO_POS.get(args.pos.lower())
        candidates = [pos_filter] if pos_filter else _ALL_POS

        definitions: List[Tuple[str, str]] = []
        examples: List[str] = []
        for pos in candidates:
            data = Wordnet.get(args.word, pos=pos, lang=args.lang)
            if data.get("definition"):
                definitions.append((_POS_LABELS[pos], data["definition"]))
                examples.extend(data.get("examples") or [])

        return DefineWordOutput(definitions=definitions, examples=examples)

    def word_relations(self, args: WordRelationsArgs) -> WordRelationsOutput:
        """Return lexical relations for *args.word*.

        Args:
            args: Validated :class:`WordRelationsArgs`.

        Returns:
            :class:`WordRelationsOutput` with synonyms, antonyms, hypernyms,
            hyponyms, and holonyms.
        """
        pos = _STR_TO_POS.get(args.pos.lower(), NOUN)
        data = Wordnet.get(args.word, pos=pos, lang=args.lang)
        return WordRelationsOutput(
            synonyms=data.get("lemmas") or [],
            antonyms=data.get("antonyms") or [],
            hypernyms=data.get("hypernyms") or [],
            hyponyms=data.get("hyponyms") or [],
            holonyms=data.get("holonyms") or [],
        )

    def discover_tools(self) -> List[AgentTool]:
        """Register the two WordNet tools with the OVOS agent framework.

        Returns:
            List containing the ``define_word`` and ``word_relations`` tools.
        """
        return [
            AgentTool(
                name="define_word",
                description=(
                    "Look up the definition of a word using WordNet. "
                    "Returns one or more definitions grouped by part of speech, "
                    "plus usage examples. Supports multiple languages."
                ),
                argument_schema=DefineWordArgs,
                output_schema=DefineWordOutput,
                tool_call=self.define_word,
            ),
            AgentTool(
                name="word_relations",
                description=(
                    "Look up lexical relations for a word: synonyms, antonyms, "
                    "hypernyms (broader terms), hyponyms (narrower terms), and holonyms "
                    "(groups/wholes the word belongs to). "
                    "Useful for thesaurus-style queries and semantic reasoning."
                ),
                argument_schema=WordRelationsArgs,
                output_schema=WordRelationsOutput,
                tool_call=self.word_relations,
            ),
        ]


if __name__ == "__main__":
    engine = WordnetRetrievalEngine()
    for text, score in engine.query("bank", lang="en"):
        print(f"[{score:.2f}] {text}")

    print()
    tb = WordnetToolbox()
    print(tb.define_word(DefineWordArgs(word="bank", lang="en")))
    print(tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun")))
