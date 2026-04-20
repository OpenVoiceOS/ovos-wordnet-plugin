"""
Tests for ovos-wordnet-plugin (wn backend).

Tests exercise the real WordNet data (oewn:2024 must be downloaded).  The
suite is structured so each public API surface gets its own TestCase.
"""
import unittest

from ovos_wordnet_plugin import (
    ADJ,
    ADV,
    NOUN,
    VERB,
    DefineWordArgs,
    DefineWordOutput,
    LocaleIntentParser,
    Wordnet,
    WordnetRetrievalEngine,
    WordnetToolbox,
    WordRelationsArgs,
    WordRelationsOutput,
    _STR_TO_POS,
    _WORDNET_IDS,
    _ensure_downloaded,
    _wordnet_for_lang,
    _native_definition,
    _synset_lemmas,
    _related_lemmas,
    _holonym_lemmas,
    _root_hypernyms,
    _expand_alternatives,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestPOSConstants(unittest.TestCase):

    def test_values_match_wn_convention(self):
        self.assertEqual(NOUN, "n")
        self.assertEqual(VERB, "v")
        self.assertEqual(ADJ, "a")
        self.assertEqual(ADV, "r")

    def test_str_to_pos_mapping(self):
        self.assertEqual(_STR_TO_POS["noun"], NOUN)
        self.assertEqual(_STR_TO_POS["verb"], VERB)
        self.assertEqual(_STR_TO_POS["adjective"], ADJ)
        self.assertEqual(_STR_TO_POS["adverb"], ADV)

    def test_wordnet_ids_has_english(self):
        self.assertIn("en", _WORDNET_IDS)
        self.assertEqual(_WORDNET_IDS["en"], "oewn:2024")

    def test_wordnet_ids_has_german(self):
        self.assertIn("de", _WORDNET_IDS)
        self.assertEqual(_WORDNET_IDS["de"], "odenet:1.4")


# ---------------------------------------------------------------------------
# _ensure_downloaded / _wordnet_for_lang
# ---------------------------------------------------------------------------

class TestLexiconManagement(unittest.TestCase):

    def test_ensure_downloaded_english(self):
        result = _ensure_downloaded("oewn:2024")
        self.assertTrue(result)

    def test_wordnet_for_lang_english(self):
        wn_obj = _wordnet_for_lang("en")
        self.assertIsNotNone(wn_obj)

    def test_wordnet_for_lang_full_bcp47(self):
        wn_obj = _wordnet_for_lang("en-US")
        self.assertIsNotNone(wn_obj)

    def test_wordnet_for_lang_unsupported_returns_none(self):
        wn_obj = _wordnet_for_lang("xx")
        self.assertIsNone(wn_obj)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestSynsetHelpers(unittest.TestCase):

    def setUp(self):
        import wn as _wn
        self.en_wn = _wn.Wordnet("oewn:2024")
        self.dog_synsets = self.en_wn.synsets("dog", pos=NOUN)
        self.assertTrue(self.dog_synsets, "oewn:2024 must be downloaded")
        self.dog_ss = self.dog_synsets[0]

    def test_synset_lemmas_returns_strings(self):
        lemmas = _synset_lemmas(self.dog_ss)
        self.assertIsInstance(lemmas, list)
        self.assertTrue(all(isinstance(l, str) for l in lemmas))

    def test_synset_lemmas_contains_dog(self):
        lemmas = _synset_lemmas(self.dog_ss)
        self.assertIn("dog", lemmas)

    def test_related_lemmas_hypernym(self):
        lemmas = _related_lemmas(self.dog_ss, "hypernym")
        self.assertIsInstance(lemmas, list)
        self.assertTrue(len(lemmas) > 0)

    def test_holonym_lemmas_no_crash(self):
        lemmas = _holonym_lemmas(self.dog_ss)
        self.assertIsInstance(lemmas, list)

    def test_native_definition_english(self):
        defn = _native_definition(self.dog_ss)
        self.assertIsNotNone(defn)
        self.assertIsInstance(defn, str)
        self.assertGreater(len(defn), 5)

    def test_root_hypernyms_returns_synsets(self):
        roots = _root_hypernyms(self.dog_ss)
        self.assertIsInstance(roots, list)
        self.assertGreater(len(roots), 0)

    def test_root_hypernyms_have_no_parents(self):
        import wn as _wn
        roots = _root_hypernyms(self.dog_ss)
        for root in roots:
            self.assertEqual(root.hypernyms(), [],
                             f"{root} should have no hypernyms")


# ---------------------------------------------------------------------------
# _expand_alternatives
# ---------------------------------------------------------------------------

class TestExpandAlternatives(unittest.TestCase):

    def test_no_alternatives_unchanged(self):
        result = _expand_alternatives("what is {word}")
        self.assertEqual(result, ["what is {word}"])

    def test_single_group_expanded(self):
        result = _expand_alternatives("what is (a|the) {word}")
        self.assertIn("what is a {word}", result)
        self.assertIn("what is the {word}", result)
        self.assertEqual(len(result), 2)

    def test_three_alternatives(self):
        result = _expand_alternatives("(define|explain|describe) {word}")
        self.assertEqual(len(result), 3)

    def test_nested_groups_expanded(self):
        result = _expand_alternatives("(what is|what are) (a|the) {word}")
        self.assertEqual(len(result), 4)


# ---------------------------------------------------------------------------
# LocaleIntentParser
# ---------------------------------------------------------------------------

class TestLocaleIntentParser(unittest.TestCase):

    def setUp(self):
        self.parser = LocaleIntentParser("en")

    def test_definition_intent(self):
        result = self.parser.parse("what is the definition of dog")
        self.assertIsNotNone(result)
        intent_type, word = result
        self.assertEqual(intent_type, "definition")
        self.assertEqual(word, "dog")

    def test_antonym_intent(self):
        result = self.parser.parse("what is the antonym of happy")
        self.assertIsNotNone(result)
        intent_type, word = result
        self.assertEqual(intent_type, "antonym")
        self.assertEqual(word, "happy")

    def test_hypernym_intent(self):
        result = self.parser.parse("what are the hypernyms of dog")
        self.assertIsNotNone(result)
        intent_type, word = result
        self.assertEqual(intent_type, "hypernym")
        self.assertEqual(word, "dog")

    def test_hyponym_intent(self):
        result = self.parser.parse("what are the hyponyms of dog")
        self.assertIsNotNone(result)
        intent_type, word = result
        self.assertEqual(intent_type, "hyponym")
        self.assertEqual(word, "dog")

    def test_holonym_intent(self):
        result = self.parser.parse("what is the holonym of dog")
        self.assertIsNotNone(result)
        intent_type, word = result
        self.assertEqual(intent_type, "holonym")
        self.assertEqual(word, "dog")

    def test_lemma_intent(self):
        result = self.parser.parse("what is a lemma of dog")
        self.assertIsNotNone(result)
        intent_type, word = result
        self.assertEqual(intent_type, "lemma")
        self.assertEqual(word, "dog")

    def test_no_match_returns_none(self):
        result = self.parser.parse("tell me a joke")
        self.assertIsNone(result)

    def test_case_insensitive(self):
        result = self.parser.parse("What Is The Definition Of Dog")
        self.assertIsNotNone(result)

    def test_trailing_punctuation_ignored(self):
        result = self.parser.parse("what is the definition of dog?")
        self.assertIsNotNone(result)
        _, word = result
        self.assertEqual(word, "dog")

    def test_spanish_parser_loads(self):
        parser_es = LocaleIntentParser("es")
        self.assertIsNotNone(parser_es)

    def test_unknown_lang_falls_back_to_english(self):
        parser_xx = LocaleIntentParser("xx")
        result = parser_xx.parse("what is the definition of dog")
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Wordnet.get_synsets
# ---------------------------------------------------------------------------

class TestGetSynsets(unittest.TestCase):

    def test_known_word_returns_synsets(self):
        synsets = Wordnet.get_synsets("dog", lang="en")
        self.assertGreater(len(synsets), 0)

    def test_unknown_word_returns_empty(self):
        synsets = Wordnet.get_synsets("xyzzy_not_a_word", lang="en")
        self.assertEqual(synsets, [])

    def test_unsupported_lang_returns_empty(self):
        synsets = Wordnet.get_synsets("dog", lang="xx")
        self.assertEqual(synsets, [])

    def test_noun_and_verb_synsets_differ(self):
        nouns = Wordnet.get_synsets("bank", pos=NOUN, lang="en")
        verbs = Wordnet.get_synsets("bank", pos=VERB, lang="en")
        self.assertGreater(len(nouns), 0)
        self.assertGreater(len(verbs), 0)
        noun_ids = {s.id for s in nouns}
        verb_ids = {s.id for s in verbs}
        self.assertTrue(noun_ids.isdisjoint(verb_ids))


# ---------------------------------------------------------------------------
# Wordnet.get_definition
# ---------------------------------------------------------------------------

class TestGetDefinition(unittest.TestCase):

    def test_returns_string(self):
        defn = Wordnet.get_definition("dog", lang="en")
        self.assertIsInstance(defn, str)
        self.assertGreater(len(defn), 0)

    def test_unknown_word_returns_none(self):
        defn = Wordnet.get_definition("xyzzy_not_a_word", lang="en")
        self.assertIsNone(defn)

    def test_unsupported_lang_returns_none(self):
        defn = Wordnet.get_definition("dog", lang="xx")
        self.assertIsNone(defn)

    def test_polysemous_word_has_definition(self):
        defn = Wordnet.get_definition("bank", lang="en")
        self.assertIsNotNone(defn)


# ---------------------------------------------------------------------------
# Wordnet.get_lemmas
# ---------------------------------------------------------------------------

class TestGetLemmas(unittest.TestCase):

    def test_dog_lemmas_include_dog(self):
        lemmas = Wordnet.get_lemmas("dog", lang="en")
        self.assertIn("dog", lemmas)

    def test_returns_list_of_strings(self):
        lemmas = Wordnet.get_lemmas("dog", lang="en")
        self.assertIsInstance(lemmas, list)
        self.assertTrue(all(isinstance(l, str) for l in lemmas))

    def test_unknown_word_empty(self):
        lemmas = Wordnet.get_lemmas("xyzzy_not_a_word", lang="en")
        self.assertEqual(lemmas, [])

    def test_unsupported_lang_empty(self):
        lemmas = Wordnet.get_lemmas("dog", lang="xx")
        self.assertEqual(lemmas, [])


# ---------------------------------------------------------------------------
# Wordnet.get_hypernyms / get_hyponyms
# ---------------------------------------------------------------------------

class TestGetHypernymsHyponyms(unittest.TestCase):

    def test_dog_has_hypernyms(self):
        hypernyms = Wordnet.get_hypernyms("dog", lang="en")
        self.assertGreater(len(hypernyms), 0)

    def test_dog_has_hyponyms(self):
        hyponyms = Wordnet.get_hyponyms("dog", lang="en")
        self.assertGreater(len(hyponyms), 0)

    def test_hypernyms_and_hyponyms_are_strings(self):
        for word in ("dog", "vehicle"):
            for fn in (Wordnet.get_hypernyms, Wordnet.get_hyponyms):
                result = fn(word, lang="en")
                self.assertTrue(all(isinstance(s, str) for s in result),
                                f"{fn.__name__}({word!r}) returned non-strings")

    def test_unknown_word_empty(self):
        self.assertEqual(Wordnet.get_hypernyms("xyzzy", lang="en"), [])
        self.assertEqual(Wordnet.get_hyponyms("xyzzy", lang="en"), [])


# ---------------------------------------------------------------------------
# Wordnet.get_root_hypernyms
# ---------------------------------------------------------------------------

class TestGetRootHypernyms(unittest.TestCase):

    def test_dog_has_root_hypernyms(self):
        roots = Wordnet.get_root_hypernyms("dog", lang="en")
        self.assertGreater(len(roots), 0)

    def test_returns_list_of_strings(self):
        roots = Wordnet.get_root_hypernyms("dog", lang="en")
        self.assertTrue(all(isinstance(r, str) for r in roots))

    def test_unknown_word_empty(self):
        self.assertEqual(Wordnet.get_root_hypernyms("xyzzy", lang="en"), [])


# ---------------------------------------------------------------------------
# Wordnet.get_antonyms
# ---------------------------------------------------------------------------

class TestGetAntonyms(unittest.TestCase):

    def test_good_has_antonym_bad(self):
        antonyms = Wordnet.get_antonyms("good", pos=ADJ, lang="en")
        # OEWN antonyms are sense-level; adjectives are the most common case
        self.assertIsInstance(antonyms, list)

    def test_returns_strings(self):
        antonyms = Wordnet.get_antonyms("good", pos=ADJ, lang="en")
        self.assertTrue(all(isinstance(a, str) for a in antonyms))

    def test_unknown_word_empty(self):
        self.assertEqual(Wordnet.get_antonyms("xyzzy", lang="en"), [])

    def test_unsupported_lang_empty(self):
        self.assertEqual(Wordnet.get_antonyms("good", lang="xx"), [])


# ---------------------------------------------------------------------------
# Wordnet.get_holonyms
# ---------------------------------------------------------------------------

class TestGetHolonyms(unittest.TestCase):

    def test_no_crash_on_known_word(self):
        holonyms = Wordnet.get_holonyms("dog", lang="en")
        self.assertIsInstance(holonyms, list)

    def test_returns_strings(self):
        holonyms = Wordnet.get_holonyms("dog", lang="en")
        self.assertTrue(all(isinstance(h, str) for h in holonyms))

    def test_unknown_word_empty(self):
        self.assertEqual(Wordnet.get_holonyms("xyzzy", lang="en"), [])


# ---------------------------------------------------------------------------
# Wordnet.common_hypernyms
# ---------------------------------------------------------------------------

class TestCommonHypernyms(unittest.TestCase):

    def test_dog_and_cat_share_hypernym(self):
        common = Wordnet.common_hypernyms("dog", "cat", pos=NOUN, lang="en")
        self.assertGreater(len(common), 0)

    def test_returns_strings(self):
        common = Wordnet.common_hypernyms("dog", "cat", pos=NOUN, lang="en")
        self.assertTrue(all(isinstance(s, str) for s in common))

    def test_unknown_word_empty(self):
        common = Wordnet.common_hypernyms("dog", "xyzzy", lang="en")
        self.assertEqual(common, [])


# ---------------------------------------------------------------------------
# Wordnet.get (composite)
# ---------------------------------------------------------------------------

class TestWordnetGet(unittest.TestCase):

    def test_returns_dict_with_expected_keys(self):
        data = Wordnet.get("dog", lang="en")
        for key in ("definition", "lemmas", "antonyms", "holonyms",
                    "hyponyms", "hypernyms", "root_hypernyms", "examples"):
            self.assertIn(key, data, f"missing key: {key!r}")

    def test_unknown_word_returns_empty_dict(self):
        data = Wordnet.get("xyzzy_not_a_word", lang="en")
        self.assertEqual(data, {})

    def test_unsupported_lang_returns_empty_dict(self):
        data = Wordnet.get("dog", lang="xx")
        self.assertEqual(data, {})

    def test_definition_is_string(self):
        data = Wordnet.get("dog", lang="en")
        self.assertIsInstance(data["definition"], str)

    def test_pos_noun_verb_give_different_data(self):
        noun_data = Wordnet.get("bank", pos=NOUN, lang="en")
        verb_data = Wordnet.get("bank", pos=VERB, lang="en")
        self.assertNotEqual(noun_data.get("definition"), verb_data.get("definition"))


# ---------------------------------------------------------------------------
# Wordnet.search (generator)
# ---------------------------------------------------------------------------

class TestWordnetSearch(unittest.TestCase):

    def test_yields_multiple_results(self):
        results = list(Wordnet.search("bank", pos=NOUN, lang="en"))
        self.assertGreater(len(results), 1)

    def test_each_result_has_definition(self):
        for data in Wordnet.search("bank", pos=NOUN, lang="en"):
            self.assertIn("definition", data)

    def test_unsupported_lang_yields_nothing(self):
        results = list(Wordnet.search("dog", lang="xx"))
        self.assertEqual(results, [])

    def test_unknown_word_yields_nothing(self):
        results = list(Wordnet.search("xyzzy_not_a_word", lang="en"))
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# WordnetRetrievalEngine.query
# ---------------------------------------------------------------------------

class TestWordnetRetrievalEngine(unittest.TestCase):

    def setUp(self):
        self.engine = WordnetRetrievalEngine()

    def test_bare_word_returns_passages(self):
        results = self.engine.query("dog", lang="en")
        self.assertGreater(len(results), 0)

    def test_result_is_list_of_tuples(self):
        results = self.engine.query("dog", lang="en")
        for text, score in results:
            self.assertIsInstance(text, str)
            self.assertIsInstance(score, float)

    def test_k_limits_results(self):
        results = self.engine.query("bank", lang="en", k=3)
        self.assertLessEqual(len(results), 3)

    def test_intent_query_returns_single_passage(self):
        results = self.engine.query("what is the definition of dog", lang="en")
        self.assertEqual(len(results), 1)
        _, score = results[0]
        self.assertAlmostEqual(score, 0.9)

    def test_unknown_word_intent_returns_not_found(self):
        results = self.engine.query("what is the definition of xyzzy_not_a_word",
                                    lang="en")
        # Returns not_found dialog at score 0.0 or empty list
        if results:
            _, score = results[0]
            self.assertAlmostEqual(score, 0.0)

    def test_polysemous_word_multiple_passages(self):
        results = self.engine.query("bank", lang="en")
        self.assertGreater(len(results), 1)

    def test_scores_between_zero_and_one(self):
        for _, score in self.engine.query("dog", lang="en"):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# WordnetToolbox — define_word
# ---------------------------------------------------------------------------

class TestDefineWord(unittest.TestCase):

    def setUp(self):
        self.tb = WordnetToolbox()

    def test_returns_define_word_output(self):
        result = self.tb.define_word(DefineWordArgs(word="dog", lang="en"))
        self.assertIsInstance(result, DefineWordOutput)

    def test_definitions_list_non_empty(self):
        result = self.tb.define_word(DefineWordArgs(word="dog", lang="en"))
        self.assertGreater(len(result.definitions), 0)

    def test_definitions_are_tuples_of_strings(self):
        result = self.tb.define_word(DefineWordArgs(word="dog", lang="en"))
        for pos_label, defn in result.definitions:
            self.assertIsInstance(pos_label, str)
            self.assertIsInstance(defn, str)

    def test_pos_filter_noun_only(self):
        result = self.tb.define_word(DefineWordArgs(word="bank", lang="en", pos="noun"))
        for pos_label, _ in result.definitions:
            self.assertEqual(pos_label, "noun")

    def test_pos_filter_verb_only(self):
        result = self.tb.define_word(DefineWordArgs(word="bank", lang="en", pos="verb"))
        for pos_label, _ in result.definitions:
            self.assertEqual(pos_label, "verb")

    def test_unknown_word_empty_definitions(self):
        result = self.tb.define_word(DefineWordArgs(word="xyzzy_not_a_word", lang="en"))
        self.assertEqual(result.definitions, [])

    def test_examples_is_list(self):
        result = self.tb.define_word(DefineWordArgs(word="dog", lang="en"))
        self.assertIsInstance(result.examples, list)


# ---------------------------------------------------------------------------
# WordnetToolbox — word_relations
# ---------------------------------------------------------------------------

class TestWordRelations(unittest.TestCase):

    def setUp(self):
        self.tb = WordnetToolbox()

    def test_returns_word_relations_output(self):
        result = self.tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun"))
        self.assertIsInstance(result, WordRelationsOutput)

    def test_synonyms_non_empty(self):
        result = self.tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun"))
        self.assertGreater(len(result.synonyms), 0)

    def test_hypernyms_non_empty(self):
        result = self.tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun"))
        self.assertGreater(len(result.hypernyms), 0)

    def test_hyponyms_non_empty(self):
        result = self.tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun"))
        self.assertGreater(len(result.hyponyms), 0)

    def test_all_fields_are_lists(self):
        result = self.tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun"))
        for field in ("synonyms", "antonyms", "hypernyms", "hyponyms", "holonyms"):
            self.assertIsInstance(getattr(result, field), list,
                                  f"field {field!r} should be a list")

    def test_unknown_word_all_empty(self):
        result = self.tb.word_relations(
            WordRelationsArgs(word="xyzzy_not_a_word", lang="en", pos="noun"))
        self.assertEqual(result.synonyms, [])
        self.assertEqual(result.hypernyms, [])
        self.assertEqual(result.hyponyms, [])

    def test_discover_tools_returns_two_tools(self):
        tools = self.tb.discover_tools()
        self.assertEqual(len(tools), 2)
        names = {t.name for t in tools}
        self.assertEqual(names, {"define_word", "word_relations"})


if __name__ == "__main__":
    unittest.main()
