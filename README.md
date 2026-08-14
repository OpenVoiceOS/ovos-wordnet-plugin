# ovos-wordnet-plugin

This OVOS plugin exposes [WordNet](https://wordnet.princeton.edu/) as a
`RetrievalEngine` and as a `ToolBox` with two agent tools.

It supports 30+ languages through the [`wn`](https://pypi.org/project/wn/)
package, using [Open English WordNet 2024](https://github.com/globalwordnet/english-wordnet)
for English, [ODENet](https://github.com/hdaSprachtechnologie/odenet) for
German, and [OMW 1.4](https://omwn.org/) packs for other languages. The
plugin downloads lexicons on first use and caches them locally.

Most non-English and non-German OMW packs have no native definitions. For
these languages, the engine fetches the English OEWN definition and
translates it into the target language. It uses the OVOS translation plugin
configured on the system, `ovos-translate-plugin-server` by default.

---

## Installation

```bash
pip install ovos-wordnet-plugin
```

The first query downloads the required WordNet lexicon (about 30-60 MB for English).
Subsequent calls are served from the local cache with no network access.

---

## Components

### `WordnetRetrievalEngine`

An OVOS [`RetrievalEngine`](https://github.com/OpenVoiceOS/ovos-plugin-manager)
that returns natural-language passages for a word lookup.  Each passage covers
one WordNet sense and is formatted as:

```
dog (noun): a member of the genus Canis. Also known as: domestic dog, Canis familiaris. Example: the dog barked all night
```

When the query matches a locale intent pattern (e.g. *"what are the antonyms
of happy?"*) the engine resolves that specific relation and returns a single
rendered dialog response instead.

Passages are drawn from all four parts of speech (noun → verb → adjective →
adverb) up to the requested `k` limit.

```python
from ovos_wordnet_plugin import WordnetRetrievalEngine

engine = WordnetRetrievalEngine()
for passage, score in engine.query("bank", lang="en", k=3):
    print(f"[{score:.2f}] {passage}")
```

```
[0.70] bank (noun): sloping land (especially the slope beside a body of water). Example: he sat on the bank of the river
[0.70] bank (noun): a financial institution that accepts deposits. Also known as: depository financial institution.
[0.70] bank (verb): tip laterally. Example: the pilot had to bank the aircraft
```

Intent-style queries return a single spoken-language response:

```python
for passage, score in engine.query("what are the hyponyms of dog", lang="en"):
    print(f"[{score:.2f}] {passage}")
# [0.90] The hyponyms of dog are: poodle, corgi, dalmatian, ...
```

### `WordnetToolbox`

An OVOS `ToolBox` that registers two agent tools.

#### `define_word`

Returns definitions grouped by part of speech, plus usage examples.

| Argument | Type | Default | Description |
|---|---|---|---|
| `word` | str | n/a | Word to define |
| `lang` | str | `"en"` | BCP-47 language code |
| `pos` | str | `"any"` | `"noun"`, `"verb"`, `"adjective"`, `"adverb"`, or `"any"` |

```python
from ovos_wordnet_plugin import WordnetToolbox, DefineWordArgs

tb = WordnetToolbox()
out = tb.define_word(DefineWordArgs(word="run", lang="en", pos="verb"))
for pos_label, definition in out.definitions:
    print(f"{pos_label}: {definition}")
# verb: move fast by using one's feet, with one foot off the ground at any given time
print(out.examples[:2])
# ['she ran her first marathon', ...]
```

#### `word_relations`

Returns synonyms, antonyms, hypernyms, hyponyms, and holonyms.

| Argument | Type | Default | Description |
|---|---|---|---|
| `word` | str | n/a | Word to look up |
| `lang` | str | `"en"` | BCP-47 language code |
| `pos` | str | `"noun"` | `"noun"`, `"verb"`, `"adjective"`, or `"adverb"` |

```python
from ovos_wordnet_plugin import WordnetToolbox, WordRelationsArgs

tb = WordnetToolbox()
out = tb.word_relations(WordRelationsArgs(word="dog", lang="en", pos="noun"))
print("synonyms:", out.synonyms)     # ['dog', 'domestic dog', 'Canis familiaris']
print("hypernyms:", out.hypernyms)   # ['canine', 'canid', ...]
print("hyponyms:", out.hyponyms[:3]) # ['corgi', 'dalmatian', 'poodle']
print("holonyms:", out.holonyms)     # ['pack', ...]
```

### Low-level `Wordnet` helper

The `Wordnet` class provides direct access to all WordNet relations:

```python
from ovos_wordnet_plugin import Wordnet, NOUN, ADJ

# First sense only
data = Wordnet.get("bank", pos=NOUN, lang="en")

# All senses (generator)
for sense in Wordnet.search("bank", pos=NOUN, lang="en"):
    print(sense["definition"])

# Individual lookups
Wordnet.get_definition("bank", lang="en")
Wordnet.get_lemmas("bank", lang="en")
Wordnet.get_hypernyms("dog", lang="en")
Wordnet.get_hyponyms("dog", lang="en")
Wordnet.get_holonyms("dog", lang="en")
Wordnet.get_antonyms("good", pos=ADJ, lang="en")
Wordnet.get_root_hypernyms("dog", lang="en")
Wordnet.common_hypernyms("dog", "cat", lang="en")
```

---

## Supported Languages

English uses OEWN 2024, with native definitions throughout. German uses
ODENet, with native definitions for about 85% of synsets. All other
languages use their OMW 1.4 pack. These packs provide lemmas but typically
no glosses, so the plugin translates definitions from English through the
configured OVOS translation plugin.

| BCP-47 | Language | Source |
|--------|----------|--------|
| `en` | English | Open English WordNet 2024 |
| `de` | German | ODENet 1.4 (native definitions) |
| `ar` | Arabic | OMW 1.4 |
| `bg` | Bulgarian | OMW 1.4 |
| `ca` | Catalan | OMW 1.4 |
| `cmn` | Mandarin Chinese | OMW 1.4 |
| `da` | Danish | OMW 1.4 |
| `el` | Greek | OMW 1.4 |
| `es` | Spanish | OMW 1.4 |
| `eu` | Basque | OMW 1.4 |
| `fi` | Finnish | OMW 1.4 |
| `fr` | French | OMW 1.4 |
| `gl` | Galician | OMW 1.4 |
| `he` | Hebrew | OMW 1.4 |
| `hr` | Croatian | OMW 1.4 |
| `id` | Indonesian | OMW 1.4 |
| `is` | Icelandic | OMW 1.4 |
| `it` | Italian | OMW 1.4 |
| `ja` | Japanese | OMW 1.4 |
| `lt` | Lithuanian | OMW 1.4 |
| `nb` | Norwegian Bokmål | OMW 1.4 |
| `nl` | Dutch | OMW 1.4 |
| `nn` | Norwegian Nynorsk | OMW 1.4 |
| `pl` | Polish | OMW 1.4 |
| `pt` | Portuguese | OMW 1.4 |
| `ro` | Romanian | OMW 1.4 |
| `sk` | Slovak | OMW 1.4 |
| `sl` | Slovenian | OMW 1.4 |
| `sv` | Swedish | OMW 1.4 |
| `th` | Thai | OMW 1.4 |
| `zsm` | Standard Malay | OMW 1.4 |

---

## Running tests

```bash
pip install pytest
pytest tests/
```

---

## Related projects

- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) defines the `RetrievalEngine` interface this plugin implements.
- [OpenVoiceOS/ovos-translate-plugin-server](https://github.com/OpenVoiceOS/ovos-translate-plugin-server) is the default translation plugin used for non-English definitions.

## License

Apache 2.0. See [LICENSE](LICENSE).
