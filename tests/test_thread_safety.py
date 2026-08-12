"""
Regression tests for the sqlite thread-safety bug.

Symptom (live on ser9): common_qa fans a query out to every registered
solver on its own worker thread. ``wn`` pools a single sqlite3 connection
per database path bound to whichever thread first opened it, so any other
thread touching it raised::

    sqlite3.ProgrammingError: SQLite objects created in a thread can only
    be used in that same thread

wordnet silently lost every common_qa round as a result. These tests drive
:class:`WordnetRetrievalEngine` concurrently from many threads and assert
no thread ever raises - and that the lock the fix relies on was actually
exercised, not just that ``allow_multithreading`` happened to paper over
the exception on this platform's sqlite build.
"""
import subprocess
import sys
import threading
import unittest

import ovos_wordnet_plugin as _plugin_module
from ovos_wordnet_plugin import WordnetRetrievalEngine, _ensure_downloaded

N_THREADS = 12


class TestConcurrentQueries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Force the English lexicon to be present (and its connection
        # opened) on *this* thread before spinning up worker threads, so
        # every worker is guaranteed to hit the pooled connection from a
        # thread that did not create it.
        assert _ensure_downloaded("oewn:2024"), "oewn:2024 must be downloadable"

    def setUp(self):
        self.engine = WordnetRetrievalEngine()

    def _run_concurrently(self, target, words):
        errors = []
        threads = []

        def _wrapped(word):
            try:
                target(word)
            except BaseException as exc:  # noqa: BLE001 - capture everything
                errors.append((word, exc))

        for word in words:
            t = threading.Thread(target=_wrapped, args=(word,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # A deadlock (e.g. a lock held past its critical section) must fail
        # the test loudly, not pass vacuously because we never checked
        # whether every thread actually finished within the timeout.
        stuck = [t.name for t in threads if t.is_alive()]
        self.assertEqual(
            stuck, [],
            f"{len(stuck)}/{len(threads)} threads did not complete within the "
            f"join timeout - looks like a deadlock, not a clean run"
        )

        return errors

    def test_concurrent_query_calls_do_not_raise(self):
        words = ["dog", "cat", "bank", "run", "happy", "tree",
                 "car", "book", "water", "light", "sound", "mountain"][:N_THREADS]

        before = _plugin_module._wn_lock_acquisitions
        errors = self._run_concurrently(
            lambda w: self.engine.query(w, lang="en", k=1), words
        )
        after = _plugin_module._wn_lock_acquisitions

        if errors:
            details = "\n".join(f"  {w!r}: {e!r}" for w, e in errors)
            self.fail(
                f"{len(errors)}/{len(words)} threaded WordnetRetrievalEngine.query() "
                f"calls raised (thread-safety regression):\n{details}"
            )

        # Gate the lock itself, not just the "no exception" outcome: on a
        # build with the RLock deleted entirely, allow_multithreading alone
        # was enough to avoid the exception on this platform (verified by
        # the reviewer's ablation, 10/10 green with the lock removed), which
        # let a lock-free regression pass silently. Asserting the counter
        # moved means removing the lock now fails this test directly.
        self.assertGreaterEqual(
            after - before, len(words),
            "WordnetRetrievalEngine.query() completed without visibly "
            "acquiring _wn_lock - the lock may have been removed/bypassed"
        )

    def test_concurrent_get_definition_calls_do_not_raise(self):
        words = ["dog", "cat", "bank", "run", "happy", "tree",
                 "car", "book", "water", "light", "sound", "mountain"][:N_THREADS]

        before = _plugin_module._wn_lock_acquisitions
        errors = self._run_concurrently(
            lambda w: self.engine.get_definition(w, lang="en"), words
        )
        after = _plugin_module._wn_lock_acquisitions

        if errors:
            details = "\n".join(f"  {w!r}: {e!r}" for w, e in errors)
            self.fail(
                f"{len(errors)}/{len(words)} threaded WordnetRetrievalEngine."
                f"get_definition() calls raised (thread-safety regression):\n{details}"
            )

        self.assertGreaterEqual(
            after - before, len(words),
            "WordnetRetrievalEngine.get_definition() completed without visibly "
            "acquiring _wn_lock - the lock may have been removed/bypassed"
        )

    def test_concurrent_queries_return_real_results(self):
        # Not just "no exception" - the engine must actually still answer
        # under concurrency, not silently swallow every round.
        results = {}
        lock = threading.Lock()

        def _lookup(word):
            passages = self.engine.query(word, lang="en", k=1)
            with lock:
                results[word] = passages

        words = ["dog", "cat", "bank", "run", "happy"]
        threads = [threading.Thread(target=_lookup, args=(w,)) for w in words]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        stuck = [t.name for t in threads if t.is_alive()]
        self.assertEqual(stuck, [],
                         f"{len(stuck)}/{len(threads)} threads did not complete - deadlock")

        # Every word we launched must have actually produced a result -
        # a thread silently missing from `results` (e.g. because it hung
        # and never wrote back) must fail this test, not be skipped.
        self.assertEqual(set(results), set(words),
                         f"missing results for: {set(words) - set(results)}")
        for word, passages in results.items():
            self.assertTrue(passages, f"no passages returned for {word!r} under concurrency")


class TestSearchGeneratorDoesNotDeadlock(unittest.TestCase):
    """Regression for the ``Wordnet.search()`` generator deadlock.

    A generator decorated so it holds ``_wn_lock`` for the lifetime of
    iteration wedges every other thread the moment the caller only
    partially drains it (starts iterating, then stops without exhausting
    it) - because the lock is only released when the generator is closed
    or garbage collected, not when the caller loses interest. Reproduced
    via the reviewer's ``gentest.py`` TEST B ablation against the
    generator-locking version of this fix.
    """

    @classmethod
    def setUpClass(cls):
        assert _ensure_downloaded("oewn:2024"), "oewn:2024 must be downloadable"

    def test_partially_consumed_search_does_not_block_other_threads(self):
        from ovos_wordnet_plugin import Wordnet

        gen = Wordnet.search("bank", lang="en")
        next(gen)  # start iterating, then abandon it without exhausting it

        done = []

        def _other_thread_query():
            done.append(Wordnet.get_definition("cat", lang="en"))

        t = threading.Thread(target=_other_thread_query, daemon=True)
        t.start()
        t.join(timeout=10)

        self.assertFalse(
            t.is_alive(),
            "a partially-consumed Wordnet.search() generator blocked another "
            "thread's WordNet lookup - the lock is being held across yields"
        )
        self.assertTrue(done, "other thread's get_definition() never completed")
        self.assertIsNotNone(done[0])


class TestImportOrderDoesNotReintroduceTheBug(unittest.TestCase):
    """Regression for the import-order hole in the flag/pool fix.

    ``wn._db.connect()`` only reads ``allow_multithreading`` when it opens a
    *new* pooled connection. The pool is keyed by database path and never
    rebuilt on its own, so if some other in-process consumer of ``wn``
    opened a connection *before* this plugin's module body ran (and set the
    flag), that stale same-thread-checked connection stays in the pool and
    the original ``ProgrammingError`` comes back despite the fix being
    installed. This drives that exact ordering in a subprocess, where import
    order can be controlled precisely, and hammers the plugin from many
    threads afterwards.
    """

    def test_touching_wn_before_the_plugin_import_still_works_under_threads(self):
        script = """
import sys, threading

# Simulate another in-process consumer of `wn` (e.g. a different plugin,
# or a REPL) opening the pooled sqlite connection BEFORE this plugin's
# module body (and its allow_multithreading + clear_connections() call)
# ever runs.
import wn as _wn_before
_wn_before.lexicons()  # forces wn._db.connect() to open+pool a connection

# Only now import the plugin under test.
import ovos_wordnet_plugin as plugin

assert plugin._ensure_downloaded("oewn:2024"), "oewn:2024 must be downloadable"
engine = plugin.WordnetRetrievalEngine()

errors = []
def worker(word):
    try:
        engine.query(word, lang="en", k=1)
    except BaseException as exc:
        errors.append(repr(exc))

words = ["dog", "cat", "bank", "run", "happy", "tree",
         "car", "book", "water", "light", "sound", "mountain"]
threads = [threading.Thread(target=worker, args=(w,)) for w in words]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=60)

stuck = [t.name for t in threads if t.is_alive()]
if stuck:
    print(f"DEADLOCK: {len(stuck)} threads stuck")
    sys.exit(2)
if errors:
    print(f"ERRORS: {errors}")
    sys.exit(1)
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"import-order regression reproduced:\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
