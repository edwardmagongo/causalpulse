import pytest
from cp_llm_eval.store import Store, digest, read_export


def test_a_digest_ignores_key_order_and_changes_with_any_value():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({"a": 1, "b": 2}) != digest({"a": 1, "b": 3})


def test_a_stored_record_is_read_back(tmp_path):
    store = Store(tmp_path)
    store.put("things", "k1", {"x": 1})

    assert store.has("things", "k1")
    assert store.get("things", "k1") == {"x": 1}
    assert not store.has("things", "k2")


def test_a_record_is_never_overwritten(tmp_path):
    store = Store(tmp_path)
    store.put("things", "k1", {"x": 1})

    with pytest.raises(FileExistsError):
        store.put("things", "k1", {"x": 2})
    assert store.get("things", "k1") == {"x": 1}


def test_records_come_back_in_key_order_and_an_empty_kind_is_empty(tmp_path):
    store = Store(tmp_path)
    store.put("things", "b", {"n": 2})
    store.put("things", "a", {"n": 1})

    assert store.records("things") == [{"n": 1}, {"n": 2}]
    assert store.records("nothing") == []


def test_no_partial_file_is_left_behind(tmp_path):
    Store(tmp_path).put("things", "k1", {"x": 1})

    assert [p.name for p in (tmp_path / "things").iterdir()] == ["k1.json"]


def test_an_export_round_trips_and_is_byte_stable(tmp_path):
    store = Store(tmp_path / "store")
    store.put("things", "b", {"n": 2})
    store.put("things", "a", {"n": 1})

    first, second = tmp_path / "1.jsonl.gz", tmp_path / "2.jsonl.gz"
    assert store.export("things", first) == 2
    store.export("things", second)

    assert read_export(first) == [{"n": 1}, {"n": 2}]
    assert first.read_bytes() == second.read_bytes()


def test_a_restored_export_is_stored_again_under_its_own_keys(tmp_path):
    source = Store(tmp_path / "a")
    source.put("things", "k-1", {"key": 1})
    source.put("things", "k-2", {"key": 2})
    bundle = tmp_path / "things.jsonl.gz"
    source.export("things", bundle)
    target = Store(tmp_path / "b")
    target.put("things", "k-1", {"key": 1})

    restored = target.restore("things", bundle, lambda record: f"k-{record['key']}")

    assert restored == 1
    assert target.records("things") == source.records("things")
