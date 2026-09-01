import pytest
from llm_server.kv_cache import OutOfPages, PagedKVCache


def test_page_allocation_and_release():
    cache = PagedKVCache(4, 16)
    pages = cache.allocate("r1", 3)
    assert cache.free_pages == 1
    cache.release("r1", pages)
    assert cache.free_pages == 4


def test_out_of_pages():
    cache = PagedKVCache(1, 16)
    with pytest.raises(OutOfPages):
        cache.allocate("r1", 2)


def test_longest_prefix():
    cache = PagedKVCache(8, 4)
    cache.remember_prefix([1, 2], [0])
    cache.remember_prefix([1, 2, 3], [1])
    assert cache.longest_prefix([1, 2, 3, 4]).token_ids == (1, 2, 3)

