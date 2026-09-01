import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass


class OutOfPages(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    token_ids: tuple[int, ...]
    page_ids: tuple[int, ...]


class PagedKVCache:
    """Logical page allocator mirroring GPU KV blocks without copying tensors."""

    def __init__(self, total_pages: int, page_size: int, prefix_capacity: int = 256):
        self.page_size = page_size
        self._free = list(range(total_pages - 1, -1, -1))
        self._owners: dict[int, str] = {}
        self._prefixes: OrderedDict[str, PrefixEntry] = OrderedDict()
        self._prefix_capacity = prefix_capacity

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def pages_needed(self, tokens: int) -> int:
        return math.ceil(tokens / self.page_size)

    def allocate(self, owner: str, count: int) -> list[int]:
        if count > len(self._free):
            raise OutOfPages(f"requested {count} pages; {len(self._free)} available")
        pages = [self._free.pop() for _ in range(count)]
        self._owners.update((page, owner) for page in pages)
        return pages

    def ensure_capacity(self, owner: str, pages: list[int], token_count: int) -> None:
        missing = self.pages_needed(token_count) - len(pages)
        if missing > 0:
            pages.extend(self.allocate(owner, missing))

    def release(self, owner: str, pages: list[int]) -> None:
        for page in list(pages):
            if self._owners.get(page) == owner:
                del self._owners[page]
                self._free.append(page)
        pages.clear()

    @staticmethod
    def prefix_key(tokens: list[int]) -> str:
        return hashlib.sha256(bytes(str(tokens), "utf-8")).hexdigest()

    def remember_prefix(self, tokens: list[int], pages: list[int]) -> None:
        key = self.prefix_key(tokens)
        self._prefixes[key] = PrefixEntry(tuple(tokens), tuple(pages))
        self._prefixes.move_to_end(key)
        while len(self._prefixes) > self._prefix_capacity:
            self._prefixes.popitem(last=False)

    def longest_prefix(self, tokens: list[int]) -> PrefixEntry | None:
        best = None
        for entry in self._prefixes.values():
            if (
                len(entry.token_ids) <= len(tokens)
                and tuple(tokens[: len(entry.token_ids)]) == entry.token_ids
                and (best is None or len(entry.token_ids) > len(best.token_ids))
            ):
                best = entry
        return best
