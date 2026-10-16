#!/usr/bin/env python3
"""DCP cache-address teaching example: standard library only.

No weights, tensors, network, or accelerator are required. This models index
contracts, not allocator/kernel implementations or device performance. Page IDs
are chosen for arithmetic and are not real allocations. The default is token
interleaving; block-interleave checks use widths dividing the physical page size.
"""

from dataclasses import dataclass
import unittest


@dataclass(frozen=True)
class Address:
    position: int
    virtual_slot: int
    owner: int
    local_row: int
    local_page: int
    local_offset: int


def split_virtual(slot, degree, interleave=1):
    """Nonnegative identifier -> (owner rank, compact local identifier)."""
    if slot < 0:
        raise ValueError("Padding is not a valid cache address")
    if degree < 1 or interleave < 1:
        raise ValueError("Degree and interleave width must be positive")
    block, offset = divmod(slot, interleave)
    return block % degree, (block // degree) * interleave + offset


def join_local(owner, row, degree, interleave=1):
    block, offset = divmod(row, interleave)
    return (block * degree + owner) * interleave + offset


def localize(slot, rank, degree, interleave=1):
    """Preserve invalid/non-owned entries as -1; never use -1 to index a pool."""
    if slot < 0:
        return -1
    owner, row = split_virtual(slot, degree, interleave)
    return row if owner == rank else -1


def locate(position, page_ids, degree=2, physical_page_size=4, interleave=1):
    if position < 0 or physical_page_size < 1:
        raise ValueError("Position must be nonnegative and page size positive")
    if interleave < 1 or physical_page_size % interleave:
        raise ValueError("This paged example requires interleave to divide page size")
    virtual_page_size = degree * physical_page_size
    logical_page, virtual_offset = divmod(position, virtual_page_size)
    page_id = page_ids[logical_page]
    virtual = page_id * virtual_page_size + virtual_offset
    owner, local_row = split_virtual(virtual, degree, interleave)
    local_page, local_offset = divmod(local_row, physical_page_size)
    return Address(position, virtual, owner, local_row, local_page, local_offset)


def write_position(pools, request, position, page_ids):
    address = locate(position, page_ids, degree=len(pools))
    # Store a label instead of a high-dimensional KV entry.
    assert address.local_row not in pools[address.owner]
    pools[address.owner][address.local_row] = (request, position)
    return address


def read_visible(pools, request, query_position, page_ids):
    """Build per-rank read lists; only this request's causal positions are valid."""
    reads = [[] for _ in pools]
    for position in range(query_position + 1):
        address = locate(position, page_ids, degree=len(pools))
        value = pools[address.owner][address.local_row]
        assert value == (request, position)
        reads[address.owner].append((position, address.local_row))
    return reads


def print_walkthrough():
    pages = [2, 7]
    print("DCP=2; physical page size=4; virtual page size=8")
    print("Request R0 logical-page -> virtual-page IDs:", pages)
    print("position  virtual  owner  local_row  local_page  offset")
    for p in range(10):
        a = locate(p, pages)
        print(f"{p:8d} {a.virtual_slot:8d} {a.owner:6d} "
              f"{a.local_row:10d} {a.local_page:11d} {a.local_offset:7d}")
    pools = [{}, {}]
    for p in range(6):
        write_position(pools, "R0", p, pages)
    a = write_position(pools, "R0", 6, pages)
    print("Decode position 6 write:", a)
    for rank, reads in enumerate(read_visible(pools, "R0", 6, pages)):
        print(f"Rank {rank} reads (original position, physical row): {reads}")
    print("Same row ID, different devices:", (0, 8), (1, 8))
    print("Position 8 crosses to page 7:", locate(8, pages))
    print("R1 position 0 starts on another page:", locate(0, [4, 9]))


class AddressChecks(unittest.TestCase):
    def test_documented_ten_token_table(self):
        addresses = [locate(p, [2, 7]) for p in range(10)]
        self.assertEqual([a.virtual_slot for a in addresses],
                         list(range(16, 24)) + [56, 57])
        self.assertEqual([a.owner for a in addresses], [0, 1] * 5)
        self.assertEqual([a.local_row for a in addresses],
                         [8, 8, 9, 9, 10, 10, 11, 11, 28, 28])

    def test_same_row_different_rank_is_not_aliasing(self):
        first, second = locate(0, [2]), locate(1, [2])
        self.assertEqual(first.local_row, second.local_row)
        self.assertNotEqual((first.owner, first.local_row),
                            (second.owner, second.local_row))

    def test_noncontiguous_page_boundary(self):
        self.assertEqual(locate(7, [2, 7]).local_row, 11)
        next_page = locate(8, [2, 7])
        self.assertEqual((next_page.virtual_slot, next_page.local_page,
                          next_page.local_offset, next_page.local_row),
                         (56, 7, 0, 28))

    def test_separate_requests_do_not_collide(self):
        pools = [{}, {}]
        for req, pages in [("R0", [2, 7]), ("R1", [4, 9])]:
            for p in range(10):
                write_position(pools, req, p, pages)
        self.assertEqual(sum(map(len, pools)), 20)
        self.assertEqual(pools[0][8], ("R0", 0))
        self.assertEqual(pools[0][16], ("R1", 0))

    def test_owner_only_write_and_causal_read(self):
        pools = [{}, {}]
        for p in range(10):
            write_position(pools, "R0", p, [2, 7])
        reads = read_visible(pools, "R0", 6, [2, 7])
        self.assertEqual(reads[0], [(0, 8), (2, 9), (4, 10), (6, 11)])
        self.assertEqual(reads[1], [(1, 8), (3, 9), (5, 10)])
        # Future KV is present in this toy pool, but not visible to position 6.
        self.assertTrue(all(p <= 6 for rank in reads for p, _ in rank))
        self.assertEqual(sum(len(rank) for rank in reads), 7)

    def test_local_ordinal_is_not_pool_row(self):
        a = locate(6, [2, 7])
        local_history_ordinal = 6 // 2
        self.assertEqual(local_history_ordinal, 3)
        self.assertEqual(a.local_row, 11)

    def test_padding_and_nonowned_entries(self):
        self.assertEqual(localize(-1, 0, 2), -1)
        self.assertEqual(localize(-7, 1, 2), -1)
        self.assertEqual(localize(17, 0, 2), -1)
        self.assertEqual(localize(17, 1, 2), 8)
        with self.assertRaises(ValueError):
            split_virtual(-1, 2)

    def test_block_interleaving_example(self):
        positions = [[], []]
        for p in range(8):
            owner, _ = split_virtual(p, 2, interleave=2)
            positions[owner].append(p)
        self.assertEqual(positions, [[0, 1, 4, 5], [2, 3, 6, 7]])

    def test_inverse_and_page_invariants(self):
        for degree in (1, 2, 3, 4):
            for page_size in (1, 2, 4, 8):
                for interleave in (i for i in range(1, page_size + 1)
                                   if page_size % i == 0):
                    seen = set()
                    length = 2 * degree * page_size
                    for p in range(length):
                        a = locate(p, [2, 7], degree, page_size, interleave)
                        key = (a.owner, a.local_row)
                        self.assertNotIn(key, seen)
                        seen.add(key)
                        self.assertEqual(join_local(a.owner, a.local_row, degree,
                                                    interleave), a.virtual_slot)
                        expected_page = [2, 7][p // (degree * page_size)]
                        self.assertEqual(a.local_page, expected_page)
                        self.assertLess(a.local_offset, page_size)
                        self.assertEqual(localize(a.virtual_slot, a.owner, degree,
                                                  interleave), a.local_row)

    def test_reserved_capacity_is_not_valid_length(self):
        addresses = [locate(p, [2, 7]) for p in range(10)]
        for rank in (0, 1):
            local = [a for a in addresses if a.owner == rank]
            pages = {a.local_page for a in local}
            self.assertEqual(len(pages) * 4, 8)  # Reserved rows.
            self.assertEqual(len(local), 5)     # Valid rows.


if __name__ == "__main__":
    print_walkthrough()
    print("CPU address/index invariants only; no backend/performance claim.", flush=True)
    unittest.main(verbosity=2)
