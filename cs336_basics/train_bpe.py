import regex as re
import heapq
from collections import defaultdict
import cProfile
import pstats
from concurrent.futures import ProcessPoolExecutor


class Node:
    __slots__ = ("value", "prev", "next", "active")

    def __init__(self, value: bytes):
        self.value = value
        self.prev = None
        self.next = None
        self.active = True


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value: bytes) -> Node:
        new_node = Node(value)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        return new_node

    def from_list(self, values):
        for value in values:
            self.append(value)


def pretokenization(
    input_path: str,
    special_tokens: list[str],
) -> tuple[dict[str, int], dict[str, DoublyLinkedList]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    special_set = set(special_tokens)

    token_count: dict[str, int] = {}
    token_structure: dict[str, DoublyLinkedList] = {}

    def add_pretoken(pretoken: str) -> None:
        token_count[pretoken] = token_count.get(pretoken, 0) + 1
        if pretoken not in token_structure:
            b = pretoken.encode('utf-8')
            dll = DoublyLinkedList()
            dll.from_list([bytes([x]) for x in b])
            token_structure[pretoken] = dll

    if special_tokens:
        split_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        parts = re.split(split_pattern, text)
    else:
        parts = [text]
    pat_re = re.compile(PAT)
    for part in parts:
        if not part:
            continue

        # Keep special tokens intact by skipping them in merge training.
        # They are already added to vocab separately.
        if part in special_set:
            continue

        for m in pat_re.finditer(part):
            add_pretoken(m.group(0))

    return token_count, token_structure

def chunk_pretokenization(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[str, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    special_set = set(special_tokens)

    token_count: dict[str, int] = {}

    if special_tokens:
        split_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        parts = re.split(split_pattern, text)
    else:
        parts = [text]

    pat_re = re.compile(PAT)

    for part in parts:
        if not part or part in special_set:
            continue

        for m in pat_re.finditer(part):
            pretoken = m.group(0)
            token_count[pretoken] = token_count.get(pretoken, 0) + 1

    return token_count
def build_token_structure(token_count: dict[str, int]) -> dict[str, DoublyLinkedList]:
    token_structure = {}

    for pretoken in token_count:
        b = pretoken.encode("utf-8")
        dll = DoublyLinkedList()
        dll.from_list([bytes([x]) for x in b])
        token_structure[pretoken] = dll

    return token_structure
def parallel_pretokenization(
    input_path: str,
    special_tokens: list[str],
    max_process: int,
) -> tuple[dict[str, int], dict[str, DoublyLinkedList]]:
    from cs336_basics.pretokenization_example import find_chunk_boundaries

    with open(input_path, "rb") as f:
        boundaries = list(
            find_chunk_boundaries(
                f,
                max_process,
                special_tokens[0].encode("utf-8"),
            )
        )

    starts = boundaries[:-1]
    ends = boundaries[1:]

    token_count: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=max_process) as executor:
        results = executor.map(
            chunk_pretokenization,
            [input_path] * len(starts),
            starts,
            ends,
            [special_tokens] * len(starts),
        )

        for partial_counts in results:
            for token, count in partial_counts.items():
                token_count[token] = token_count.get(token, 0) + count

    token_structure = build_token_structure(token_count)
    return token_count, token_structure
def initialize_pair_stats(
    token_count: dict[str, int],
    token_structure: dict[str, DoublyLinkedList],
) -> tuple[
    defaultdict[tuple[bytes, bytes], int],
    defaultdict[tuple[bytes, bytes], list[tuple[str, Node]]],
]:
    pair_count = defaultdict(int)
    pair_positions = defaultdict(list)

    for token, freq in token_count.items():
        cur = token_structure[token].head
        while cur is not None and cur.next is not None:
            pair = (cur.value, cur.next.value)
            pair_count[pair] += freq
            pair_positions[pair].append((token, cur))  # start node of the pair
            cur = cur.next

    return pair_count, pair_positions


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    max_process: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert len(special_tokens) + 256 <= vocab_size

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    merges: list[tuple[bytes, bytes]] = []

    token_count, token_structure = parallel_pretokenization(input_path, special_tokens, max_process)
    pair_count, pair_positions = initialize_pair_stats(token_count, token_structure)

    heap: list[tuple[int, tuple[bytes, bytes]]] = []

    def push(pair: tuple[bytes, bytes]) -> None:
        count = pair_count[pair]
        heapq.heappush(heap, (-count, pair))

    for pair in pair_count:
        if pair_count[pair] > 0:
            push(pair)

    while len(vocab) < vocab_size - len(special_tokens) and heap:
        def extract_best_pair():
            neg_count, best_pair = heapq.heappop(heap)

            best_count = -neg_count
            return best_count, best_pair
        best_pairs = []
        cur_count = best_count = 0
        pair_candidate=(b' ', b' ')
        while heap:
            cur_count, pair_candidate = extract_best_pair()
            if pair_count[pair_candidate] != cur_count:
                continue
            if cur_count <= 0:
                break
            if best_count == 0:
                best_count = cur_count
            if best_count == cur_count:
                best_pairs.append(pair_candidate)
            else:
                break
        if best_count == 0:
            break
        push(pair_candidate)
        best_pair = best_pairs[-1]
        for i in range(len(best_pairs)-1):
            push(best_pairs[i])

        merged_bytes = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_bytes
        merges.append(best_pair)

        changed_pairs: set[tuple[bytes, bytes]] = set()

        # Snapshot current occurrences of best_pair.
        # Some will become stale while we process them; we validate before merging.
        occurrences = list(pair_positions[best_pair])

        # After this round, all surviving occurrences of best_pair should be gone.
        pair_count[best_pair] = 0
        pair_positions[best_pair] = []

        for token, left in occurrences:
            right = left.next

            if not left.active:
                continue
            if right is None or not right.active:
                continue
            if (left.value, right.value) != best_pair:
                continue

            freq = token_count[token]
            prev_node = left.prev
            next_node = right.next

            # Remove old neighboring pairs that disappear
            if prev_node is not None and prev_node.active:
                old_left_pair = (prev_node.value, left.value)
                pair_count[old_left_pair] -= freq
                changed_pairs.add(old_left_pair)

            if next_node is not None and next_node.active:
                old_right_pair = (right.value, next_node.value)
                pair_count[old_right_pair] -= freq
                changed_pairs.add(old_right_pair)

            # Create merged node
            merged_node = Node(merged_bytes)

            if prev_node is not None and prev_node.active:
                prev_node.next = merged_node
                merged_node.prev = prev_node
            else:
                token_structure[token].head = merged_node

            if next_node is not None and next_node.active:
                merged_node.next = next_node
                next_node.prev = merged_node
            else:
                token_structure[token].tail = merged_node

            # Invalidate consumed nodes so old saved pointers can't be reused
            left.active = False
            right.active = False
            left.prev = left.next = None
            right.prev = right.next = None

            # Add new neighboring pairs created by the merge
            if merged_node.prev is not None:
                new_left_pair = (merged_node.prev.value, merged_node.value)
                pair_count[new_left_pair] += freq
                pair_positions[new_left_pair].append((token, merged_node.prev))
                changed_pairs.add(new_left_pair)

            if merged_node.next is not None:
                new_right_pair = (merged_node.value, merged_node.next.value)
                pair_count[new_right_pair] += freq
                pair_positions[new_right_pair].append((token, merged_node))
                changed_pairs.add(new_right_pair)

        for pair in changed_pairs:
            if pair_count[pair] > 0:
                push(pair)

    # add special tokens
    vocab_cur_size = len(vocab)
    for i, t in enumerate(special_tokens):
        vocab[i+vocab_cur_size] = t.encode('utf-8')
    return vocab, merges


def main():
    vocab, merges = train_bpe(
        "D:/TomwangWork/Notes/Extracurricular Learning/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        8
    )
    longest_token = max(vocab.values(), key=len)
    print(f"longest token (bytes): {longest_token}")
    print(f"length: {len(longest_token)}")
    print(f"decode: {longest_token.decode('utf-8', errors='replace')}")
    import pickle
    with open("D:/TomwangWork/Notes/Extracurricular Learning/cs336/assignment1-basics/output/TinyStorie-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("D:/TomwangWork/Notes/Extracurricular Learning/cs336/assignment1-basics/output/TinyStories-merges.pkl", "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(25)
