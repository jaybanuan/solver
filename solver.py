import copy

MAX = 4

def push(entry, value):
    entry.append(value)


def pop(entry):
    return entry.pop()


def peek(entry):
    return entry[-1]


def is_empty(entry):
    return entry == []


def is_full(entry):
    return len(entry) >= MAX


def is_entry_completed(entry):
    result = is_full(entry)

    if result:
        sample_value = peek(entry)
        for value in entry:
            if value != sample_value:
                result = False
                break

    return result


def is_movable(src_entry, dst_entry):
    result = False

    if not is_empty(src_entry) and not is_entry_completed(src_entry) and not is_full(dst_entry):
        if is_empty(dst_entry) or (peek(src_entry) == peek(dst_entry)):
            result = True

#    print("movable: {},  src: {},  dst: {}".format(result, src_entry, dst_entry))

    return result


def move(src_entry, dst_entry):
    push(dst_entry, pop(src_entry))


def is_table_completed(table):
    result = True

    for entry in table:
        if not is_empty(entry) and not is_entry_completed(entry):
            result = False
            break
    
    return result


def solve(table):
    current_path = [table]
    shortest_path = None

    def search_shortest_path():
        nonlocal shortest_path

        if not shortest_path or len(current_path) < len(shortest_path):
            current_table = peek(current_path)

            if is_table_completed(current_table):
                shortest_path = copy.deepcopy(current_path)
            else:
                for src_index, _ in enumerate(current_table):
                    for dst_index, _ in enumerate(current_table):
                        if src_index == dst_index:
                            continue

                        if is_movable(current_table[src_index], current_table[dst_index]):
                            next_table = copy.copy(current_table)
                            next_table[src_index] = copy.copy(current_table[src_index])
                            next_table[dst_index] = copy.copy(current_table[dst_index])

                            move(next_table[src_index], next_table[dst_index])
                            
                            if not next_table in current_path:
                                push(current_path, next_table)
                                search_shortest_path()
                                pop(current_path)

    search_shortest_path()

    return shortest_path


if __name__ == '__main__':
    table = [
        ["水", "水", "緑", "青"],
        ["灰", "草", "桃", "紫"],
        ["茶", "赤", "紫", "橙"],
        ["橙", "赤", "桃", "橙"],
        ["青", "ク", "赤", "緑"],
        ["緑", "茶", "緑", "ク"],
        ["君", "赤", "紫", "茶"],
        ["君", "桃", "紫", "君"],
        ["青", "水", "灰", "草"],
        ["草", "灰", "ク", "茶"],
        ["青", "君", "ク", "灰"],
        ["橙", "桃", "水", "草"],
        [],
        [],
    ]

    shortest_path = solve(table)

    for index, table in enumerate(shortest_path):
        print("===== {} =====".format(index))
        for entry in table:
            print(entry)

        print()
