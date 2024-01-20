import abc
import copy
import collections.abc
import hashlib
import itertools
import os
import pathlib
import sys

from typing import TypeVar, TypeVarTuple, TypeAlias, Any, Iterator, Protocol


T = TypeVar("T")
Ts = TypeVarTuple("Ts")

Board: TypeAlias = tuple[tuple, ...]


def print_board(board: Board, normalized_board: Board | None = None, message: Any = None) -> None:
    if message:
        print(f'-- {str(message)} -------------------------------------------')

    for index, _ in enumerate(board):
        if normalized_board:
            print(f'{str(board[index]):30}{str(normalized_board[index])}')
        else:
            print(f'{str(board[index])}')


class MoveNode(collections.abc.Sequence):
    @classmethod
    def __normalize(cls, board: Board) -> Board:
        return tuple(sorted(board))
    

    @classmethod
    def __generate_digest(cls, text: str) -> str:
        h = hashlib.sha256()
        h.update(text.encode())

        return h.hexdigest()


    def __init__(self, board: Board) -> None:
        self.__board = board
        self.__normalized = MoveNode.__normalize(board)
        self.__text = str(self.__normalized)
        self.__digest = MoveNode.__generate_digest(self.__text)


    def __len__(self) -> int:
        return self.board.__len__()


    def __iter__(self) -> Iterator[tuple]:
        return self.board.__iter__()


    def __reversed__(self) -> Iterator[tuple]:
        return self.board.__reversed__()
    

    def __contains__(self, board: Board) -> bool:
        return self.board.__contains__(board)


    def __getitem__(self, index: int) -> tuple:
        return self.board.__getitem__(index)


    @property
    def board(self) -> Board:
        return self.__board


    @property
    def normalized(self) -> Board:
        return self.__normalized


    @property
    def text(self) -> str:
        return self.__text


    @property
    def digest(self) -> str:
        return self.__digest


    def print(self, message: Any = None) -> None:
        print_board(self.board, self.normalized, message)
        
        sys.stdout.flush()


class MovePath(collections.abc.Sequence):
    def __init__(self, *boards: *tuple[Board, ...]) -> None:
        self.__nodes = [MoveNode(board) for board in boards]


    def __len__(self) -> int:
        return self.__nodes.__len__()


    def __iter__(self) -> Iterator[MoveNode]:
        return self.__nodes.__iter__()


    def __reversed__(self) -> Iterator[MoveNode]:
        return self.__nodes.__reversed__()
    

    def __contains__(self, board: Board) -> bool:
        return self.__nodes.__contains__(board)


    def __getitem__(self, index: int) -> MoveNode:
        return self.__nodes.__getitem__(index)


    @property
    def path(self) -> list[MoveNode]:
        return self.__nodes


    def append(self, node: MoveNode) -> None:
        self.__nodes.append(node)


    def pop(self) -> MoveNode:
        return self.__nodes.pop()


    def peek(self) -> MoveNode:
        if self.__nodes:
            return self.__nodes[-1]
        else:
            raise IndexError('no element to peek.')


    def print(self, message: str | None = None, target: Board | None = None):
        if message:
            print(f'========== {message} ===============')
        else:
            print('=====================================')

        if target:
            MoveNode(target).print('target')

        for index, node in enumerate(self.__nodes):
            node.print(index)
        
        sys.stdout.flush()


class History(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add(self, node: MoveNode, depth: int) -> bool: ...


class OnMemoryHistory(History):
    def __init__(self) -> None:
        self.__histroy: dict[Board, int] = {}


    def add(self, node: MoveNode, depth: int) -> bool:
        updated = False

        previous_depth = self.__histroy.get(node.normalized)
        if (previous_depth is None) or (depth < previous_depth):
            self.__histroy[node.normalized] = depth
            updated = True
        
        return updated


class OnStorageHistory(History):
    def __init__(self, output_dir: str | os.PathLike) -> None:
        self.__output_dir = output_dir
    

    def add(self, node: MoveNode, depth: int) -> bool:
        digest_file_path = pathlib.Path(self.__output_dir, node.digest[:4], node.digest[4:8], f'{node.digest}.txt')

        if not digest_file_path.exists():
            digest_file_path.parent.mkdir(parents=True, exist_ok=True)

        updated = False
        with open(digest_file_path, 'a+') as f:
            f.seek(0, os.SEEK_SET)

            entries: dict[str, int] = {}

            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue

                splitted = line.split('\t', 1)
                key = splitted[0]
                value = int(splitted[1])

                if (node.text == key) and (depth < value):
                    entries[node.text] = depth
                    updated = True
                    break
                else:
                    entries[key] = value
        
            if not entries:
                entries[node.text] = depth
                updated = True

            if updated:
                f.seek(0, os.SEEK_SET)
                f.truncate(0)
                for key, value in entries.items():
                    f.write(f'{node.text}\t{depth}\n')

        return updated


def create_histroy(config: dict[str, str]) -> History:
    type = config.get('history.type', 'on_memory')
    if type == 'on_memory':
        return OnMemoryHistory()
    elif type == 'on_storage':
        return OnStorageHistory(output_dir=config['history.on_storage.output_dir'])
    else:
        raise Exception(f'Unknown history type: {type}')


class Metrics:
    def __init__(self, board: Board) -> None:
        max_size = -1
        empty_count = 0
        balls = {}
        for tube in board:
            size = len(tube)
            if size == 0:
                empty_count += 1
            elif max_size == -1:
                max_size = size
            else:
                if size != max_size:
                    raise Exception('different tube size')
            
            for ball in tube:
                balls[ball] = balls.get(ball, 0) + 1
        
        if empty_count < 2:
            raise Exception('empty tube not enough')
        
        for ball, count in balls.items():
            if count != max_size:
                raise Exception(f'the number of "{ball}" balls must be {max_size}, but {count}.')
        
        self.__board = board
        self.__balls = balls
        self.__max_size = max_size


    @property
    def board(self) -> Board:
        return self.__board


    @property
    def balls(self) -> dict[Any, int]:
        return self.__balls
    

    @property
    def max_size(self) -> int:
        return self.__max_size


class EventHandler(Protocol):
    def on_begin(self) -> None: ...

    def on_search(self, current_path: MovePath, shortest_path: MovePath) -> None: ...

    def on_shortest_path(self, shortest_path: MovePath) -> None: ...

    def on_finish(self) -> None: ...


class NoopEventHandler:
    def on_begin(self) -> None:
        pass


    def on_search(self, current_path: MovePath, shortest_path: MovePath) -> None:
        pass


    def on_shortest_path(self, shortest_path: MovePath) -> None:
        pass


    def on_finish(self) -> None:
        pass


class Game:
    def __init__(self, board: Board, config: dict[str, str]) -> None:
        self.__config = config
        self.__board = board
        self.__metrics = Metrics(board)


    @property
    def board(self) -> Board:
        return self.__board


    @property
    def metrics(self) -> Metrics:
        return self.__metrics
    

    def __validate_board(self, board: Board) -> None:
        ball_count = {}
        for tube in board:
            if len(tube) > self.metrics.max_size:
                raise Exception(f'tube size over. {tube}', board)
            
            for ball in tube:
                ball_count[ball] = ball_count.get(ball, 0) + 1
        
        for ball, count in ball_count.items():
            if count != self.metrics.max_size:
                raise Exception(f'the number of "{ball}" balls must be {self.metrics.max_size}, but {count}.', board)


    def __push(self, tube: tuple[*Ts], value: T) -> tuple[*Ts, T]:
        if self.__is_full(tube):
            raise Exception(f'Tube is full: {tube}')
        
        return tube + (value,)
    
    
    def __pop(self, tube: tuple[*Ts, Any] | tuple[()]) -> tuple[*Ts]:
        if self.__is_empty(tube):
            raise Exception(f'Tube is empty: {tube}')
        
        return tube[:-1] # type: ignore
    
    
    def __peek(self, tube: tuple[*Ts, T] | tuple[()]) -> T:
        if self.__is_empty(tube):
            raise Exception(f'Tube is empty: {tube}')
        
        return tube[-1] # type: ignore
    
    
    def __is_empty(self, tube: tuple[Any, ...]) -> bool:
        return len(tube) == 0


    def __is_full(self, tube: tuple[Any, ...]) -> bool:
        return len(tube) >= self.metrics.max_size


    def __is_movable(self, src_tube: tuple, dst_tube: tuple) -> bool:
        result = False
    
        if not self.__is_empty(src_tube) and not self.__is_tube_completed(src_tube) and not self.__is_full(dst_tube):
            if self.__is_empty(dst_tube) or (self.__peek(src_tube) == self.__peek(dst_tube)):
                result = True
    
        return result


    def __move(self, node: MoveNode, src_index: int, dst_index : int) -> MoveNode:
        def move_func() -> Iterator[tuple]:
            for i, _ in enumerate(node):
                if i == src_index:
                    yield self.__pop(node[i])
                elif i == dst_index:
                    yield self.__push(node[i], self.__peek(node[src_index]))
                else:
                    yield node[i]
    
        return MoveNode(tuple(move_func()))


    def __is_tube_completed(self, tube: tuple):
        result = self.__is_full(tube)
    
        if result:
            x = self.__peek(tube)
            for y in tube:
                if y != x:
                    result = False
                    break
    
        return result


    def __is_board_completed(self, board: Board) -> bool:
        result = True
        for tube in board:
            if not self.__is_empty(tube) and not self.__is_tube_completed(tube):
                result = False
                break
    
        return result


    def describe(self) -> None:
        print('===== game description =====')
        print('== board')
        print_board(self.board)
        print('== max size')
        print(self.metrics.max_size)
        sys.stdout.flush()


    def solve(self, event_handler: EventHandler = NoopEventHandler()) -> MovePath:
        event_handler.on_begin()

        # 処理した盤面の履歴
        history = create_histroy(self.__config)

        # 解までの最短のパス
        shortest_path = MovePath()

        # 探索中のパス
        current_path = MovePath(self.__board)
        for depth, node in enumerate(current_path):
            history.add(node, depth)

        # 盤面の検証を実施するかどうか
        board_validation = bool(config.get('board.validation', 'True'))

        # 探索を行う関数
        def search_shortest_path(previous_dst_index: int = -1):
            # nonlocal宣言
            nonlocal shortest_path

            # イベントハンドラを起動
            event_handler.on_search(shortest_path=shortest_path, current_path=current_path)

            # 最短のパスがまだ見つかっていないか、探索中のパスがまだ最短のパスより短い場合に処理を進める
            if (len(shortest_path) == 0 or (len(current_path) < len(shortest_path))):
                # 探索中のパスの最後尾のMoveNodeを取得する
                node = current_path.peek()
    
                # 盤面が完了状態かどうか
                if self.__is_board_completed(node.board):
                    # 最短のパスを更新
                    shortest_path = copy.deepcopy(current_path)

                    # イベントハンドラを起動
                    event_handler.on_shortest_path(shortest_path)
                else:
                    # 現在の盤面で、移動元と移動先の組み合わせを生成して探索
                    for src_index, dst_index in itertools.product(range(len(node.board)), repeat=2):
                        # 意味のないballの移動を除外
                        # 1. 移動元と移動先が同じ
                        # 2. 前のレベルで移動したballを再度移動する
                        if src_index in [dst_index, previous_dst_index]:
                            continue

                        # ballが移動可能であれば、処理を進める
                        if self.__is_movable(node.board[src_index], node.board[dst_index]):
                            # ballを移動する
                            next_node = self.__move(node, src_index, dst_index)

                            # ball移動後の盤面の検証を実施する
                            if board_validation:
                                self.__validate_board(next_node.board)
                            
                            # 履歴に追加できれば、次のレベルの探索に進む
                            if history.add(next_node, len(current_path)):
                                current_path.append(next_node)
                                search_shortest_path(previous_dst_index=dst_index)
                                current_path.pop()
                            else:
                                #current_path.print('already in path', target=next_board)
                                pass
    
        try:
            search_shortest_path()
        except Exception as e:
            current_path.print()
            raise e
    
        event_handler.on_finish()

        return shortest_path


if __name__ == '__main__':
    board0 = (
    	(0, 1, 2, 2),
    	(0, 1, 0, 1),
    	(2, 0, 1, 2),
    	(),
    	(),
    )
    board1 = (
        ("水", "緑", "緑", "緑", "緑"),
        ("紫", "桃", "黄", "桃", "藤"),
        ("水", "桃", "橙", "水", "黄"),
        ("藤", "青", "紫", "藤", "黄"),
        ("藤", "橙", "赤", "青", "水"),
        ("橙", "緑", "青", "藤", "青"),
        ("水", "青", "桃", "橙", "黄"),
        ("赤", "紫", "黄", "赤", "紫"),
        ("紫", "赤", "桃", "橙", "赤"),
        (),
        (),
    )

    board = board1
    
    config = {
        'history.type': 'on_memory',
        'history.on_storage.output_dir': './output'
    }

    game = Game(
        board=board,
        config=config
    )
    
    game.describe()
    path = game.solve()
    path.print()
