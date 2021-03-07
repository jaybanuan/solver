package main

import (
	"fmt"
	"reflect"
)

func is_movable(src_entry []int, dst_entry []int) bool {
	var result = false

	if len(src_entry) > 0 && !is_entry_completed(src_entry) && !is_full(dst_entry) {
		if len(dst_entry) == 0 || (src_entry[len(src_entry)-1] == dst_entry[len(dst_entry)-1]) {
			result = true
		}
	}

	return result
}

func move(src_entry []int, dst_entry []int) ([]int, []int) {
	var new_dst_entry = make([]int, len(dst_entry), cap(dst_entry))
	copy(new_dst_entry, dst_entry)
	new_dst_entry = append(new_dst_entry, src_entry[len(src_entry)-1])

	var new_src_entry = src_entry[:len(src_entry)-1]

	return new_src_entry, new_dst_entry
}

func is_full(entry []int) bool {
	return len(entry) >= 4
}

func is_entry_completed(entry []int) bool {
	var result = is_full(entry)

	if result {
		var sample_value = entry[0]
		for _, value := range entry {
			if value != sample_value {
				result = false
				break
			}
		}
	}

	return result
}

func is_table_completed(table [][]int) bool {
	var result = true

	for _, entry := range table {
		if len(entry) > 0 && !is_entry_completed(entry) {
			result = false
			break
		}
	}

	return result
}

func contains_in_path(path [][][]int, table [][]int) bool {
	var result = false

	for _, t := range path {
		if reflect.DeepEqual(t, table) {
			result = true
		}
	}

	return result
}

func solve(table [][]int) [][][]int {
	var path = append(make([][][]int, 0, 1024), table)
	var shortest_path [][][]int = nil

	var search_shortest_path func()

	search_shortest_path = func() {
		if shortest_path == nil || len(path) < len(shortest_path) {
			var table = path[len(path)-1]

			if is_table_completed(table) {
				shortest_path = make([][][]int, len(path), cap(path))
				copy(shortest_path, path)
			} else {
				for src_index, _ := range table {
					for dst_index, _ := range table {
						if src_index == dst_index {
							continue
						}

						if is_movable(table[src_index], table[dst_index]) {
							var next_table = make([][]int, len(table), cap(table))
							copy(next_table, table)

							next_table[src_index], next_table[dst_index] = move(next_table[src_index], next_table[dst_index])

							if !contains_in_path(path, next_table) {
								path = append(path, next_table)
								search_shortest_path()
								path = path[:len(path)-1]
							}
						}
					}
				}
			}
		}
	}

	search_shortest_path()

	return shortest_path
}

var table = [][]int{
	{0, 1, 2, 2},
	{0, 1, 0, 1},
	{2, 0, 1, 2},
	{},
	{},
}

/*
var table = [][]int{
	{0, 1, 0, 1},
	{1, 0, 1, 0},
	{},
}
*/

func main() {
	var shortest_path = solve(table)

	for path_index, t := range shortest_path {
		fmt.Printf("===== %v =============\n", path_index)
		for _, entry := range t {
			fmt.Println(entry)
		}
	}
}
