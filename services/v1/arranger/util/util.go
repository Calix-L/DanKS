package util

import "math/rand"

func Assert(ok bool) {
	if !ok {
		panic("assertion failed")
	}
}

func FisherYates[T any](rnd *rand.Rand, values []T) {
	rnd.Shuffle(len(values), func(i, j int) {
		values[i], values[j] = values[j], values[i]
	})
}
