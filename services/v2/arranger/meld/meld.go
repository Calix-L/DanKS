package meld

type Meld struct {
	Tiles   []int8
	Pattern int8
	Type    int8
	Value   int8
	Numeric bool
	Fixed   bool
}

func (m *Meld) Load(tiles []int8) {
	m.Tiles = append(m.Tiles[:0], tiles...)
}

func (m *Meld) Repeat(tile, count int8) {
	m.Tiles = m.Tiles[:0]
	for i := int8(0); i < count; i++ {
		m.Tiles = append(m.Tiles, tile)
	}
}
