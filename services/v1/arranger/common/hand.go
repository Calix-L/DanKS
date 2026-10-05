package common

import (
	"bytes"
	"fmt"
	"math/rand"

	"github.com/guandan-service/hand-arranger/meld"
	"github.com/guandan-service/hand-arranger/util"
)

// 花色
const (
	SUIT_SPADE   = 1 // 黑桃
	SUIT_HEART   = 2 // 红桃
	SUIT_CLUB    = 3 // 梅花
	SUIT_DIAMOND = 4 // 方块
	SUIT_JOKER   = 5 // 小丑
)

// 牌值
const (
	VAL_A  = 1
	VAL_2  = 2
	VAL_3  = 3
	VAL_4  = 4
	VAL_5  = 5
	VAL_6  = 6
	VAL_7  = 7
	VAL_8  = 8
	VAL_9  = 9
	VAL_10 = 10
	VAL_J  = 11
	VAL_Q  = 12
	VAL_K  = 13
	VAL_BJ = 14 // 小王 blackjoker
	VAL_RJ = 15 // 大王 redjoker
)

const (
	IDX_A  = VAL_A - 1
	IDX_2  = VAL_2 - 1
	IDX_3  = VAL_3 - 1
	IDX_5  = VAL_5 - 1
	IDX_10 = VAL_10 - 1
	IDX_J  = VAL_J - 1
	IDX_Q  = VAL_Q - 1
	IDX_K  = VAL_K - 1
)

const (
	TILE_RJ    = int8(SUIT_JOKER)<<4 | VAL_RJ // 大鬼
	TILE_BJ    = int8(SUIT_JOKER)<<4 | VAL_BJ // 小鬼
	HAND_COUNT = 27
)

var (
	_points = [16]int8{ // 按照牌值索引的点数（用于比较大小）：大王>小王>A>K>Q>J>10>9>8>7>6>5>4>3>2
		0,  // N/A
		13, // VAL_A
		1,  // VAL_2
		2,  // VAL_3
		3,  // VAL_4
		4,  // VAL_5
		5,  // VAL_6
		6,  // VAL_7
		7,  // VAL_8
		8,  // VAL_9
		9,  // VAL_10
		10, // VAL_J
		11, // VAL_Q
		12, // VAL_K
		14, // VAL_BJ
		15, // VAL_RJ
	}
)

func GetSuit(t int8) int8 {
	return t >> 4
}

func GetValue(t int8) int8 {
	return t & 0xf
}

func MakeTile(suit, value int8) int8 {
	return suit<<4 | value
}

func MakeLaizi(value int8) int8 {
	util.Assert(value >= VAL_A && value <= VAL_K)
	return MakeTile(SUIT_HEART, value)
}

func SplitTile(t int8) (int8, int8) {
	return t >> 4, t & 0xf
}

func IsValidTile(tile int8) bool {
	s, v := SplitTile(tile)
	if s >= 1 && s <= 4 {
		return v >= 1 && v <= 13
	} else if s == 5 {
		return v >= VAL_BJ && v <= VAL_RJ
	}
	return false
}

type Hand struct {
	size   int         // 手牌数量
	counts [4][13]int8 // 花色牌
	jokers [2]int8     // 0:小王,1:大王
}

func (this *Hand) CopyFrom(hand *Hand) {
	this.size = hand.size
	this.counts = hand.counts
	this.jokers = hand.jokers
}

func (this *Hand) Add(tiles ...int8) {
	for _, tile := range tiles {
		this._add(tile)
	}
	this.size += len(tiles)
}

func (this *Hand) Count(tile int8) int8 {
	p := this._locate(tile)
	return *p
}

// 删除指定的手牌（如果其中一张牌删除失败不会改变原有手牌）
func (this *Hand) Remove(tiles ...int8) bool {
	for i, t1 := range tiles {
		if !this._remove(t1) {
			for _, t2 := range tiles[:i] {
				this._add(t2)
			}
			return false
		}
	}
	this.size -= len(tiles)
	return true
}

func (this *Hand) _locate(tile int8) *int8 {
	s, v := SplitTile(tile)
	var p *int8
	if s >= 1 && s <= 4 {
		p = &this.counts[s-1][v-1]
	} else if s == 5 && v >= VAL_BJ && v <= VAL_RJ {
		p = &this.jokers[v-VAL_BJ]
	} else {
		panic(fmt.Errorf("invalid tile=0x%x", tile))
	}
	return p
}

func (this *Hand) _remove(tile int8) bool {
	p := this._locate(tile)
	if *p <= 0 {
		return false
	}
	*p--
	return true
}

func (this *Hand) _add(tile int8) {
	p := this._locate(tile)
	*p++
}

func (this *Hand) Clear() {
	this.counts = [4][13]int8{}
	this.jokers = [2]int8{}
	this.size = 0
}

func (this *Hand) Size() int {
	return this.size
}

func (this *Hand) IsEmpty() bool {
	return this.size == 0
}

func (this *Hand) String() string {
	if this.size == 0 {
		return "[]"
	}
	b := bytes.NewBuffer(make([]byte, 0, 128))
	b.WriteString("0x[")
	var count int8
	for i, counts := range this.counts {
		suit := int8(i + 1)
		for j, n := range counts {
			if n == 0 {
				continue
			}
			tile := MakeTile(suit, int8(j+1))
			for k := int8(0); k < n; k++ {
				if count > 0 {
					b.WriteByte(',')
				}
				fmt.Fprintf(b, "%x", tile)
				count++
			}
		}
	}
	for i, n := range this.jokers {
		if n == 0 {
			continue
		}
		tile := MakeTile(SUIT_JOKER, int8(i+VAL_BJ))
		for j := int8(0); j < n; j++ {
			if count > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(b, "%x", tile)
			count++
		}
	}
	b.WriteByte(']')
	return b.String()
}

func (this *Hand) CopyTo(tiles []int8) []int8 {
	if this.size == 0 {
		return tiles
	}
	for i, counts := range this.counts {
		suit := int8(i + 1)
		for j, n := range counts {
			if n == 0 {
				continue
			}
			tile := MakeTile(suit, int8(j+1))
			for range n {
				tiles = append(tiles, tile)
			}
		}
	}
	for i, n := range this.jokers {
		if n == 0 {
			continue
		}
		tile := MakeTile(SUIT_JOKER, int8(i+VAL_BJ))
		for range n {
			tiles = append(tiles, tile)
		}
	}
	return tiles
}

func (this *Hand) Reload(tiles []int8) {
	this.Clear()
	this.Add(tiles...)
}

// 检查进贡的牌（逢人配不可以用来进贡；如果最大的牌是逢人配，则选第2大的牌作为进贡牌）
func (this *Hand) CheckTribute(level, tile int8) bool {
	s, v := SplitTile(tile)
	if s == SUIT_JOKER {
		if v == VAL_RJ {
			return true
		} else {
			return this.Count(TILE_RJ) == 0
		}
	}
	if s == SUIT_HEART && v == level {
		// 不能进贡赖子
		return false
	}
	if this.jokers[0]+this.jokers[1] > 0 {
		return false
	}
	for i, counts := range this.counts {
		s2 := int8(i + 1)
		for j, n := range counts {
			if n == 0 {
				continue
			}
			v2 := int8(j + 1)
			// 忽略赖子
			if s2 == SUIT_HEART && v2 == level {
				continue
			}
			if Follow(level, v, v2) {
				return false
			}
		}
	}
	return true
}

// 是否手牌中最小的
func (this *Hand) IsMin(level, tile int8) bool {
	value := GetValue(tile)
	for _, counts := range this.counts {
		for i, n := range counts {
			v := int8(i + 1)
			if n > 0 && Follow(level, v, value) {
				return false
			}
		}
	}
	for i, n := range this.jokers {
		if n > 0 && Follow(level, int8(VAL_BJ+i), value) {
			return false
		}
	}
	return true
}

// 是否手牌中除了拖牌外最小的
func (this *Hand) IsMin2(level, tuo, tile int8) bool {
	value := GetValue(tile)
	for _, counts := range this.counts {
		for i, n := range counts {
			v := int8(i + 1)
			if n > 0 && tuo != v && Follow(level, v, value) {
				return false
			}
		}
	}
	for i, n := range this.jokers {
		if n > 0 && Follow(level, int8(VAL_BJ+i), value) {
			return false
		}
	}
	return true
}

// 返回进贡的牌值（如果最大的牌只有逢人配，则返回第二大的牌）
func (this *Hand) GetTributionValue(level int8) int8 {
	if this.jokers[1] > 0 {
		return VAL_RJ
	}
	if this.jokers[0] > 0 {
		return VAL_BJ
	}
	index := level - 1
	for i := 0; i < 4; i++ {
		if i+1 != SUIT_HEART && this.counts[i][index] > 0 {
			return level
		}
	}
	for _, i := range _straight {
		if i == index {
			continue
		}
		for j := 0; j < 4; j++ {
			if this.counts[j][i] > 0 {
				return i + 1
			}
		}
	}
	panic("unreachable")
}

// 返回最大的牌（如果最大的牌只有逢人配，则返回第二大的牌）
func (this *Hand) GetTributionTile(level int8) int8 {
	if this.jokers[1] > 0 {
		return TILE_RJ
	}
	if this.jokers[0] > 0 {
		return TILE_BJ
	}
	index := level - 1
	for i := int8(0); i < 4; i++ {
		if i+1 != SUIT_HEART && this.counts[i][index] > 0 {
			return MakeTile(i+1, level)
		}
	}
	for _, i := range _straight {
		if i == index {
			continue
		}
		for j := int8(0); j < 4; j++ {
			if this.counts[j][i] > 0 {
				return MakeTile(j+1, i+1)
			}
		}
	}
	panic("unreachable")
}

// 返回还贡的牌
func (this *Hand) GetReturnTile(level int8) int8 {
	index := level - 1
	// 优先单牌、对子、三张中小于等于10的最小的牌
	var count int8
	for i := int8(IDX_2); i <= IDX_10; i++ {
		if i == index {
			continue
		}
		count = 0
		for j := int8(0); j < 4; j++ {
			count += this.counts[j][i]
		}
		if count > 0 && count < 4 {
			for j := int8(0); j < 4; j++ {
				if this.counts[j][i] > 0 {
					return MakeTile(j+1, i+1)
				}
			}
			panic("unreachable")
		}
	}
	// 最小的牌
	for i := len(_straight) - 1; i >= 0; i-- {
		vi := _straight[i]
		if vi == index {
			continue
		}
		for j := int8(0); j < 4; j++ {
			if this.counts[j][vi] > 0 {
				return MakeTile(j+1, vi+1)
			}
		}
	}
	for _, i := range [4]int8{0, 2, 3, 1} {
		if this.counts[i][index] > 0 {
			return MakeTile(i+1, level)
		}
	}
	if this.jokers[0] > 0 {
		return TILE_BJ
	}
	if this.jokers[1] > 0 {
		return TILE_RJ
	}
	panic("unreachable")
}

// 是否整套牌（两副完整扑克牌）
func (hand *Hand) IsFullSuite() bool {
	if hand.jokers[0] != 2 || hand.jokers[1] != 2 {
		return false
	}
	for i := range len(hand.counts) {
		for _, n := range hand.counts[i] {
			if n != 2 {
				return false
			}
		}
	}
	return true
}

// 计算拖牌数量（红桃拖牌算2张）
func (hand *Hand) CountTuo(tuo int8) int8 {
	util.Assert(tuo > 0)
	var count int8
	index := tuo - 1
	const IDX_HEART = SUIT_HEART - 1
	for i, list := range hand.counts {
		n := list[index]
		if n == 0 {
			continue
		}
		if i == IDX_HEART {
			count += n * 2
		} else {
			count += n
		}
	}
	return count
}

func InitTiles(rnd *rand.Rand, tiles []int8) []int8 {
	tiles = tiles[:0]
	for s := int8(1); s <= 4; s++ {
		for v := int8(1); v <= 13; v++ {
			tile := MakeTile(s, v)
			tiles = append(tiles, tile)
		}
	}
	tiles = append(tiles, TILE_BJ, TILE_RJ)
	tiles = append(tiles, tiles[:54]...)
	util.Assert(len(tiles) == 108)
	return tiles
}

// 金币场洗牌（经典场）
func InitGoldTiles1(rnd *rand.Rand, tiles []int8) []int8 {
	tiles = tiles[:0]
	for s := int8(1); s <= 4; s++ {
		for v := int8(1); v <= 13; v++ {
			tile := MakeTile(s, v)
			tiles = append(tiles, tile)
		}
	}
	tiles = append(tiles, TILE_BJ, TILE_RJ)
	tiles = append(tiles, tiles[:54]...)
	util.Assert(len(tiles) == 108)
	for range 5 {
		util.FisherYates(rnd, tiles)
	}
	return tiles
}

// 金币场洗牌（不洗牌）
func InitGoldTiles2(rnd *rand.Rand, tiles []int8) []int8 {
	list := make([]int8, 0, 108)
	type Rate struct {
		size int16
		odds int16
	}
	rates := []Rate{
		{
			size: 4, // 4炸
			odds: 400,
		},
		{
			size: 5, // 5炸
			odds: 300,
		},
		{
			size: 6, // 6炸
			odds: 100,
		},
		{
			size: 7, // 7炸
			odds: 30,
		},
		{
			size: 8, // 8炸
			odds: 1,
		},
	}
	makeBomb := func() (int16, bool) {
		odds := int16(rnd.Intn(1000))
		var sum int16
		for _, it := range rates {
			sum += it.odds
			if odds < sum {
				return it.size, true
			}
		}
		return 0, false
	}
	type Box struct {
		chips  [][]int8
		values [13]int8 // 牌值数量
		joker  int
	}
	var boxes [4]Box
	for i := range len(boxes) {
		boxes[i].chips = make([][]int8, 0, 27)
	}
	for v := int8(1); v <= 13; v++ {
		for s := int8(1); s <= 4; s++ {
			tile := MakeTile(s, v)
			list = append(list, tile, tile)
		}
		index := v - 1
		chip := list[len(list)-8:]
		util.Assert(len(chip) == 8)
		size, ok := makeBomb()
		if ok {
			util.Assert(size >= 4 && size <= 8)
			// 炸弹
			bomb := chip[:size]
			chip = chip[size:]
			pos := rnd.Intn(len(boxes))
			for {
				ok = false
				for range len(boxes) {
					box := &boxes[pos]
					if box.values[index] == 0 {
						box.chips = append(box.chips, bomb)
						box.values[index] += int8(len(bomb))
						ok = true
						break
					}
					pos = (pos + 1) % len(boxes)
				}
				util.Assert(ok)
				// 有几率继续组炸弹（对于四炸），以增加炸弹数量
				if len(chip) < 4 || rnd.Intn(100) < 10 {
					break
				}
				// 把额外的炸弹放到离先前炸弹比较远的位置，减小重组为8炸的可能
				pos = (pos + 2) % len(boxes)
				bomb = chip
				chip = chip[len(chip):]
			}
		}
		for len(chip) > 0 {
			size := len(chip)
			if size >= 4 {
				size = 3
			}
			pos := rnd.Intn(len(boxes))
			ok := false
			for range len(boxes) {
				box := &boxes[pos]
				// 尽量防止组成更大的炸弹
				if box.values[index] == 0 {
					box.chips = append(box.chips, chip[:size])
					box.values[index] += int8(size)
					chip = chip[size:]
					ok = true
					break
				}
				pos = (pos + 1) % len(boxes)
			}
			util.Assert(ok)
		}
	}
	list = append(list, TILE_RJ, TILE_RJ, TILE_BJ, TILE_BJ)
	util.Assert(len(list) == 108)
	chip := list[len(list)-4:]
	util.Assert(len(chip) == 4)
	// 天王炸概率
	if rnd.Intn(1000) < 100 {
		box := &boxes[rnd.Intn(len(boxes))]
		box.chips = append(box.chips, chip)
	} else {
		for len(chip) > 0 {
			pos := rnd.Intn(len(boxes))
			var ok bool
			for range len(boxes) {
				box := &boxes[pos]
				// 尽量防止组成天王炸弹
				if box.joker < 3 {
					box.chips = append(box.chips, chip[:1])
					box.joker++
					chip = chip[1:]
					ok = true
					break
				}
				pos = (pos + 1) % len(boxes)
			}
			util.Assert(ok)
		}
	}
	tiles = tiles[:0]
	for _, box := range boxes {
		for _, chip := range box.chips {
			tiles = append(tiles, chip...)
		}
	}
	util.Assert(len(tiles) == 108)
	return tiles
}

// 不洗牌
func InitTiles3(rnd *rand.Rand, tiles []int8) []int8 {
	type Rate struct {
		odds int16
		size int8
	}
	rates := []Rate{
		{
			size: 4, // 4炸
			odds: 400,
		},
		{
			size: 5, // 5炸
			odds: 300,
		},
		{
			size: 6, // 6炸
			odds: 100,
		},
		{
			size: 7, // 7炸
			odds: 30,
		},
		{
			size: 8, // 8炸
			odds: 1,
		},
	}
	makeBomb := func() (int8, bool) {
		odds := int16(rnd.Intn(1000))
		var sum int16
		for _, it := range rates {
			sum += it.odds
			if odds < sum {
				return it.size, true
			}
		}
		return 0, false
	}
	makeChip := func(value int8) [8]int8 {
		var chip [8]int8
		index := 0
		for s := int8(1); s <= 4; s++ {
			tile := MakeTile(s, value)
			chip[index] = tile
			index++
			chip[index] = tile
			index++
		}
		return chip
	}
	const (
		ITEM_KING     = 1 // 天王炸弹
		ITEM_BOMB     = 2 // 同张炸弹
		ITEM_MULTIPLE = 3 // 三张、对子、单张
	)
	type Item struct {
		tiles [8]int8
		size  int8
		typ   int8
		index int8
	}
	items := make([]Item, 108)
	var count int
	for i := int8(IDX_A); i <= IDX_K; i++ {
		value := i + 1
		chip := makeChip(value)
		size, ok := makeBomb()
		if ok {
			util.Assert(size >= 4 && size <= 8)
			item := &items[count]
			count++
			item.typ = ITEM_BOMB
			item.size = size
			item.index = i
			copy(item.tiles[:size], chip[:size])
			if size > 4 {
				// 炸弹剩余部分：如果长度小于4则不切分
				item := &items[count]
				count++
				item.typ = ITEM_MULTIPLE
				item.size = 8 - size
				item.index = i
				copy(item.tiles[:item.size], chip[size:])
			} else {
				util.Assert(size == 4)
				if rnd.Intn(100) < 90 {
					// 有几率继续组炸弹（对于四炸），以增加炸弹数量
					item = &items[count]
					count++
					item.typ = ITEM_BOMB
					item.size = size
					item.index = i
					copy(item.tiles[:4], chip[4:])
				} else {
					// 炸弹剩余部分：如果长度等于4，分为2/2
					for range 2 {
						item = &items[count]
						count++
						item.typ = ITEM_MULTIPLE
						item.size = 2
						item.index = i
						copy(item.tiles[:2], chip[size:size+2])
						size += 2
					}
				}
			}
		} else {
			// 非炸弹：切分为3/3/2
			size = 0
			for size < 8 {
				n := 8 - size
				if n >= 4 {
					n = 3
				}
				item := &items[count]
				count++
				item.typ = ITEM_MULTIPLE
				item.size = n
				item.index = i
				copy(item.tiles[:n], chip[size:size+n])
				size += n
			}
		}
	}
	// 大小王
	if rnd.Intn(100) < 10 {
		// 天王炸
		item := &items[count]
		count++
		item.typ = ITEM_KING
		item.index = -1
		item.size = 4
		for i, t := range [4]int8{TILE_RJ, TILE_RJ, TILE_BJ, TILE_BJ} {
			item.tiles[i] = t
		}
	} else {
		for _, t := range [2]int8{TILE_RJ, TILE_BJ} {
			if rnd.Intn(100) < 50 {
				item := &items[count]
				count++
				item.typ = ITEM_MULTIPLE
				item.size = 2
				item.index = -1
				item.tiles[0] = t
				item.tiles[1] = t
			} else {
				for range 2 {
					item := &items[count]
					count++
					item.typ = ITEM_MULTIPLE
					item.size = 1
					item.index = -1
					item.tiles[0] = t
				}
			}
		}
	}
	// 洗牌
	for i := count; i > 0; {
		j := rnd.Intn(i)
		i--
		if i != j {
			items[i], items[j] = items[j], items[i]
		}
	}
	type Box struct {
		items  []*Item
		values [15]int8
	}
	var boxes [4]Box
	for i := range len(boxes) {
		box := &boxes[i]
		box.items = make([]*Item, 0, 16)
	}
	for i := range count {
		item := &items[i]
		pos := rnd.Intn(len(boxes))
		box := &boxes[pos]
		if item.index < 0 {
			box.items = append(box.items, item)
			continue
		}
		ok := false
		for range len(boxes) {
			if box.values[item.index] == 0 {
				box.values[item.index]++
				box.items = append(box.items, item)
				ok = true
				break
			}
			pos = (pos + 1) % len(boxes)
		}
		util.Assert(ok)
	}
	tiles = tiles[:0]
	for i := range len(boxes) {
		box := &boxes[i]
		for _, item := range box.items {
			tiles = append(tiles, item.tiles[:item.size]...)
		}
	}
	util.Assert(len(tiles) == 108)
	return tiles
}

// 是否拖牌成功，返回拖牌张数（红桃拖牌算2张）
func CheckTuo(tiles []int8, tuo int8) (int8, bool) {
	var count int8
	for _, t := range tiles {
		s, v := SplitTile(t)
		if v != tuo {
			return 0, false
		}
		if s == SUIT_HEART {
			count += 2
		} else {
			count++
		}
	}
	return count, true
}

// 获取拖牌牌值
func GetTuo(level int8) int8 {
	if level != VAL_2 {
		return VAL_2
	} else {
		return VAL_3
	}
}

func FindValue(meld *meld.Meld, value int8) bool {
	for _, t := range meld.Tiles {
		if GetValue(t) == value {
			return true
		}
	}
	return false
}
