package common

import (
	"fmt"
	"sort"

	"github.com/guandan-service/hand-arranger/meld"
	"github.com/guandan-service/hand-arranger/util"
)

const (
	MODE_BOMB   = 1 // 炸弹优先：炸弹数量最多=>同花顺数量最多=>手数最少
	MODE_FLUSH  = 2 // 同花顺优先：同花顺数量最多=>炸弹最多=>手数最少
	MODE_QUICK  = 3 // 整牌优先：单牌最少=>炸弹数量最多=>同花顺数量最多
	MODE_SIMPLE = 4 // 简单理牌：不组顺子、同花顺，组天王炸弹，用于金币场
)

func IsValidMode(mode int8) bool {
	return mode >= MODE_BOMB && mode <= MODE_SIMPLE
}

const (
	VALUE_MAXCOUNT = 12 // 同一个数字牌最多的张数
	LAIZI_MAXCOUNT = 2  // 最多两个红心级牌
)

type Meld struct {
	Tiles [VALUE_MAXCOUNT + LAIZI_MAXCOUNT]int8 // 牌数组
	Group Group                                 // 牌型信息
	Fixed bool                                  // 是否锁牌
}

func (this Meld) String() string {
	return fmt.Sprintf("\n{pattern=%d,minor=%d,value=%d,size=%d,tiles=0x%x}", this.Group.Pattern, this.Group.Minor, this.Group.Value, this.Group.Size, this.List())
}

func (this *Meld) Equal(pattern, minor, value, size int8, tiles []int8) bool {
	if pattern != this.Group.Pattern || minor != this.Group.Minor || value != this.Group.Value || size != this.Group.Size {
		return false
	}
	if len(tiles) != int(this.Group.Size) {
		return false
	}
	m := make(map[int8]int8, 8)
	for _, t := range tiles {
		m[t]++
	}
	for _, t := range this.Tiles[:len(tiles)] {
		n, ok := m[t]
		if !ok || n == 0 {
			return false
		}
		m[t]--
	}
	return true
}

func (this *Meld) Equal2(pattern, minor, value int8, tiles []int8) bool {
	if pattern != this.Group.Pattern || minor != this.Group.Minor || value != this.Group.Value {
		return false
	}
	if len(tiles) != int(this.Group.Size) {
		return false
	}
	m := make(map[int8]int8, 8)
	for _, t := range tiles {
		m[t]++
	}
	for _, t := range this.Tiles[:len(tiles)] {
		n, ok := m[t]
		if !ok || n == 0 {
			return false
		}
		m[t]--
	}
	return true
}

func (this *Meld) clear() {
	this.Group.Clear()
}

func (this *Meld) Reload(tiles ...int8) {
	copy(this.Tiles[:len(tiles)], tiles)
	this.Group.Size = int8(len(tiles))
}

func (this *Meld) addTiles(tiles ...int8) {
	for _, t := range tiles {
		this.Tiles[this.Group.Size] = t
		this.Group.Size++
	}
}

func (this *Meld) addTile(tile, count int8) {
	for i := int8(0); i < count; i++ {
		this.Tiles[this.Group.Size] = tile
		this.Group.Size++
	}
}

func (this *Meld) List() []int8 {
	return this.Tiles[:this.Group.Size]
}

func makeGroup(p, m, v, s int8) Group {
	return Group{
		Pattern: p,
		Minor:   m,
		Value:   v,
		Size:    s,
	}
}

type Hint struct {
	next  *Hint
	list  [5]Seq
	group Group
	suit  int8 // 同花顺花色索引
}

func (this Hint) String() string {
	var list []Seq
	switch this.group.Pattern {
	case PATTERN_BOMB:
		if this.group.Minor == BOMB_FLUSH {
			list = this.list[:]
		} else {
			list = this.list[:1]
		}
	case PATTERN_STRAIGHT:
		list = this.list[:]
	case PATTERN_STRAIGHTPAIR:
		list = this.list[:3]
	case PATTERN_STRAIGHTTRIPLE:
		list = this.list[:2]
	}
	return fmt.Sprintf("{pattern=%d,minor=%d,value=%d,size=%d,list=0x%x}\n",
		this.group.Pattern, this.group.Minor, this.group.Value, this.group.Size, list)
}

func (this *Hint) clear() {
	clear(this.list[:])
	this.group = Group{}
	this.next = nil
}

// 数字牌
type Rank struct {
	counts [4]int8 // 不同花色数量
	value  int8    // 牌值
	total  int8    // 总数
}

func (this *Rank) init(value int8) {
	this.counts = [4]int8{}
	this.total = 0
	this.value = value
}

func (this Rank) dump(tiles []int8) []int8 {
	for i, n := range this.counts {
		if n > 0 {
			tile := MakeTile(int8(i+1), this.value)
			for j := int8(0); j < n; j++ {
				tiles = append(tiles, tile)
			}
		}
	}
	return tiles
}

func (this Rank) dumpMaxCount(tiles []int8, count int8) []int8 {
	var size int8
	for i, n := range this.counts {
		if n > 0 {
			tile := MakeTile(int8(i+1), this.value)
			if size+n >= count {
				for j := count - size; j > 0; j-- {
					tiles = append(tiles, tile)
				}
				return tiles
			} else {
				for j := int8(0); j < n; j++ {
					tiles = append(tiles, tile)
				}
				size += n
			}
		}
	}
	return tiles
}

// 删除指定牌数，total不变
func (this *Rank) pop(meld *Meld, count int8) {
	left := count
	for i := int8(3); i >= 0; i-- {
		n := this.counts[i]
		if n > 0 {
			tile := MakeTile(int8(i+1), this.value)
			if n >= left {
				this.counts[i] -= left
				meld.addTile(tile, left)
				return
			}
			meld.addTile(tile, n)
			left -= n
			this.counts[i] = 0
		}
	}
	panic("unreachable")
}

type Stack struct {
	hints [HAND_COUNT]Hint // 遍历出的无花色分组
	size  int              // 栈深度
}

func (this *Stack) reload(hints []Hint) {
	copy(this.hints[:len(hints)], hints)
	this.size = len(hints)
}

func (this *Stack) clear() {
	this.size = 0
}

func (this *Stack) pop() {
	util.Assert(this.size > 0)
	this.size--
}

func (this *Stack) alloc() *Hint {
	hint := &this.hints[this.size]
	hint.clear()
	this.size++
	return hint
}

func (this *Stack) list() []Hint {
	return this.hints[:this.size]
}

// 备选方案
type Motion struct {
	hints []Hint
	info  Stat
}

func (this *Motion) init() {
	this.hints = make([]Hint, 0, HAND_COUNT)
}

func (this *Motion) reset() {
	this.hints = this.hints[:0]
	this.info.reset()
}

func levelPattern(p int8) int8 {
	switch p {
	case PATTERN_STRAIGHTTRIPLE:
		return 4
	case PATTERN_STRAIGHTPAIR:
		return 3
	case PATTERN_TRIPLEPLUS:
		return 2
	case PATTERN_STRAIGHT:
		return 1
	case PATTERN_BOMB:
		return 6
	default:
		// case PATTERN_TRIPLE, PATTERN_PAIR, PATTERN_SINGLE:
		return 5
	}
}

func ToGroup(meld *meld.Meld) Group {
	var group Group
	group.Pattern = meld.Pattern
	group.Value = meld.Value
	group.Minor = meld.Type
	group.Size = int8(len(meld.Tiles))
	return group
}

func cmpGroup(g1, g2 Group, level int8) bool {
	l1 := levelPattern(g1.Pattern)
	l2 := levelPattern(g2.Pattern)
	if l1 != l2 {
		return l1 > l2
	}
	if l1 != 5 {
		return g1.Follow(level, g2)
	} else {
		// 三张、对子、单张按照牌值排序
		return Follow(level, g2.Value, g1.Value)
	}
}

// 理完牌后，会分为排在左边的牌型，和排在右边的牌型，中间的没理的牌，按原来正常时竖版的排列。
// 排在左边的，按炸弹的大小从左到右
// 排在右边的，从左到右，牌型按钢板>>连对>>三带对>>顺子， 同牌型的按左大右小排列
func SortMeld(melds []Meld, level int8) {
	sort.SliceStable(melds, func(i, j int) bool {
		return cmpGroup(melds[i].Group, melds[j].Group, level)
	})
}

func levelMeld(meld *meld.Meld) int8 {
	switch {
	case meld.Pattern == PATTERN_BOMB:
		if !meld.Numeric {
			return 1
		} else {
			return 2
		}
	case IsMultiple(meld.Pattern):
		return 2
	default:
		return 3
	}
}

func CmdMeld(m1, m2 *meld.Meld, level int8) bool {
	l1 := levelMeld(m1)
	l2 := levelMeld(m2)
	if l1 != l2 {
		return l1 < l2
	}
	switch l1 {
	case 1:
		// 非numeric标记的炸弹：按照炸弹大小从大到小排列
		return ToGroup(m1).Follow(level, ToGroup(m2))
	case 2:
		// 同张：单张、对子、三张和标记Numeric的炸弹，按照牌值从大到小排列
		return Follow(level, m2.Value, m1.Value)
	default:
		// 特定牌型：从左到右，牌型按钢板>>连对>>三带对>>顺子， 同牌型的按大小从左到右
		w1 := levelPattern(m1.Pattern)
		w2 := levelPattern(m2.Pattern)
		if w1 != w2 {
			return w1 > w2
		} else {
			return ToGroup(m1).Follow(level, ToGroup(m2))
		}
	}
}

// 理牌
type Splitter struct {
	best      *Motion                // 最优方案
	curr      *Motion                // 当前方案
	melds     []Meld                 // 分组
	stack     Stack                  // 递归栈
	env       Env                    // 遍历环境
	sequence  [STRAIGHT_MAXSIZE]int8 // 牌值大小次序（从大到小）
	level     int8                   // 级别
	laiziTile int8                   // 癞子牌
	size      int8                   // 牌数（不包括大小王）
}

func (this *Splitter) Init() {
	this.best = new(Motion)
	this.best.init()
	this.curr = new(Motion)
	this.curr.init()
	this.melds = make([]Meld, 0, HAND_COUNT)
}

func (this *Splitter) SetLevel(level int8) {
	this.level = level
	this.laiziTile = MakeTile(SUIT_HEART, level)
	if level == VAL_A {
		this.sequence[0] = IDX_A
		for i := 1; i < len(this.sequence); i++ {
			this.sequence[i] = int8(VAL_K - i)
		}
	} else {
		this.sequence[0] = level - 1
		this.sequence[1] = IDX_A
		index := 2
		for v := int8(VAL_K); v >= VAL_2; v-- {
			if v != level {
				this.sequence[index] = v - 1
				index++
			}
		}
	}
}

func (this *Splitter) load(hand *Hand) {
	util.Assert(hand.size > 0 && hand.size <= HAND_COUNT)
	util.Assert(this.level > 0)
	this.env.load(hand, this.level, this.laiziTile)
	this.size = int8(hand.size) - hand.jokers[0] - hand.jokers[1]
	this.stack.clear()
	this.best.reset()
	this.curr.reset()
	this.melds = this.melds[:0]
}

func repeat(tile, count int8) []int8 {
	s := make([]int8, count)
	for i := int8(0); i < count; i++ {
		s[i] = tile
	}
	return s
}

// 不理牌，直接排列
func (this *Splitter) Arrange(hand *Hand, alloc func() *meld.Meld) {
	pattern := func(n int8) int8 {
		util.Assert(n > 0)
		var p int8
		if n >= 4 {
			p = PATTERN_BOMB
		} else {
			switch n {
			case 3:
				p = PATTERN_TRIPLE
			case 2:
				p = PATTERN_PAIR
			default:
				p = PATTERN_SINGLE
			}
		}
		return p
	}
	if n := hand.jokers[1]; n > 0 {
		meld := alloc()
		meld.Pattern = pattern(n)
		meld.Value = VAL_RJ
		meld.Repeat(TILE_RJ, n)
	}
	if n := hand.jokers[0]; n > 0 {
		meld := alloc()
		meld.Pattern = pattern(n)
		meld.Value = VAL_BJ
		meld.Repeat(TILE_BJ, n)
	}
	var ranks [13]Rank
	for i, counts := range hand.counts {
		for j, n := range counts {
			if n > 0 {
				rank := &ranks[j]
				rank.counts[i] += n
				rank.total += n
			}
		}
	}
	var size int8
	for i := range len(ranks) {
		rank := &ranks[i]
		rank.value = int8(i + 1)
		size += rank.total
	}
	var typ int8
	for _, index := range this.sequence {
		rank := &ranks[index]
		if rank.total > 0 {
			if rank.total >= 4 {
				typ = BOMB_NORMAL
			} else {
				typ = 0
			}
			meld := alloc()
			meld.Tiles = rank.dump(meld.Tiles[:0])
			meld.Pattern = pattern(rank.total)
			meld.Type = typ
			meld.Value = rank.value
		}
	}
}

func listMeld(melds []Meld) []*Meld {
	list := make([]*Meld, len(melds))
	for i := 0; i < len(list); i++ {
		list[i] = &melds[i]
	}
	return list
}

func (this *Splitter) splitMode(hand *Hand, mode int8) []*Meld {
	melds := this.Split(hand, mode)
	SortMeld(melds, this.level)
	return listMeld(melds)
}

func (this *Splitter) _splitJoker(bj, rj int8) {
	if bj == 2 && rj == 2 {
		// 天王炸弹
		last := len(this.melds)
		this.melds = this.melds[:last+1]
		meld := &this.melds[last]
		meld.Group = makeGroup(PATTERN_BOMB, BOMB_JOKER, VAL_RJ, 4)
		meld.Fixed = false
		meld.Reload(TILE_RJ, TILE_RJ, TILE_BJ, TILE_BJ)
	} else {
		// 小王对子/单张
		if bj > 0 {
			last := len(this.melds)
			this.melds = this.melds[:last+1]
			meld := &this.melds[last]
			if bj == 2 {
				meld.Group = makeGroup(PATTERN_PAIR, 0, VAL_BJ, 2)
			} else {
				meld.Group = makeGroup(PATTERN_SINGLE, 0, VAL_BJ, 1)
			}
			meld.Fixed = false
			meld.Group.Size = 0
			meld.addTile(TILE_BJ, bj)
		}
		// 大王对子/单张
		if rj > 0 {
			last := len(this.melds)
			this.melds = this.melds[:last+1]
			meld := &this.melds[last]
			if rj == 2 {
				meld.Group = makeGroup(PATTERN_PAIR, 0, VAL_RJ, 2)
			} else {
				meld.Group = makeGroup(PATTERN_SINGLE, 0, VAL_RJ, 1)
			}
			meld.Fixed = false
			meld.Group.Size = 0
			meld.addTile(TILE_RJ, rj)
		}
	}
}

// 按照指定模式理牌，返回*未排序*分组
func (this *Splitter) Split(hand *Hand, mode int8) []Meld {
	this.load(hand)
	if this.size > 0 {
		switch mode {
		case MODE_BOMB:
			(*Splitter1)(this).split()
		case MODE_FLUSH:
			(*Splitter2)(this).split()
		case MODE_QUICK:
			(*Splitter3)(this).split()
		case MODE_SIMPLE:
			(*Splitter4)(this).split()
		default:
			panic(fmt.Errorf("mode=%d", mode))
		}
		this.paint(&this.env, this.best.hints)
	}
	this._splitJoker(hand.jokers[0], hand.jokers[1])
	return this.melds
}

func (this *Splitter) Sort(melds []Meld) {
	SortMeld(melds, this.level)
}

// s2牌值是否大于s1
func cmpSequence(seq *[13]int8, s1, s2 *Value) int8 {
	for _, i := range seq {
		n1 := s1.values[i]
		n2 := s2.values[i]
		if n2 > n1 {
			return 1
		} else if n2 < n1 {
			return -1
		}
	}
	return 0
}

// s2是否大于s1
func cmpSequence2(level int8, s1, s2 *Multiple) int8 {
	util.Assert(s1.count == s2.count)
	for i := int8(0); i < s1.count; i++ {
		v1 := s1.values[i]
		v2 := s2.values[i]
		if v1.count != v2.count {
			if v2.count > v1.count {
				return 1
			} else {
				return -1
			}
		}
		if v1.value != v2.value {
			if Follow(level, v1.value, v2.value) {
				return 1
			} else {
				return -1
			}
		}
	}
	return 0
}

// s2序列是否优于s1序列：牌值更小则优先
func (this *Splitter) _cmpSequence(s1, s2 *Value) int8 {
	if s2.count != s1.count {
		if s2.count > s1.count {
			return 1
		} else {
			return -1
		}
	}
	if s1.count > 0 {
		return cmpSequence(&this.sequence, s2, s1)
	}
	return 0
}

// 炸弹优先：对比数量，inf2是否优于inf1
func (this *Splitter) cmpBombCount1(inf1, inf2 *Stat) int8 {
	if n1, n2 := inf1.getBombCount(), inf2.getBombCount(); n1 != n2 {
		if n2 > n1 {
			return 1
		} else {
			return -1
		}
	}
	if inf2.flush.count != inf1.flush.count {
		if inf2.flush.count > inf1.flush.count {
			return 1
		} else {
			return -1
		}
	}
	return 0
}

// 炸弹优先：对比大小，inf2是否优于inf1
func (this *Splitter) cmpBombWeight1(inf1, inf2 *Stat) int8 {
	if inf2.flush.count > 0 {
		if n := cmpSequence(&_straight, &inf1.flush, &inf2.flush); n != 0 {
			return n
		}
	}
	if inf2.multiple.count > 0 {
		if n := cmpSequence2(this.level, &inf1.multiple, &inf2.multiple); n != 0 {
			return n
		}
	}
	return 0
}

// 同花顺优先：对比数量，inf2是否优于inf1
func (this *Splitter) cmpBombCount2(inf1, inf2 *Stat) int8 {
	if inf2.flush.count != inf1.flush.count {
		if inf2.flush.count > inf1.flush.count {
			return 1
		} else {
			return -1
		}
	}
	if n1, n2 := inf1.getBombCount(), inf2.getBombCount(); n1 != n2 {
		if n2 > n1 {
			return 1
		} else {
			return -1
		}
	}
	return 0
}

// 同花顺优先：对比大小，inf2是否优于inf1
func (this *Splitter) cmpBombWeight2(inf1, inf2 *Stat) int8 {
	if inf2.flush.count > 0 {
		if n := cmpSequence(&_straight, &inf1.flush, &inf2.flush); n != 0 {
			return n
		}
	}
	if inf2.multiple.count > 0 {
		if n := cmpSequence2(this.level, &inf1.multiple, &inf2.multiple); n != 0 {
			return n
		}
	}
	return 0
}

func (this *Splitter) cmpSequence(inf1, inf2 *Stat) int8 {
	// 顺子优先
	if n := this._cmpSequence(&inf1.straight, &inf2.straight); n != 0 {
		return n
	}
	// 其次连对
	if n := this._cmpSequence(&inf1.straightPair, &inf2.straightPair); n != 0 {
		return n
	}
	return 0
}

// 比较单牌/对子，inf2是否优于inf1
func (this *Splitter) cmePieces(inf1, inf2 *Stat) int8 {
	if diff := inf1.piece - inf2.piece; diff != 0 {
		// 零牌少优先
		return diff
	}
	for _, index := range this.sequence {
		diff := inf2.pieces[index] - inf1.pieces[index]
		if diff != 0 {
			// 牌值大的零牌多优先
			return diff
		}
	}
	return 0
}

// inf2是否优于inf1（炸弹最多=>同花顺最多=>单牌最少）
func (this *Splitter) win1(inf1, inf2 *Stat) bool {
	if n := this.cmpBombCount1(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	if inf2.hand != inf1.hand {
		return inf2.hand < inf1.hand
	}
	if n := this.cmpBombWeight1(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	// 顺子/连对优先取小的牌
	if n := this.cmpSequence(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	return inf2.single < inf1.single
}

// inf2是否优于inf1（同花顺最多=>炸弹最多=>单牌最少）
func (this *Splitter) win2(inf1, inf2 *Stat) bool {
	if n := this.cmpBombCount2(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	if inf2.hand != inf1.hand {
		return inf2.hand < inf1.hand
	}
	if n := this.cmpBombWeight2(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	// 顺子/连对优先取小的牌
	if n := this.cmpSequence(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	return inf2.single < inf1.single
}

// inf2是否优于inf1（单牌+手数最少=>炸弹最多=>同花顺最多）
func (this *Splitter) win3(inf1, inf2 *Stat) bool {
	if n1, n2 := inf1.single+inf1.hand, inf2.single+inf2.hand; n1 != n2 {
		return n2 < n1
	}
	if n := this.cmpBombCount1(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	if n := this.cmpBombWeight1(inf1, inf2); n > 0 {
		return true
	} else if n < 0 {
		return false
	}
	// 如果零碎牌数量不一样，则顺子/连对优先取小的牌；否则优先取大的牌
	if n := this.cmePieces(inf1, inf2); n != 0 {
		return this.cmpSequence(inf1, inf2) > 0
	} else {
		return this.cmpSequence(inf1, inf2) < 0
	}
}

func _pattern(size int8) int8 {
	util.Assert(size > 0)
	switch size {
	case 1:
		return PATTERN_SINGLE
	case 2:
		return PATTERN_PAIR
	case 3:
		return PATTERN_TRIPLE
	default:
		return PATTERN_BOMB
	}
}

// 自动出牌
func (this *Splitter) AutoPlay(hand *Hand) ([]int8, int8) {
	env := &this.env
	env.load(hand, this.level, this.laiziTile)
	// 打出单张、对子、三张中的最小单张
	for i := len(this.sequence) - 1; i >= 0; i-- {
		rank := &env.ranks[this.sequence[i]]
		if rank.total > 0 && rank.total <= 3 {
			tiles := rank.dumpMaxCount(make([]int8, 0, 1), 1)
			util.Assert(len(tiles) == 1)
			return tiles, PATTERN_SINGLE
		}
	}
	// 如果只剩大小王和癞子，则先打出癞子单张
	joker := hand.jokers[0] + hand.jokers[1]
	if env.laizi > 0 && env.laizi+joker == int8(hand.size) {
		return []int8{this.laiziTile}, PATTERN_SINGLE
	}
	// 打出小王、大王
	if joker > 0 {
		bj := hand.jokers[0]
		rj := hand.jokers[1]
		if bj == 2 && rj == 2 {
			// 如果只剩2个大王2个小王，按照天王炸弹打出
			return []int8{TILE_RJ, TILE_RJ, TILE_BJ, TILE_BJ}, PATTERN_BOMB
		}
		if bj > 0 {
			return repeat(TILE_BJ, bj), _pattern(bj)
		}
		if rj > 0 {
			return repeat(TILE_RJ, rj), _pattern(rj)
		}
	}
	// 打出最小炸弹
	index := int8(-1)
	for i := len(this.sequence) - 1; i >= 0; i-- {
		j := this.sequence[i]
		rank := &env.ranks[j]
		if rank.total == 0 {
			continue
		}
		util.Assert(rank.total >= 4)
		if index < 0 || rank.total < env.ranks[index].total {
			index = j
		}
	}
	util.Assert(index >= 0)
	rank := &env.ranks[index]
	tiles := rank.dump(make([]int8, 0, rank.total))
	return tiles, _pattern(rank.total)
}

// 牌值数量
type Value struct {
	values [13]int8 // 各牌值数量
	count  int8     // 牌型数量
}

func (this *Value) reset() {
	this.values = [13]int8{}
	this.count = 0
}

// 统计牌型中的牌
func (this *Value) stat1(meld *Meld) {
	this.count++
	for i := int8(0); i < meld.Group.Size; i++ {
		v := GetValue(meld.Tiles[i])
		this.values[v-1]++
	}
}

// 统计牌型中的牌
func (this *Value) stat3(hint *Hint, pad, size, level int8) {
	this.count++
	for i := int8(0); i < size; i++ {
		item := hint.list[i]
		if item.count > 0 {
			this.values[item.index] += item.count
		}
		if item.count < pad {
			this.values[level-1] += pad - item.count
		}
	}
}

// 统计牌型牌值
func (this *Value) stat2(value int8) {
	this.count++
	this.values[value-1]++
}

// 普通炸弹的牌值数量
type Multiple struct {
	values [6]struct { // 各牌值数量（最多6个普通炸弹）
		value int8
		count int8
	}
	count int8
}

func (this *Multiple) reset() {
	this.count = 0
}

func (this *Multiple) stat(value, count int8) {
	// 有可能把一个>=8张的普通炸弹拆分为2个炸弹
	v := &this.values[this.count]
	v.value = value
	v.count = count
	this.count++
}

func (this *Multiple) sort(level int8) {
	list := this.values[:this.count]
	sort.Slice(list, func(i, j int) bool {
		v1, v2 := list[i], list[j]
		if v1.count != v2.count {
			return v1.count > v2.count
		} else if v2.value != v1.value {
			return Follow(level, v2.value, v1.value)
		} else {
			return false
		}
	})
}

// 统计信息
type Stat struct {
	straight     Value       // 顺子统计
	straightPair Value       // 连对统计
	flush        Value       // 同花顺
	multiple     Multiple    // 普通炸弹
	pieces       [VAL_K]int8 // 零碎牌组合：牌值=>组合数量
	hand         int8        // 手数
	size         int8        // 牌数
	single       int8        // 单牌数量
	piece        int8        // 零碎牌组合的数量（单张、对子）
}

func (this *Stat) String() string {
	return fmt.Sprintf("{single=%d, hand=%d}", this.single, this.hand)
}

func (this *Stat) reset() {
	this.straight.reset()
	this.straightPair.reset()
	this.multiple.reset()
	this.flush.reset()
	this.hand = 0
	this.size = 0
	this.single = 0
	this.piece = 0
	clear(this.pieces[:])
}

func (this *Stat) getBombCount() int8 {
	return this.flush.count + this.multiple.count
}

func (this *Stat) stat(hints []Hint, level int8) {
	var pairs, triples [13]int8 // 连对（数字牌）、三张牌值
	var pairCount, tripleCount int8
	for i := range len(hints) {
		hint := &hints[i]
		group := hint.group
		this.size += group.Size
		if group.Pattern == PATTERN_BOMB {
			switch group.Minor {
			case BOMB_FLUSH:
				// 炸弹手数算-1（可以回收一手）
				this.hand--
				this.flush.stat2(group.Value)
			case BOMB_NORMAL:
				// 张数大于等于8，抵消手数更多
				if extra := group.Size - 8; extra >= 0 {
					this.hand -= extra*2 + 3
				} else {
					this.hand--
				}
				this.multiple.stat(group.Value, group.Size)
			}
		} else {
			this.hand++
			switch group.Pattern {
			case PATTERN_SINGLE:
				this.single++
				this.piece++
				this.pieces[group.Value-1]++
			case PATTERN_STRAIGHT:
				this.straight.stat3(hint, 1, 5, level)
			case PATTERN_STRAIGHTPAIR:
				this.straightPair.stat3(hint, 2, 3, level)
			case PATTERN_TRIPLE:
				triples[group.Value-1]++
				tripleCount++
			case PATTERN_PAIR:
				this.piece++
				this.pieces[group.Value-1]++
				if group.Value > VAL_K {
					break
				}
				pairs[group.Value-1]++
				pairCount++
			}
		}
	}
	// 能组成连对的三个连续对子算一手
	if pairCount >= 3 {
		// AKQ
		if pairs[IDX_A] > 0 && pairs[IDX_K] > 0 && pairs[IDX_Q] > 0 {
			pairs[IDX_A]--
			pairs[IDX_K]--
			pairs[IDX_Q]--
			pairCount -= 3
			this.hand -= 2
		}
		for i := IDX_K; i >= IDX_3 && pairCount >= 3; i-- {
			if pairs[i] > 0 && pairs[i-1] > 0 && pairs[i-2] > 0 {
				pairs[i]--
				pairs[i-1]--
				pairs[i-2]--
				pairCount -= 3
				this.hand -= 2
			}
		}
	}
	// 能组成钢板的两个连续三张算一手
	if tripleCount >= 2 {
		// AK
		if triples[IDX_A] > 0 && triples[IDX_K] > 0 {
			triples[IDX_A]--
			triples[IDX_K]--
			tripleCount -= 2
			this.hand--
		}
		for i := IDX_K; i >= IDX_2 && tripleCount >= 2; i-- {
			if triples[i] > 0 && triples[i-1] > 0 {
				triples[i]--
				triples[i-1]--
				tripleCount -= 2
				this.hand--
			}
		}
	}
	// 小于等于10的三张和对子算三带对一手
	if tripleCount > 0 && pairCount > 0 {
		if level <= VAL_10 {
			index := level - 1
			if triples[index] > 0 {
				triples[index] = 0
				tripleCount--
			}
			if pairs[index] > 0 {
				pairs[index] = 0
				pairCount--
			}
		}
		if tripleCount > 0 && pairCount > 0 {
			for i := IDX_J; i <= IDX_K; i++ {
				if pairs[i] > 0 {
					pairs[i] = 0
					pairCount--
				}
				if triples[i] > 0 {
					triples[i] = 0
					tripleCount--
				}
			}
			if tripleCount > 0 && pairCount > 0 {
				this.hand -= min(tripleCount, pairCount)
			}
		}
	}
	// 对同张炸弹进行排序，方便优先级比较
	if this.multiple.count > 1 {
		this.multiple.sort(level)
	}
}

func inRange(v, start, end int8) bool {
	return v >= start && v <= end
}

func (this *Splitter) balance1(m *Motion, laizi int8) {
	var max *Hint
	for i := 0; i < len(m.hints); i++ {
		hint := &m.hints[i]
		group := hint.group
		if group.Pattern == PATTERN_BOMB && group.Minor == BOMB_NORMAL && (max == nil || group.Follow(this.level, max.group)) {
			max = hint
		}
	}
	if max != nil {
		// 如果有普通炸弹，就把癞子加到最大的普通炸弹上
		max.group.Size += laizi
	} else {
		// 否则癞子作为单张/对子
		var hint Hint
		if laizi == 1 {
			hint.group = makeGroup(PATTERN_SINGLE, 0, this.level, 1)
		} else {
			util.Assert(laizi == 2)
			hint.group = makeGroup(PATTERN_PAIR, 0, this.level, 2)
		}
		m.hints = append(m.hints, hint)
	}
}

func (this *Splitter) balance2(m *Motion, laizi int8) {
	var (
		triples, pairs [STRAIGHT_MAXSIZE]*Hint // 三张、对子
		multiple       *Hint                   // 最大的普通炸弹
		bombs          [VAL_K + 1]int8         // 普通炸弹
	)
	for i := range len(m.hints) {
		hint := &m.hints[i]
		group := hint.group
		switch group.Pattern {
		case PATTERN_BOMB:
			if group.Minor != BOMB_NORMAL {
				break
			}
			bombs[group.Value]++
			if multiple == nil || group.Follow(this.level, multiple.group) {
				multiple = hint
			}
		case PATTERN_TRIPLE:
			index := group.Value - 1
			hint.next = triples[index]
			triples[index] = hint
		case PATTERN_PAIR:
			if group.Value <= VAL_K {
				index := group.Value - 1
				hint.next = pairs[index]
				pairs[index] = hint
			}
		}
	}
	// 如果有多余的癞子，尝试按照牌值大小的次序把三张组成炸弹
	for _, i := range this.sequence {
		hint := triples[i]
		for hint != nil {
			// 不组成重复的普通炸弹
			if bombs[hint.group.Value] == 0 {
				hint.group.Pattern = PATTERN_BOMB
				hint.group.Minor = BOMB_NORMAL
				hint.group.Size++
				laizi--
				if laizi == 0 {
					return
				}
			}
			hint = hint.next
		}
	}
	// 如果有多余的癞子，尝试按照牌值大小的次序把对子组成炸弹
	if laizi >= 2 {
		for _, i := range this.sequence {
			hint := pairs[i]
			for hint != nil && laizi >= 2 {
				// 不组成重复的普通炸弹
				if bombs[hint.group.Value] == 0 {
					hint.group.Pattern = PATTERN_BOMB
					hint.group.Minor = BOMB_NORMAL
					hint.group.Size += 2
					laizi -= 2
				}
				hint = hint.next
			}
			if laizi < 2 {
				break
			}
		}
	}
	if laizi > 0 {
		if multiple != nil {
			// 如果有普通炸弹，就把癞子加到最大的普通炸弹上
			multiple.group.Size += laizi
		} else {
			// 否则癞子作为单张/对子
			var hint Hint
			if laizi == 1 {
				hint.group = makeGroup(PATTERN_SINGLE, 0, this.level, 1)
			} else {
				hint.group = makeGroup(PATTERN_PAIR, 0, this.level, 2)
			}
			m.hints = append(m.hints, hint)
		}
		laizi = 0
	}
}

func (this *Splitter) load1(m *Motion, env *Env) {
	list := this.stack.list()
	// 不能改变栈的内容
	m.hints = m.hints[:len(list)]
	copy(m.hints, list)
	// 分配剩余癞子
	if env.laizi > 0 {
		this.balance1(m, env.laizi)
	}
	m.info.reset()
	m.info.stat(m.hints, this.level)
}

func (this *Splitter) load2(m *Motion, env *Env) {
	list := this.stack.list()
	// 不能改变栈的内容
	m.hints = m.hints[:len(list)]
	copy(m.hints, list)
	// 分配剩余癞子
	if env.laizi > 0 {
		this.balance2(m, env.laizi)
	}
	m.info.reset()
	m.info.stat(m.hints, this.level)
}

func (this *Splitter) load3(m *Motion, env *Env) {
	list := this.stack.list()
	// 不能改变栈的内容
	m.hints = m.hints[:len(list)]
	copy(m.hints, list)
	// 分配剩余癞子
	if env.laizi > 0 {
		this.balance2(m, env.laizi)
	}
}

func (this *Splitter) end1(env *Env) {
	this.load1(this.curr, env)
	if len(this.best.hints) == 0 || this.win1(&this.best.info, &this.curr.info) {
		this.best, this.curr = this.curr, this.best
	}
}

func (this *Splitter) end2(env *Env) {
	this.load1(this.curr, env)
	if len(this.best.hints) == 0 || this.win2(&this.best.info, &this.curr.info) {
		this.best, this.curr = this.curr, this.best
	}
}

func (this *Splitter) end3(env *Env) {
	this.load2(this.curr, env)
	if len(this.best.hints) == 0 || this.win3(&this.best.info, &this.curr.info) {
		this.best, this.curr = this.curr, this.best
	}
}

func (this *Splitter) end4(env *Env) {
	this.load3(this.curr, env)
	this.best, this.curr = this.curr, this.best
}

// 连续序列成员
type Seq struct {
	index int8 // rank索引
	count int8 // 牌数
}

// 是否连续序列，返回最大牌值
func isSequence(pos, size int8) (int8, bool) {
	last := pos + size - 1
	if last >= SEQUENCE_MAXSIZE {
		return 0, false
	}
	return _sequence[last] + 1, true
}

func (this *Splitter) paint(env *Env, hints []Hint) {
	// 先给同花顺上色
	this.melds = this.melds[:len(hints)]
	clear(this.melds)
	for i := 0; i < len(hints); i++ {
		hint := &this.best.hints[i]
		if hint.group.Pattern != PATTERN_BOMB || hint.group.Minor != BOMB_FLUSH {
			continue
		}
		meld := &this.melds[i]
		meld.Group = hint.group
		meld.Group.Size = 0
		suit := hint.suit - 1
		for j := len(hint.list) - 1; j >= 0; j-- {
			item := hint.list[j]
			if item.count > 0 {
				rank := &env.ranks[item.index]
				util.Assert(rank.counts[suit] > 0)
				rank.counts[suit]--
				rank.total--
				meld.addTile(MakeTile(hint.suit, rank.value), 1)
			} else {
				util.Assert(env.laizi > 0)
				meld.addTile(this.laiziTile, 1)
				env.laizi--
			}
		}
	}
	// 再给其他牌型上色
	for i := 0; i < len(hints); i++ {
		hint := &hints[i]
		group := hint.group
		if group.Pattern == PATTERN_BOMB && group.Minor == BOMB_FLUSH {
			continue
		}
		meld := &this.melds[i]
		meld.Group = group
		meld.Group.Size = 0
		switch group.Pattern {
		case PATTERN_BOMB:
			switch group.Minor {
			case BOMB_NORMAL:
				item := hint.list[0]
				rank := &env.ranks[item.index]
				rank.pop(meld, item.count)
				if short := group.Size - item.count; short > 0 {
					util.Assert(env.laizi >= short)
					env.laizi -= short
					meld.addTile(this.laiziTile, short)
				}
			case BOMB_JOKER:
				meld.Reload(TILE_RJ, TILE_RJ, TILE_BJ, TILE_BJ)
			}
		case PATTERN_STRAIGHTPAIR:
			for i := int8(2); i >= 0; i-- {
				item := hint.list[i]
				if item.count > 0 {
					rank := &env.ranks[item.index]
					rank.pop(meld, item.count)
				}
				if short := 2 - item.count; short > 0 {
					util.Assert(env.laizi >= short)
					env.laizi -= short
					meld.addTile(this.laiziTile, short)
				}
			}
		case PATTERN_STRAIGHT:
			for i := int8(4); i >= 0; i-- {
				item := hint.list[i]
				if item.count > 0 {
					rank := &env.ranks[item.index]
					rank.pop(meld, 1)
				} else {
					util.Assert(env.laizi > 0)
					env.laizi--
					meld.addTile(this.laiziTile, 1)
				}
			}
		case PATTERN_STRAIGHTTRIPLE, PATTERN_TRIPLEPLUS:
			panic(fmt.Errorf("unexepectd meld=%s", hint))
		case PATTERN_SINGLE:
			if group.Value <= VAL_K {
				item := hint.list[0]
				rank := &env.ranks[item.index]
				if item.count > 0 {
					rank.pop(meld, 1)
				} else {
					// 癞子作单张
					util.Assert(env.laizi > 0)
					env.laizi--
					meld.addTile(this.laiziTile, 1)
				}
			} else {
				switch group.Value {
				case VAL_BJ:
					meld.addTile(TILE_BJ, 1)
				case VAL_RJ:
					meld.addTile(TILE_RJ, 1)
				}
			}
		case PATTERN_PAIR:
			if group.Value <= VAL_K {
				item := hint.list[0]
				rank := &env.ranks[item.index]
				if item.count > 0 {
					rank.pop(meld, item.count)
				}
				if short := 2 - item.count; short > 0 {
					util.Assert(env.laizi >= short)
					env.laizi -= short
					meld.addTile(this.laiziTile, short)
				}
			} else {
				switch group.Value {
				case VAL_BJ:
					meld.addTile(TILE_BJ, 2)
				case VAL_RJ:
					meld.addTile(TILE_RJ, 2)
				}
			}
		case PATTERN_TRIPLE:
			item := hint.list[0]
			rank := &env.ranks[item.index]
			if item.count > 0 {
				rank.pop(meld, item.count)
			}
			if short := 3 - item.count; short > 0 {
				util.Assert(env.laizi >= short)
				env.laizi -= short
				meld.addTile(this.laiziTile, short)
			}
		}
		util.Assert(meld.Group.Size == group.Size)
	}
}

type Env struct {
	ranks [STRAIGHT_MAXSIZE]Rank
	laizi int8
}

func (this *Env) load(hand *Hand, level, laizi int8) {
	for i := range len(this.ranks) {
		rank := &this.ranks[i]
		rank.init(int8(i + 1))
	}
	for i := range 4 {
		for j, n := range hand.counts[i] {
			rank := &this.ranks[j]
			rank.counts[i] += n
			rank.total += n
		}
	}
	this.laizi = hand.Count(laizi)
	if this.laizi > 0 {
		// 删除红桃级牌
		rank := &this.ranks[level-1]
		rank.total -= rank.counts[SUIT_HEART-1]
		rank.counts[SUIT_HEART-1] = 0
	}
}

func (this *Env) isStraight(pos int8) (int8, bool) {
	end := pos + 5
	if end > SEQUENCE_MAXSIZE {
		return 0, false
	}
	var laizi int8
	if rank := &this.ranks[_sequence[pos]]; rank.total == 0 {
		// 如果起始用赖子替代，顺子的实际值可以取更大值
		if this.laizi == 0 || end+1 <= SEQUENCE_MAXSIZE {
			return 0, false
		}
		laizi++
	}
	for _, index := range _sequence[pos+1 : end] {
		if this.ranks[index].total == 0 {
			if laizi >= this.laizi {
				return 0, false
			}
			laizi++
		}
	}
	return _sequence[end-1] + 1, true
}

// 组顺子（不需要3张或以上单牌的条件）
func (this *Env) popStraight1(pos int8) ([5]Seq, bool) {
	end := pos + 5
	if end > SEQUENCE_MAXSIZE {
		return [5]Seq{}, false
	}
	var items [5]Seq
	var laizi int8
	for i, index := range _sequence[pos:end] {
		rank := &this.ranks[index]
		item := &items[i]
		item.index = index
		if rank.total > 0 {
			item.count = 1
		} else if laizi < this.laizi {
			// 如果起始用赖子替代，顺子的实际值可以取更大值
			if i == 0 && end+1 <= SEQUENCE_MAXSIZE {
				return [5]Seq{}, false
			}
			laizi++
		} else {
			return [5]Seq{}, false
		}
	}
	for _, item := range items {
		if item.count > 0 {
			this.ranks[item.index].total--
		} else {
			this.laizi--
		}
	}
	return items, true
}

// 组顺子（要有3张或以上的单牌才组顺子）
func (this *Env) popStraight2(pos int8) ([5]Seq, bool, bool) {
	end := pos + 5
	if end > SEQUENCE_MAXSIZE {
		return [5]Seq{}, false, false
	}
	var items [5]Seq
	var laizi, multiple int8
	for i, index := range _sequence[pos:end] {
		rank := &this.ranks[index]
		item := &items[i]
		item.index = index
		if rank.total > 0 {
			if rank.total > 1 {
				multiple++
			}
			item.count = 1
		} else if laizi < this.laizi {
			// 如果起始用赖子替代，顺子的实际值可以取更大值
			if i == 0 && end+1 <= SEQUENCE_MAXSIZE {
				return [5]Seq{}, false, false
			}
			laizi++
		} else {
			return [5]Seq{}, false, false
		}
	}
	// 要有3张或以上的单牌才组顺子
	if multiple > 2 {
		return [5]Seq{}, true, false
	}
	for _, item := range items {
		if item.count > 0 {
			this.ranks[item.index].total--
		} else {
			this.laizi--
		}
	}
	return items, false, true
}

func (this *Env) popFlush(pos, suit int8) ([5]Seq, bool) {
	seq := _sequence[pos : pos+5]
	// 如果起始用赖子替代，同花顺的实际值可以取更大值
	if this.ranks[seq[0]].counts[suit] == 0 && seq[4] != IDX_A {
		return [5]Seq{}, false
	}
	var items [5]Seq
	for i, index := range seq {
		rank := &this.ranks[index]
		item := &items[i]
		item.index = index
		// 因为遍历其他花色无关牌型时只扣除total没有扣除花色牌，所以这里要先检查total
		if rank.total > 0 && rank.counts[suit] > 0 {
			rank.counts[suit]--
			rank.total--
			item.count = 1
		} else if this.laizi > 0 {
			this.laizi--
		} else {
			// roll back
			for _, it := range items[:i] {
				if it.count > 0 {
					r := &this.ranks[it.index]
					r.counts[suit]++
					r.total++
				} else {
					this.laizi++
				}
			}
			return [5]Seq{}, false
		}
	}
	return items, true
}

// 增加序列牌
func (this *Env) pushFlush(list [5]Seq, suit int8) {
	for _, item := range list {
		if item.count > 0 {
			rank := &this.ranks[item.index]
			rank.counts[suit]++
			rank.total++
		} else {
			this.laizi++
		}
	}
}

func (this *Env) pushStraight(hint *Hint) {
	for _, item := range hint.list {
		if item.count > 0 {
			rank := &this.ranks[item.index]
			rank.total++
		} else {
			this.laizi++
		}
	}
}

// 删除序列牌
func (this *Env) popSeq(seq [5]Seq, size, pad int8) {
	for _, item := range seq[:size] {
		if item.count > 0 {
			util.Assert(item.count <= pad)
			rank := &this.ranks[item.index]
			util.Assert(rank.total >= item.count)
			rank.total -= item.count
		}
		laizi := pad - item.count
		if laizi > 0 {
			util.Assert(this.laizi >= laizi)
			this.laizi -= laizi
		}
	}
}

// 增加序列牌
func (this *Env) pushSeq(seq [5]Seq, size, pad int8) {
	for _, item := range seq[:size] {
		if item.count > 0 {
			this.ranks[item.index].total += item.count
		}
		laizi := pad - item.count
		if laizi > 0 {
			this.laizi += laizi
		}
	}
}

func (this *Env) findSequence(pos, size, pad int8) ([5]Seq, int8, bool) {
	var laizi int8
	var list [5]Seq
	for i, index := range _sequence[pos : pos+size] {
		item := &list[i]
		item.index = index
		rank := &this.ranks[index]
		if short := pad - rank.total; short > 0 {
			laizi += short
			if laizi > this.laizi {
				return [5]Seq{}, 0, false
			}
			item.count = rank.total
		} else {
			item.count = pad
		}
	}
	return list, laizi, true
}

func BuildMelds(melds []Meld, alloc func() *meld.Meld) {
	for i := range len(melds) {
		meld := &melds[i]
		item := alloc()
		item.Load(meld.List())
		item.Pattern = meld.Group.Pattern
		item.Type = meld.Group.Minor
		item.Value = meld.Group.Value
		item.Fixed = meld.Fixed
	}
}

// 评估理牌组合分数（金币场/经典）
func ValueMelds1(melds []Meld, level int8) int16 {
	var score int16
	const (
		SCORE_BOMB6          = 22 // 6炸及以上 22分
		SCORE_BOMB5          = 12 // 5炸 12分
		SCORE_BOMB4          = 8  // 4炸 8分
		SCORE_FLUSH          = 16 // 同花顺 16分
		SCORE_RJ             = 4  // 大王 4分
		SCORE_BJ             = 2  // 小王 2分
		SCORE_STRAIGHTPAIR   = 2  // 木板 2分
		SCORE_STRAIGHTTRIPLE = 0  // 钢板 0分
		SCORE_LEVEL          = 0  // 级牌 每张0分
	)
	var triples [13]int8
	var triple, seq int8
	for i := range len(melds) {
		group := melds[i].Group
		switch group.Pattern {
		case PATTERN_BOMB:
			switch group.Minor {
			case BOMB_FLUSH:
				// 同花顺
				score += SCORE_FLUSH
			case BOMB_NORMAL:
				if group.Size >= 6 {
					// 6炸及以上
					score += SCORE_BOMB6
				} else if group.Size >= 5 {
					// 5炸
					score += SCORE_BOMB5
				} else {
					// 4炸
					score += SCORE_BOMB4
				}
			default:
				util.Assert(group.Minor == BOMB_JOKER)
				// 天王炸 按照单张组合计算分数
				score += 2 * (SCORE_RJ + SCORE_BJ)
			}
		case PATTERN_STRAIGHTPAIR:
			// 木板
			score += SCORE_STRAIGHTPAIR
		case PATTERN_STRAIGHTTRIPLE:
			// 钢板
			score += SCORE_STRAIGHTTRIPLE
		case PATTERN_STRAIGHT:
			// 顺子：最大牌Q或以上 0分
		case PATTERN_TRIPLE:
			if group.Value == level {
				score += SCORE_LEVEL * 3
			} else {
				triples[group.Value-1]++
				triple++
			}
		case PATTERN_PAIR, PATTERN_SINGLE:
			var count int16
			var maxValue int8
			if group.Pattern == PATTERN_PAIR {
				count = 2
				maxValue = VAL_7
			} else {
				count = 1
				maxValue = VAL_10
			}
			switch {
			case group.Value == VAL_RJ:
				score += SCORE_RJ * count
			case group.Value == VAL_BJ:
				score += SCORE_BJ * count
			case group.Value == level:
				score += SCORE_LEVEL * count
			case !Gte(level, maxValue, group.Value):
				// 小于10的单张、小于7的对子 -1分
				score--
			}
		}
	}
	if triple > 1 {
		for i := IDX_A; i <= IDX_K; i++ {
			if triples[i] == 0 {
				seq = 0
				continue
			}
			seq++
			if seq < 2 {
				continue
			}
			// 钢板
			score += SCORE_STRAIGHTTRIPLE
			triples[i] = 0
			triples[i-1] = 0
			triple -= 2
			seq = 0
		}
		if triples[IDX_A] > 0 && triples[IDX_K] > 0 {
			score += SCORE_STRAIGHTTRIPLE
			triples[IDX_A] = 0
			triples[IDX_K] = 0
			triple -= 2
		}
	}
	// 小于7的三张，每份-1分
	if triple > 0 {
		for i := IDX_A; i <= IDX_K; i++ {
			if triples[i] > 0 && !Gte(level, VAL_7, int8(i+1)) {
				score--
				triple--
				if triple <= 0 {
					break
				}
			}
		}
	}
	return score
}

// 评估理牌组合分数（金币场/不洗牌）
func ValueMelds2(melds []Meld, level int8) int16 {
	var score int16
	for i := range len(melds) {
		group := melds[i].Group
		switch group.Pattern {
		case PATTERN_BOMB:
			switch group.Minor {
			case BOMB_JOKER:
				// 天王炸 50分
				score += 50
			case BOMB_NORMAL:
				if group.Size >= 8 {
					// 8炸及以上 40分
					score += 40
				} else {
					switch group.Size {
					case 7:
						// 7炸 30分
						score += 30
					case 6:
						// 6炸 20分
						score += 20
					case 5:
						// 5炸 10分
						score += 10
					default:
						// 4炸 8分
						util.Assert(group.Size >= 4)
						score += 8
					}
				}
			}
		case PATTERN_TRIPLE:
			// 小于A的三张 -5分
			if group.Value != level && group.Value != VAL_A {
				score += -5
			}
		case PATTERN_PAIR:
			// 小于A的对子 -5分
			if group.Value != level && group.Value != VAL_A {
				score += -5
			}
		case PATTERN_SINGLE:
			if Follow(level, group.Value, VAL_BJ) {
				// 小于小王的单张 -5分
				score += -5
			}
		}
	}
	return score
}
