package common

import (
	"fmt"

	"github.com/guandan-service/hand-arranger/util"
)

// 牌型
// 炸弹和以上的牌型打小牌型不需要相同牌数。炸弹以下的小牌型之间需要同牌型同张数才能打
const (
	PATTERN_NONE           = 0 // 无牌型
	PATTERN_SINGLE         = 1 // 单张：一张牌为单张，以牌点来比对大小。例如：3、7
	PATTERN_PAIR           = 2 // 对子：牌点相同的两张牌，以牌点来对比大小。例如：44、99  大小王不能作为对子打出
	PATTERN_STRAIGHTPAIR   = 3 // 木板（三连对子）：牌点连续的三个对子，以牌尾对子作牌点对比大小。例如：334455、JJQQKK。
	PATTERN_TRIPLE         = 4 // 三张：牌点相同的三张牌，以牌点来对比大小。例如：666、555
	PATTERN_STRAIGHTTRIPLE = 5 // 钢板（二连三张）：牌点连续的两个三张，以牌尾三张作牌点对比大小。例如：333 444
	PATTERN_STRAIGHT       = 6 // 顺子：牌点连续的五张的单牌，以结尾（最大一张牌）的牌点对比大小。例如：34567
	PATTERN_TRIPLEPLUS     = 7 // 三带二：牌点相同的三张牌加一个对子，以三张牌的数字比大小
	PATTERN_BOMB           = 8 // 炸弹（普通炸弹、同花顺、四鬼）
)

// 炸弹类型
const (
	BOMB_NORMAL = 1 // 同张炸弹：牌点相同的四张或四张以上牌被称为炸弹。例如：6666、888888
	BOMB_FLUSH  = 2 // 同花顺：相同花色的顺子
	BOMB_JOKER  = 3 // 四鬼：四个王
)

const (
	STRAIGHT_MAXSIZE = 13                   // 顺子序列的最大长度
	SEQUENCE_MAXSIZE = STRAIGHT_MAXSIZE + 1 // 连续序列的最大长度（A同时在队首和队尾）
)

var (
	Empty = Group{}
)

var (
	// 连续序列：A同时在队首和队尾
	_sequence [SEQUENCE_MAXSIZE]int8 // IDX_A, IDX_2, IDX_3, ..., IDX_J, IDX_Q, IDX_K, IDX_A
	// 顺子序列：A在队首
	_straight [STRAIGHT_MAXSIZE]int8 // IDX_A, IDX_K, IDX_Q, ..., IDX_3, IDX_2
)

func init() {
	for i := 0; i < STRAIGHT_MAXSIZE; i++ {
		_sequence[i] = int8(IDX_A + i)
	}
	_sequence[STRAIGHT_MAXSIZE] = IDX_A
	_straight[0] = IDX_A
	for i := 1; i < STRAIGHT_MAXSIZE; i++ {
		_straight[i] = int8(VAL_K - i)
	}
}

func IsValidPattern(p int8) bool {
	return p > PATTERN_NONE && p <= PATTERN_BOMB
}

// 是否同张（不包括炸弹）
func IsMultiple(p int8) bool {
	return p == PATTERN_SINGLE || p == PATTERN_PAIR || p == PATTERN_TRIPLE
}

// 数字点数比较（级牌还原）：v2的牌点是否大过v1
func Win(v1, v2 int8) bool {
	if v1 != v2 {
		return v2 == VAL_A || (v1 != VAL_A && v1 < v2)
	}
	return false
}

// 牌点大小：（以打10为例）牌点从大到小的排序是：大王、小王、10、A、K、Q、J、9、8、7、6、5、4、3、2
// 牌点比较：v2的牌点是否大过v1，level为主牌
func Follow(level, v1, v2 int8) bool {
	switch {
	case v1 == v2:
		return false
	case v2 == level:
		return v1 != VAL_BJ && v1 != VAL_RJ
	case v1 == level:
		return v2 == VAL_BJ || v2 == VAL_RJ
	default:
		return _points[v2] > _points[v1]
	}
}

// 牌组
type Group struct {
	Pattern int8 // 牌型
	Minor   int8 // 子牌型，只对炸弹有效。值为BOMB_XXX
	Value   int8 // 牌值，选择其中最大的牌值，四鬼炸弹的牌值固定为15
	Size    int8 // 长度
}

func (this Group) String() string {
	if this.Pattern == PATTERN_BOMB {
		return fmt.Sprintf("{pattern=%d, minor=%d, value=0x%x, size=%d}", this.Pattern, this.Minor, this.Value, this.Size)
	} else {
		return fmt.Sprintf("{pattern=%d, value=0x%x}", this.Pattern, this.Value)
	}
}

func (this *Group) Clear() {
	this.Pattern = 0
	this.Minor = 0
	this.Value = 0
	this.Size = 0
}

// 是否可以跟（大过）指定牌组oth
func (this Group) Follow(level int8, oth Group) bool {
	if this.Pattern != oth.Pattern {
		return this.Pattern == PATTERN_BOMB
	}
	if this.Pattern != PATTERN_BOMB {
		switch this.Pattern {
		case PATTERN_SINGLE, PATTERN_PAIR, PATTERN_TRIPLE, PATTERN_TRIPLEPLUS:
			return Follow(level, oth.Value, this.Value)
		default:
			// 序列牌（三连对、钢板、顺子）比较大小不算主牌
			return _points[this.Value] > _points[oth.Value]
		}
	}
	// 炸弹牌型之间的大小： 四鬼>8炸>7炸>6炸>同花顺>5炸>4炸
	if this.Minor == oth.Minor {
		switch this.Minor {
		case BOMB_JOKER:
			return false
		case BOMB_FLUSH:
			return _points[this.Value] > _points[oth.Value]
		case BOMB_NORMAL:
			if this.Size != oth.Size {
				return this.Size > oth.Size
			} else {
				return Follow(level, oth.Value, this.Value)
			}
		}
	} else {
		switch this.Minor {
		case BOMB_JOKER:
			return true
		case BOMB_FLUSH:
			return oth.Minor == BOMB_NORMAL && oth.Size <= 5
		case BOMB_NORMAL:
			return this.Size >= 6 && oth.Minor == BOMB_FLUSH
		}
	}
	return false
}

// 解析牌型，laizi为逢人配的数量
func Parse(tiles []int8, pattern, laizi, level int8) (Group, bool) {
	count := len(tiles) + int(laizi)
	switch pattern {
	case PATTERN_SINGLE:
		// 单张
		if count != 1 {
			return Empty, false
		}
		var group Group
		group.Pattern = PATTERN_SINGLE
		group.Size = 1
		group.Value = GetValue(tiles[0])
		return group, true
	case PATTERN_PAIR:
		// 对子
		if count != 2 {
			return Empty, false
		}
		value := GetValue(tiles[0])
		if len(tiles) > 1 && GetValue(tiles[1]) != value {
			return Empty, false
		}
		// 赖子不能替代大小王
		if laizi > 0 && GetSuit(tiles[0]) == SUIT_JOKER {
			return Empty, false
		}
		var group Group
		group.Pattern = PATTERN_PAIR
		group.Size = 2
		group.Value = value
		return group, true
	case PATTERN_STRAIGHTPAIR:
		// 三连对：牌点连续的三个对子
		if count != 6 {
			return Empty, false
		}
		return parseStraightPair(tiles, laizi)
	case PATTERN_TRIPLE:
		// 三张
		if count != 3 {
			return Empty, false
		}
		return parseTriple(tiles, laizi)
	case PATTERN_STRAIGHTTRIPLE:
		// 钢板：牌点连续的两个三张
		if count != 6 {
			return Empty, false
		}
		return parseStraightTriple(tiles, laizi)
	case PATTERN_STRAIGHT:
		// 顺子：牌点连续的五张的单牌
		if count != 5 {
			return Empty, false
		}
		return parseStraight(tiles, laizi)
	case PATTERN_TRIPLEPLUS:
		// 三带二：牌点相同的三张牌加一个对子
		if count != 5 {
			return Empty, false
		}
		return parseTriplePlus(tiles, level)
	case PATTERN_BOMB:
		// 炸弹
		if count < 4 {
			return Empty, false
		}
		return parseBomb(tiles, laizi)
	default:
		return Empty, false
	}
}

func _parseStraight(tiles []int8, laizi, group int8) (int8, bool) {
	var counts [STRAIGHT_MAXSIZE]int8
	for _, t := range tiles {
		value := GetValue(t)
		if value > VAL_K {
			return 0, false
		}
		counts[value-1]++
	}
	return _isStraight(counts, laizi, group)
}

func isSameSuit(tiles []int8) bool {
	suit := GetSuit(tiles[0])
	for i := 1; i < len(tiles); i++ {
		if GetSuit(tiles[i]) != suit {
			return false
		}
	}
	return true
}

// 顺子：牌点连续的五张的单牌，以结尾（最大一张牌）的牌点对比大小。例如：34567
func parseStraight(tiles []int8, laizi int8) (Group, bool) {
	// 最多两个赖子，所以不会只有一个牌点的情况
	value, ok := _parseStraight(tiles, laizi, 1)
	if !ok {
		return Empty, false
	}
	if isSameSuit(tiles) {
		// 同花顺炸弹不是顺子
		return Empty, false
	}
	var group Group
	group.Pattern = PATTERN_STRAIGHT
	group.Value = value
	group.Size = 5
	return group, true
}

// 三连对：牌点连续的三个对子，以牌尾对子作牌点对比大小。例如：334455、JJQQKK
func parseStraightPair(tiles []int8, laizi int8) (Group, bool) {
	// 最多两个赖子，所以不会只有一个牌点的情况
	value, ok := _parseStraight(tiles, laizi, 2)
	if !ok {
		return Empty, false
	}
	var group Group
	group.Pattern = PATTERN_STRAIGHTPAIR
	group.Value = value
	group.Size = 6
	return group, true
}

// 钢板：牌点连续的两个三张，以牌尾三张作牌点对比大小。例如：333 444
func parseStraightTriple(tiles []int8, laizi int8) (Group, bool) {
	// 最多两个赖子，所以不会只有一个牌点的情况
	value, ok := _parseStraight(tiles, laizi, 3)
	if !ok {
		return Empty, false
	}
	var group Group
	group.Pattern = PATTERN_STRAIGHTTRIPLE
	group.Value = value
	group.Size = 6
	return group, true
}

// 三张：牌点相同的三张牌，以牌点来对比大小。例如：666、555
func parseTriple(tiles []int8, laizi int8) (Group, bool) {
	if len(tiles)+int(laizi) != 3 {
		return Empty, false
	}
	// 赖子不能替代大小王
	if laizi > 0 && GetSuit(tiles[0]) == SUIT_JOKER {
		return Empty, false
	}
	value := GetValue(tiles[0])
	for i := 1; i < len(tiles); i++ {
		if value != GetValue(tiles[i]) {
			return Empty, false
		}
	}
	var group Group
	group.Pattern = PATTERN_TRIPLE
	group.Value = value
	group.Size = 3
	return group, true
}

// 三带二：牌点相同的三张牌加一个对子，以三张牌的数字比大小
func parseTriplePlus(tiles []int8, level int8) (Group, bool) {
	// 不需要检查同花顺，因为最多2个赖子
	var counts [VAL_RJ]int8
	var bj, rj int8
	for _, t := range tiles {
		value := GetValue(t)
		switch value {
		case VAL_BJ:
			bj++
		case VAL_RJ:
			rj++
		default:
			counts[value-1]++
		}
	}
	type P struct {
		value int8
		count int8
	}
	list := make([]P, 0, 2)
	for i, n := range counts {
		if n == 0 {
			continue
		}
		if n > 3 || len(list) == 2 {
			return Empty, false
		}
		list = append(list, P{value: int8(i + 1), count: n})
	}
	var value int8
	if bj+rj > 0 {
		// 大小王做对子
		if bj > 0 && rj > 0 {
			// 不能同时有大小王
			return Empty, false
		}
		if bj != 2 && rj != 2 {
			// 癞子不能替代大小王
			return Empty, false
		}
		if len(list) != 1 {
			// 必须有数字牌
			return Empty, false
		}
		value = list[0].value
	} else {
		if len(list) != 2 {
			// 必须刚好两个数字牌
			return Empty, false
		}
		p1 := list[0]
		p2 := list[1]
		if p1.count == 3 {
			value = p1.value
		} else if p2.count == 3 {
			value = p2.value
		} else if Follow(level, p1.value, p2.value) {
			value = p2.value
		} else {
			value = p1.value
		}
	}
	var group Group
	group.Pattern = PATTERN_TRIPLEPLUS
	group.Value = value
	group.Size = 5
	return group, true
}

// 解析炸弹：四鬼>8炸>7炸>6炸>同花顺>5炸>4炸
func parseBomb(tiles []int8, laizi int8) (Group, bool) {
	count := int8(len(tiles)) + laizi
	var group Group
	group.Size = count
	group.Pattern = PATTERN_BOMB
	suit, value := SplitTile(tiles[0])
	switch count {
	case 4:
		// 四鬼
		if suit == SUIT_JOKER {
			// 赖子不能替代大小王
			if laizi > 0 {
				return Empty, false
			}
			for i := 1; i < len(tiles); i++ {
				if GetSuit(tiles[i]) != SUIT_JOKER {
					return Empty, false
				}
			}
			group.Minor = BOMB_JOKER
			group.Value = VAL_RJ
			return group, true
		}
	case 5:
		// 同花顺
		if suit == SUIT_JOKER {
			return Empty, false
		}
		var diffs, diffv int8
		var counts [STRAIGHT_MAXSIZE]int8
		counts[value-1]++
		for i := 1; i < len(tiles); i++ {
			s, v := SplitTile(tiles[i])
			if s == SUIT_JOKER {
				return Empty, false
			}
			counts[v-1]++
			if s != suit {
				diffs++
			}
			if v != value {
				diffv++
			}
		}
		if diffs == 0 && diffv == int8(len(tiles)-1) {
			v, ok := _isStraight(counts, laizi, 1)
			if ok {
				// 同花顺
				group.Minor = BOMB_FLUSH
				group.Value = v
				return group, true
			}
		}
		if diffv == 0 {
			// 5炸
			group.Minor = BOMB_NORMAL
			group.Value = value
			return group, true
		}
		// 不是炸弹
		return Empty, false
	default:
		if suit == SUIT_JOKER {
			return Empty, false
		}
	}
	// X炸
	for i := 1; i < len(tiles); i++ {
		if value != GetValue(tiles[i]) {
			return Empty, false
		}
	}
	group.Minor = BOMB_NORMAL
	group.Value = value
	return group, true
}

// 是否连续序列，返回最大的牌点（已保证牌数是group的倍数）
func _isStraight(counts [STRAIGHT_MAXSIZE]int8, laizi, group int8) (int8, bool) {
	// 先在2-K找到有值的区间
	min := int8(IDX_2)
	for min <= IDX_K {
		if counts[min] > 0 {
			break
		}
		min++
	}
	// 如果2-K区间没有牌，即使有A和赖子也不是连续序列
	if min > IDX_K {
		return 0, false
	}
	max := int8(IDX_K)
	for max > min {
		if counts[max] > 0 {
			break
		}
		max--
	}
	// KA234不会判断为合法，因为234-K之间有空挡
	for i := min; i <= max; i++ {
		short := group - counts[i]
		if short == 0 {
			continue
		}
		if short < 0 || laizi < short {
			return 0, false
		}
		laizi -= short
	}
	if a := counts[IDX_A]; a > 0 {
		// 必须有A
		short := group - a
		if short < 0 {
			return 0, false
		}
		if short > 0 {
			if laizi < short {
				return 0, false
			}
			laizi -= short
		}
		// QKA
		if (IDX_K-max)*group <= laizi {
			return VAL_A, true
		}
		// A23
		short = (min - IDX_2) * group
		if short > 0 {
			if short > laizi {
				return 0, false
			}
			laizi -= short
		}
		if laizi > 0 {
			max += laizi / group
		}
	} else if laizi > 0 {
		short := (IDX_K - max + 1) * group
		if short <= laizi {
			return VAL_A, true
		}
		max += laizi / group
	}
	return max + 1, true
}

// 升级（不能跳过ending指定的级别）
func LevelUp1(level, count, ending int8) int8 {
	util.Assert(level != ending)
	newLevel := level + count
	if ending == VAL_A {
		if newLevel > VAL_K {
			return VAL_A
		} else {
			return newLevel
		}
	} else {
		if newLevel > ending {
			return ending
		} else {
			return newLevel
		}
	}
}

// 升级（在VAL_2-VAL_K,VAL_A间循环）
func LevelUp2(level, count int8) int8 {
	return (level+count-1)%VAL_K + 1
}

// 指定牌是否可以还贡（还贡的牌不能大于10）
func CanReturn(level, tile int8) bool {
	return !Gte(level, VAL_J, GetValue(tile))
}

// v2是否大于等于v1（v1必须是数字牌）
func Gte(level, v1, v2 int8) bool {
	util.Assert(v1 >= VAL_A && v1 <= VAL_K)
	if v2 == VAL_BJ || v2 == VAL_RJ || v2 == VAL_A || v2 == level {
		return true
	}
	util.Assert(v2 >= VAL_A && v2 <= VAL_K)
	return v1 >= VAL_2 && v2 >= v1
}

// 炸弹倍数（不洗牌模式）
func FanBomb(group Group) int {
	util.Assert(group.Pattern == PATTERN_BOMB)
	switch group.Minor {
	case BOMB_JOKER:
		// 天王炸
		return 10
	case BOMB_FLUSH:
		// 同花顺
		return 3
	default:
		util.Assert(group.Minor == BOMB_NORMAL)
		if group.Size >= 6 {
			// 6炸及以上
			return 5
		} else if group.Size == 5 {
			// 5炸
			return 2
		} else {
			// 4炸
			return 1
		}
	}
}
