package common

import (
	"fmt"

	"github.com/guandan-service/hand-arranger/util"
)

// 理牌：炸弹优先
type Splitter1 Splitter

func (this *Splitter1) split() {
	this._split1(&this.env, 0)
}

// 四次遍历：单张、对子、三张
func (this *Splitter1) _split4(env *Env, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		(*Splitter)(this).end1(env)
		return
	}
	index := _sequence[pos]
	rank := &env.ranks[index]
	// 拆分三张/对子/单张（当前牌不能全用癞子替代，至少需要一张）
	if pos == 0 || rank.total == 0 {
		// 如果是A，因为还要检查32A序列，所以三张/对子/单张放到最后处理。
		this._split4(env, pos+1)
		return
	}
	util.Assert(rank.total < 4)
	hint := this.stack.alloc()
	hint.list[0] = Seq{
		index: index,
		count: rank.total,
	}
	switch rank.total {
	case 1:
		// 单张
		hint.group = makeGroup(PATTERN_SINGLE, 0, rank.value, 1)
	case 2:
		// 对子
		hint.group = makeGroup(PATTERN_PAIR, 0, rank.value, 2)
	case 3:
		// 三张
		hint.group = makeGroup(PATTERN_TRIPLE, 0, rank.value, 3)
	default:
		panic(fmt.Errorf("total=%d", rank.total))
	}
	count := rank.total
	rank.total = 0
	this._split4(env, pos+1)
	this.stack.pop()
	rank.total = count
}

// 三次遍历：顺子（要有3张或以上的单牌才组顺子）
func (this *Splitter1) _split3(env *Env, flag int16, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		this._split4(env, 0)
		return
	}
	next := pos + 1
	// 顺子
	seq, multiple, ok := env.popStraight2(pos)
	if ok {
		hint := this.stack.alloc()
		hint.group = makeGroup(PATTERN_STRAIGHT, 0, seq[4].index+1, 5)
		hint.list = seq
		if flag == 0 {
			// 不需要在当前位置再检查顺子，因为如果有顺子也因为单张不足不满足条件
			this._split3(env, 0, next)
		} else {
			// 检查之前因为单张条件不满足的顺子
			var head *Hint
			last := pos
			for i, m := pos-1, flag; i >= 0 && m != 0; i-- {
				mask := int16(1) << i
				if m&mask == 0 {
					continue
				}
				if last-i > 4 && (last != IDX_10 || i != IDX_A) {
					// 如果两个顺子间距过大不会相互影响，则不需要再检查
					break
				}
				m &^= mask
				seq, _, ok = env.popStraight2(i)
				if ok {
					item := this.stack.alloc()
					item.group = makeGroup(PATTERN_STRAIGHT, 0, seq[4].index+1, 5)
					item.list = seq
					item.next = head
					head = item
					last = i
				}
			}
			this._split3(env, 0, next)
			for head != nil {
				this.stack.pop()
				env.pushStraight(head)
				head = head.next
			}
		}
		this.stack.pop()
		env.pushStraight(hint)
	} else if multiple {
		// 因为单张不足导致不能组顺子，标记需要重新遍历的位置
		flag |= 1 << pos
	}
	this._split3(env, flag, next)
}

// 二次遍历：连对（木板）
func (this *Splitter1) _split2(env *Env, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		this._split3(env, 0, 0)
		return
	}
	// 连对
	index := _sequence[pos]
	rank := &env.ranks[index]
	if rank.total+env.laizi >= 2 {
		if value, ok := isSequence(pos, 3); ok && inRange(value, VAL_4, VAL_10) && !inRange(this.level, value-2, value) {
			seq, _, ok := env.findSequence(pos, 3, 2)
			if ok {
				hint := this.stack.alloc()
				hint.group = makeGroup(PATTERN_STRAIGHTPAIR, 0, value, 6)
				hint.list = seq
				env.popSeq(seq, 3, 2)
				this._split2(env, pos)
				this.stack.pop()
				env.pushSeq(seq, 3, 2)
			}
		}
	}
	this._split2(env, pos+1)
}

// 第一次遍历：普通炸弹和同花顺
func (this *Splitter1) _split1(env *Env, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		// 如果炸弹数更少就不需要再遍历
		if this.stack.size >= int(this.best.info.getBombCount()) {
			this._split2(env, 0)
		}
		return
	}
	if value, ok := env.isStraight(pos); ok {
		// 同花顺
		for i := range int8(4) {
			seq, ok := env.popFlush(pos, i)
			if ok {
				hint := this.stack.alloc()
				hint.group = makeGroup(PATTERN_BOMB, BOMB_FLUSH, value, 5)
				hint.list = seq
				hint.suit = i + 1
				this._split1(env, pos)
				this.stack.pop()
				env.pushFlush(seq, i)
			}
		}
	}
	index := _sequence[pos]
	rank := &env.ranks[index]
	next := pos + 1
	// 拆分同张炸弹（当前牌不能全用癞子替代，至少需要一张）
	if pos == 0 || rank.total+env.laizi < 4 {
		// 如果是A，因为还要检查32A序列，所以炸弹放到最后处理。
		this._split1(env, next)
		return
	}
	// 同张炸弹（遍历出尽可能多的炸弹，如果数量不够就用癞子凑成4张）
	if total := rank.total; total <= 4 {
		hint := this.stack.alloc()
		hint.list[0] = Seq{
			index: index,
			count: rank.total,
		}
		laizi := 4 - total
		hint.group = makeGroup(PATTERN_BOMB, BOMB_NORMAL, rank.value, total+laizi)
		rank.total = 0
		env.laizi -= laizi
		this._split1(env, next)
		this.stack.pop()
		rank.total = total
		env.laizi += laizi
		// 尝试把癞子分配到其他牌上
		if laizi > 0 {
			this._split1(env, next)
		}
	} else {
		// 把炸弹4张之外的拆出来用于组成顺子、连对
		// TODO: 性能优化，先检查是否有顺子、连对
		for n := int8(4); n <= total; n++ {
			rank.total -= n
			hint := this.stack.alloc()
			hint.list[0] = Seq{
				index: index,
				count: n,
			}
			hint.group = makeGroup(PATTERN_BOMB, BOMB_NORMAL, rank.value, n)
			// 如果张数大于等于8，可能拆成多个炸弹，需要再遍历一次当前牌
			this._split1(env, pos)
			this.stack.pop()
			rank.total += n
		}
	}
}
