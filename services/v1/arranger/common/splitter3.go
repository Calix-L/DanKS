package common

import (
	"github.com/guandan-service/hand-arranger/util"
)

// 理牌：整牌优先
type Splitter3 Splitter

func (this *Splitter3) split() {
	this._split1(&this.env, 0)
}

// 第一次遍历：同花顺（确保尽量使用原生牌）、普通炸弹和连对
// pos 当前位置
func (this *Splitter3) _split1(env *Env, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		this._split2(env, 0, 0)
		return
	}
	// 同花顺
	if value, ok := env.isStraight(pos); ok {
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
	// 连对：10以上（包括A和级牌）不组连对（钢板都不组）
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
				this._split1(env, pos)
				this.stack.pop()
				env.pushSeq(seq, 3, 2)
			}
		}
	}
	// 拆分同张炸弹（当前牌不能全用癞子替代，至少需要一张）
	next := pos + 1
	if total := rank.total; total > 0 && total+env.laizi >= 4 {
		if total < 4 {
			// 赖子组成炸弹
			hint := this.stack.alloc()
			hint.list[0] = Seq{
				index: index,
				count: total,
			}
			laizi := 4 - total
			hint.group = makeGroup(PATTERN_BOMB, BOMB_NORMAL, rank.value, total+laizi)
			rank.total = 0
			env.laizi -= laizi
			this._split1(env, next)
			this.stack.pop()
			rank.total = total
			env.laizi += laizi
		} else {
			// 把炸弹4张之外的拆出来用于组成顺子、连对
			for n := int8(4); n <= total; n++ {
				rank.total -= n
				hint := this.stack.alloc()
				hint.list[0] = Seq{
					index: index,
					count: n,
				}
				hint.group = makeGroup(PATTERN_BOMB, BOMB_NORMAL, rank.value, n)
				this._split1(env, next)
				this.stack.pop()
				rank.total += n
			}
		}
	}
	// 不组炸弹，放到第二次遍历时组顺子
	this._split1(env, next)
}

// 第二次遍历：顺子（需要3张及以上单牌，因为可能要回头重新遍历，所以不能和单张、对子和三张的处理放在一起）
func (this *Splitter3) _split2(env *Env, flag int16, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		this._split3(env, 0)
		return
	}
	next := pos + 1
	// 顺子（需要3张或以上单牌的条件）（同花顺已经在第一次遍历中消耗掉）
	seq, multiple, ok := env.popStraight2(pos)
	if ok {
		hint := this.stack.alloc()
		hint.group = makeGroup(PATTERN_STRAIGHT, 0, seq[4].index+1, 5)
		hint.list = seq
		if flag == 0 {
			// 不需要在当前位置再检查顺子，因为如果有顺子也因为单张不足不满足条件
			this._split2(env, 0, next)
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
			this._split2(env, 0, next)
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
	this._split2(env, flag, next)
}

// 第三次遍历：单张、对子、三张
func (this *Splitter3) _split3(env *Env, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		(*Splitter)(this).end3(env)
		return
	}
	index := _sequence[pos]
	rank := &env.ranks[index]
	next := pos + 1
	if rank.total == 0 {
		this._split3(env, next)
		return
	}
	// 单张、对子、三张
	if rank.total >= 4 {
		// 不能拆分为多个普通炸弹，直接废弃遍历结果
		return
	}
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
		util.Assert(false)
	}
	count := rank.total
	rank.total = 0
	this._split3(env, next)
	this.stack.pop()
	rank.total = count
}
