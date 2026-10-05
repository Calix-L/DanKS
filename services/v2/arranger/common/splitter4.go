package common

// 理牌：炸弹优先
type Splitter4 Splitter

func (this *Splitter4) split() {
	this._split(&this.env, 0)
}

func (this *Splitter4) _split(env *Env, pos int8) {
	if pos >= SEQUENCE_MAXSIZE {
		(*Splitter)(this).end4(env)
		return
	}
	index := _sequence[pos]
	rank := &env.ranks[index]
	next := pos + 1
	if rank.total > 0 {
		hint := this.stack.alloc()
		hint.list[0] = Seq{
			index: index,
			count: rank.total,
		}
		if rank.total >= 4 {
			hint.group = makeGroup(PATTERN_BOMB, BOMB_NORMAL, rank.value, rank.total)
		} else {
			switch rank.total {
			case 3:
				hint.group = makeGroup(PATTERN_TRIPLE, 0, rank.value, rank.total)
			case 2:
				hint.group = makeGroup(PATTERN_PAIR, 0, rank.value, rank.total)
			default:
				hint.group = makeGroup(PATTERN_SINGLE, 0, rank.value, rank.total)
			}
		}
		rank.total = 0
	}
	this._split(env, next)
}
