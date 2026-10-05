const EXACT_TEXT = new Map([
  ["AI 暂时不可用", "AI temporarily unavailable"],
  ["牌局无法继续", "This game cannot continue"],
  ["重试 AI", "Retry AI"],
  ["正在重试…", "Retrying…"],
  ["返回入口", "Back to entry"],
  ["牌局已保留，重试后从当前回合继续，不会更换机器人。", "Your game is kept. Retry this turn with the same AI."],
  ["本局无法继续，请返回入口创建新桌。", "This game cannot continue. Return to entry and create a new table."],
  ["牌局已暂停，可重试 AI 或返回入口。", "Game paused. Retry AI or return to entry."],
  ["尚未恢复，请检查 AI 服务后再次重试。", "Not recovered yet. Check the AI service and retry."],
  ["此桌已不存在，请创建新桌。", "This table no longer exists. Create a new one."],
  ["当前会话已失效，请点击继续对局重新进入。", "This session is no longer valid. Select Resume game to re-enter."],
  ["正在加载牌桌…", "Loading table…"],
  ["正在恢复对局…", "Restoring game…"],
  ["牌桌未能加载，请检查网络后重新加载。", "The table could not load. Check your connection and reload."],
  ["重新加载", "Reload"],
  ["未能恢复对局。请检查网络，或重新输入桌号继续。", "Could not restore the game. Check your connection, or enter the table number again."],
  ["单人挑战", "Solo challenge"],
  ["AI 队友", "AI partner"],
  ["左侧对手", "Left opponent"],
  ["右侧对手", "Right opponent"],
  ["开始对局", "Start game"],
  ["准备", "Ready"],
  ["工具", "Tools"],
  ["头游", "1st"],
  ["二游", "2nd"],
  ["三游", "3rd"],
  ["末游", "4th"],
  ["你与 AI 队友组队，对战两位 AI。", "Team up with an AI partner against two AIs."],
  ["3 位 AI 已准备，点击开始发牌。", "All three AIs are ready. Start to deal."],
  ["等待其他玩家准备，你可以先准备。", "Waiting for other players. You can get ready now."],
  ["正在准备…", "Getting ready…"],
  ["正在取消…", "Cancelling…"],
  ["正在确认准备状态…", "Confirming readiness…"],
  ["已准备，等待发牌。", "Ready. Waiting for the deal."],
  ["连接恢复后即可准备。", "Reconnect to get ready."],
  ["准备未成功，请重试。", "Could not get ready. Try again."],
  ["操控设置", "Controls"],
  ["关闭操控设置", "Close controls"],
  ["关闭出牌记录", "Close trick history"],
  ["关闭", "Close"],
  ["仅改变按钮大小和位置，不影响牌局。", "Changes button size and position only, not the game."],
  ["按钮大小", "Button size"],
  ["自动适配", "Automatic"],
  ["标准按钮", "Standard buttons"],
  ["大按钮", "Large buttons"],
  ["自动适配会在手机横屏时使用大按钮。", "Automatic uses large buttons on landscape phones."],
  ["操作位置", "Button position"],
  ["右手操作", "Right-handed"],
  ["左手操作", "Left-handed"],
  ["上一轮出牌", "Last trick"],
  ["牌桌工具", "Table tools"],
  ["从先手到本轮结束，仅展示已公开的出牌和不出。", "From the opening play to the trick end. Public plays and passes only."],
  ["本局还没有完成的一轮出牌。", "No trick has finished in this round yet."],
  ["当前服务未提供上一轮记录。", "Last-trick history is unavailable on this service."],
  ["这一轮记录不完整，仅展示已确认的公开动作。", "This trick is incomplete. Confirmed public actions only."],
  ["选牌", "Select"],
  ["全屏", "Fullscreen"],
  ["退出全屏", "Exit fullscreen"],
  ["当前浏览器不支持全屏，请横屏游玩。", "Fullscreen is unavailable. Play in landscape."],
  ["未能进入全屏，可稍后重试。", "Could not enter fullscreen. Try again later."],
  ["展开叠牌", "Spread stacks"],
  ["展开选牌", "Expanded selection"],
  ["关闭展开选牌", "Close expanded selection"],
  ["完成选牌", "Done selecting"],
  ["完成后回到牌桌出牌", "Return to the table to play"],
  ["整组选择", "Select group"],
  ["收紧叠牌", "Compact stacks"],
  ["出牌中", "Playing"],
  ["核对出牌", "Check play"],
  ["请求超时，尚未收到服务器确认。", "Request timed out. Server confirmation has not arrived."],
  ["浏览器无法保存对局信息。本页仍可游玩，刷新或关闭后可能无法续局。", "The browser cannot save this session. You can keep playing here, but may not be able to resume after reloading or closing."],
  ["继续下一局", "Next round"],
  ["正在开始下一局…", "Starting next round…"],
  ["当前只能不出", "Only passing is available"],
  ["选牌已更新，请重新选择", "Cards changed; select again"],
  ["出牌未成功，选牌已保留，可重新提交。", "Play was rejected. Your selection is kept; try again."],
  ["桌面已变化，正在更新，请重新确认选牌。", "The table changed. Refreshing; check your selection."],
  ["未收到出牌确认，正在核对桌面…", "No play confirmation received. Checking the table…"],
  ["未收到出牌确认，恢复连接后继续。", "No play confirmation received. Reconnect to continue."],
  ["未收到出牌确认，可点击「核对出牌」。", "Play confirmation missing. Select Check play."],
  ["当前压牌", "Current target"],
  ["自由出牌", "Lead any legal hand"],
  ["请选择合法牌型", "Choose any legal hand"],
  ["同组", "Same group"],
  ["提示音：关", "Sound: off"],
  ["提示音：开", "Sound: on"],
  ["只在确认出牌和轮到你时播放提示音", "Sound on confirmed plays and your turn only"],
  ["大王", "BIG"],
  ["小王", "SMALL"],
  ["单张", "Single"],
  ["对子", "Pair"],
  ["三张", "Triple"],
  ["顺子", "Straight"],
  ["三带二", "Full house"],
  ["三连对", "Three consecutive pairs"],
  ["钢板", "Two consecutive triples"],
  ["同花顺", "Straight flush"],
  ["炸弹", "Bomb"],
  ["天王炸", "Joker bomb"],
  ["同点叠牌", "By rank"],
  ["横屏体验更好", "Best played in landscape"],
  ["牌型分组", "Grouped layout"],
  ["选牌工具", "Selection tools"],
  ["队友 · AI", "Partner · AI"],
  ["对手 · AI", "Opponent · AI"],
  ["滑选", "Swipe select"],
  ["清空", "Clear"],
  ["点选手牌，也可整组选择", "Tap cards or select a group"],
  ["当前组合不可出，请调整选牌", "Invalid selection; adjust cards"],
  ["可以出牌，提交时选择牌型", "Playable; choose type on submit"],
  ["当前组合可以出牌", "Ready to play"],
  ["正在确认出牌…", "Confirming play…"],
  ["正在重新分组…", "Regrouping…"],
  ["连接恢复后可继续选牌", "Reconnect to select cards"],
  ["轮到你后可选牌", "Select cards on your turn"],
  ["滑动选牌；再次点击滑选可恢复浏览", "Swipe to select; toggle off to scroll"],
  ["DanKS · GuanDan（掼蛋）", "DanKS · GuanDan"],
  ["单人挑战 DanKS", "Challenge DanKS"],
  ["DanKS 是 CardKS 面向 GuanDan（掼蛋）的游戏 AI；你固定坐在 1 号位，其余三个座位全部由 Bot 接管。", "DanKS is a CardKS GuanDan AI instance. You sit at Seat 1 and the other three seats are Bots."],
  ["DanKS，CardKS 的 GuanDan（掼蛋）AI，4 人对局", "DanKS, a CardKS GuanDan AI, 4-player game"],
  ["创建单人桌", "Create Solo Table"],
  ["桌号继续", "Resume by Table No."],
  ["继续此桌", "Resume Table"],
  ["上次对局", "Recent Game"],
  ["继续对局", "Resume Game"],
  ["正在恢复…", "Restoring…"],
  ["返回原座位继续", "Resume your seat"],
  ["已回到上次对局。", "Previous game restored."],
  ["恢复失败", "Restore failed"],
  ["创建单人桌，或输入原桌号继续未完成的大局。", "Create a solo table, or resume an unfinished match with its table number."],
  ["此设备会记住你的座位，返回原桌即可继续。", "This device remembers your seat so you can return to the same table."],
  ["桌号", "Table No."],
  ["输入桌号", "Enter table number"],
  ["正在加入…", "Resuming…"],
  ["已恢复桌子。", "Table resumed."],
  ["选择游戏", "Choose Game"],
  ["掼蛋", "GuanDan"],
  ["斗地主", "DouDizhu"],
  ["4 人对局", "4-player game"],
  ["3 人对局", "3-player game"],
  ["昵称", "Nickname"],
  ["例如：北京牌友", "e.g. Player One"],
  ["例如：南京牌友", "e.g. Player One"],
  ["未连接", "Disconnected"],
  ["人类对局桌面", "Human Game Table"],
  ["平台页面", "Page Navigation"],
  ["加入方式", "Entry Mode"],
  ["本桌固定规则", "Fixed Table Rules"],
  ["双方级数和当前级牌", "Team Levels and Current Level"],
  ["我的手牌", "My Hand"],
  ["我的玩家信息", "My Player Information"],
  ["我的本回合剩余秒数", "My Remaining Turn Time"],
  ["掼蛋理牌方式", "GuanDan Hand Layout"],
  ["按理牌分组显示的我的手牌", "My Hand Grouped by Layout"],
  ["左右滑动查看手牌", "Swipe to see more cards"],
  ["斗地主公开对局状态", "Public DouDizhu Game State"],
  ["斗地主阶段操作", "DouDizhu Phase Actions"],
  ["公开的三张地主底牌", "Three Revealed Landlord Bottom Cards"],
  ["关闭规则说明", "Close Rules"],
  ["实时已连接", "Live"],
  ["正在恢复桌面", "Restoring table"],
  ["正在连接实时桌面", "Connecting"],
  ["正在验证实时桌面", "Verifying connection"],
  ["连接中断，正在重连", "Connection lost. Reconnecting"],
  ["需要重新加入", "Session expired"],
  ["大厅", "Lobby"],
  ["大厅等待", "Lobby"],
  ["等待准备", "Waiting for Ready"],
  ["正在发牌", "Dealing"],
  ["出牌阶段", "Playing"],
  ["进贡阶段", "Tribute"],
  ["还贡阶段", "Return Tribute"],
  ["叫地主", "Bidding"],
  ["加倍阶段", "Doubling"],
  ["农民加倍", "Farmer Double"],
  ["地主再加倍", "Landlord Redouble"],
  ["结算中", "Settling"],
  ["对局结束", "Round Complete"],
  ["玩家", "Player"],
  ["观众", "Spectator"],
  ["规则说明", "Rules"],
  ["复制桌号", "Copy"],
  ["最近回放", "Latest Replay"],
  ["不进贡", "No Tribute"],
  ["随机发牌", "Random Deal"],
  ["不托管", "No Autoplay"],
  ["观战开", "Spectating On"],
  ["理牌开", "Hand Sorting On"],
  ["段位关", "Ranking Off"],
  ["双方独立升级", "Independent Team Levels"],
  ["局数无限", "Unlimited Rounds"],
  ["我方", "Us"],
  ["对方", "Them"],
  ["客户端 PDF 规则", "Client PDF Rules"],
  ["掼蛋规则", "GuanDan Rules"],
  ["体育局规则", "Sports Bureau Rules"],
  ["本桌合法动作按客户端 PDF 规则判定", "Legal actions use Client PDF Rules"],
  ["本桌合法动作按掼蛋规则判定", "Legal actions use GuanDan Rules"],
  ["本桌合法动作按体育局规则判定", "Legal actions use Sports Bureau Rules"],
  ["Bot · 规则测试", "Bot · Rule Engine"],
  ["点击准备", "Ready"],
  ["已准备 · 点击取消", "Ready · Click to Cancel"],
  ["点击准备后开始单人挑战；其余三个座位均由 Bot 接管。", "Click Ready to start. The other three seats are Bots."],
  ["点击准备后开始单人挑战；其余两个座位均由 Bot 接管。", "Click Ready to start. The other two seats are Bots."],
  ["点击准备后发牌；Bot 默认准备，20 秒倒计时不会替你出牌。", "Click Ready to deal. Bots are ready by default, and the 20-second timer never autoplays for you."],
  ["玩家准备", "Player Ready"],
  ["你 · 人类玩家", "You · Human"],
  ["人类玩家", "Human"],
  ["人机玩家", "Bot"],
  ["人机 · 默认准备", "Bot · Ready"],
  ["地主 · 人机 · 默认准备", "Landlord · Bot · Ready"],
  ["农民 · 人机 · 默认准备", "Farmer · Bot · Ready"],
  ["你 · 地主 · 人类玩家", "You · Landlord · Human"],
  ["你 · 农民 · 人类玩家", "You · Farmer · Human"],
  ["对手 · 人机 · 默认准备", "Opponent · Bot · Ready"],
  ["队友 · 人机 · 默认准备", "Partner · Bot · Ready"],
  ["双数位队 · 人机", "Even-seat Team · Bot"],
  ["单数位队 · 人机", "Odd-seat Team · Bot"],
  ["已准备", "Ready"],
  ["未准备", "Not Ready"],
  ["理牌方式", "Hand Layout"],
  ["理牌", "Layout"],
  ["选择方式", "Choose Layout"],
  ["炸弹优先", "Bombs First"],
  ["同花顺优先", "Straight Flush"],
  ["整牌优先", "Made Hands"],
  ["简单理牌", "Simple Sort"],
  ["点数自动分组", "Auto-group by Rank"],
  ["只调整显示分组，不改变规则手牌或合法出牌。", "Changes display groups only; rules and legal actions are unchanged."],
  ["按 3 到大王的点数自动分组，只改变展示顺序。", "Groups cards from 3 to Big Joker; display order only."],
  ["正在重新分组；已选牌保持不变…", "Regrouping; selected cards are preserved…"],
  ["可预选手牌，轮到你时校验", "Preselect cards · Checked on your turn"],
  ["重复点击当前方式切换方案", "Tap the active mode again to cycle layouts"],
  ["理牌服务暂不可用，当前按安全回退顺序显示；规则手牌未改变。", "Hand layout is unavailable; showing the safe fallback order."],
  ["等待其他玩家", "Waiting…"],
  ["等待下一个决策", "Waiting…"],
  ["不出", "Pass"],
  ["出牌", "Play"],
  ["本轮你先出：必须选择合法牌型，不能不出", "Lead: choose a legal play."],
  ["轮到你了：选牌后点击出牌，也可以不出", "Your turn: choose cards, then Play or Pass."],
  ["轮到你：可出牌或不出", "Your turn · Play or Pass"],
  ["你先出：请选择牌", "Your lead · Select cards"],
  ["已提交，等待服务器确认…", "Submitted. Waiting for the server…"],
  ["正在创建…", "Creating…"],
  ["桌子已创建，请准备。", "Table created. Click Ready."],
  ["创建并入座", "Create and Sit"],
  ["服务器未返回完整的入桌信息。", "The server returned incomplete table information."],
  ["本局已结束；所有人类玩家准备后开始下一局", "Round complete. Click Ready to start the next round."],
  ["本局结果", "Round Result"],
  ["对局已结束", "Round Complete"],
  ["完赛名次", "Finish Order"],
  ["逐席得分", "Seat Scores"],
  ["级数结算", "Level Settlement"],
  ["双上", "One-Two Finish"],
  ["一三游", "First-Third Finish"],
  ["一四游", "First-Last Finish"],
  ["下局级牌", "Next Round Level"],
  ["准备下一局", "Ready for Next Round"],
  ["已准备 · 等待其他玩家", "Ready · Waiting"],
  ["同一组牌有多种合法语义", "These cards have multiple legal meanings"],
  ["请选择本次出牌", "Choose the intended play"],
  ["取消", "Cancel"],
  ["确认", "Confirm"],
  ["不叫", "No Bid"],
  ["地主", "Landlord"],
  ["农民", "Farmer"],
  ["最高叫分", "Highest Bid"],
  ["公共倍数", "Multiplier"],
  ["炸弹 / 火箭", "Bombs / Rockets"],
  ["不加倍", "No Double"],
  ["加倍", "Double"],
  ["不再加倍", "No Redouble"],
  ["再加倍", "Redouble"],
  ["轮到你选择是否加倍", "Choose whether to double"],
  ["轮到你选择是否再加倍", "Choose whether to redouble"],
  ["请选择本阶段动作", "Choose an action"],
  ["等待叫地主", "Waiting for bids"],
  ["待定", "Pending"],
  ["地主底牌", "Landlord Bottom Cards"],
  ["连接中断，桌面操作已冻结；恢复连接后可继续操作。", "Connection lost. Table actions are paused until reconnection."],
  ["冠军权重与运行契约已校验", "Champion weights and runtime contract verified"],
  ["策略已加载，但冠军身份未通过校验", "Policy loaded, but champion identity is unverified"],
  ["策略未就绪", "Policy is not ready"],
  ["本平台掼蛋桌固定配置", "Fixed GuanDan Table Rules"],
  ["本平台斗地主桌固定配置", "Fixed DouDizhu Table Rules"],
  ["掼蛋玩法与客户端说明", "GuanDan Rules and Client Guide"],
  ["斗地主玩法与客户端说明", "DouDizhu Rules and Client Guide"],
  ["对局流程", "Game Flow"],
  ["基本牌型", "Hand Types"],
  ["炸弹大小", "Bomb Ranking"],
  ["跨局升级", "Level Progression"],
  ["发牌与叫地主", "Deal and Bidding"],
  ["加倍流程", "Doubling"],
  ["出牌与胜负", "Play and Victory"],
  ["客户端约定", "Client Behavior"],
  ["知道了", "Close"],
  ["使用两副牌，共 108 张，每人 27 张；对家为同队。全员准备后随机发牌，首局从 2 开始并随机先手，后续由上一局头游先手，出牌按逆时针进行。每轮先手必须出牌，不能选择“不出”；同队两人均出完即结束本局，随后全员重新准备下一局。", "Two decks (108 cards) are used, with 27 cards per player. Opposite seats are partners. After everyone is ready, cards are dealt randomly. The first round starts at level 2 with a random leader; later rounds are led by the previous first finisher. Play proceeds counterclockwise. A leader must play and cannot pass. The round ends when both partners on one team finish."],
  ["单张、对子、三连对、三张、钢板、顺子、三带二，以及炸弹。红桃级牌是百搭牌，但不能代替大小王。", "Legal hands include singles, pairs, three consecutive pairs, triples, two consecutive triples, straights, full houses, and bombs. The heart level card is wild but cannot replace a Joker."],
  ["天王炸 ＞ 六张及以上普通炸弹 ＞ 同花顺 ＞ 五张炸弹 ＞ 四张炸弹；同类炸弹再按张数与点数比较。", "Joker bomb > six-or-more-card bomb > straight flush > five-card bomb > four-card bomb. Bombs of the same class are compared by size and rank."],
  ["我方和对方分别保存自己的级数。头游搭档为二游、三游、末游时，头游方分别从自己的级数升 3、2、1 级，另一方级数不变；下一局当前级牌等于头游方升级后的级数。级数不会跳过 A，A 级头游与末游同队时仍停留在 A。", "Each team keeps its own level. If the first finisher's partner places second, third, or last, that team advances 3, 2, or 1 levels respectively; the other team's level is unchanged. The next round uses the upgraded team's level. A cannot be skipped, and a first-plus-last finish at A remains at A."],
  ["“理牌”只改变本地展示分组，不会改变规则引擎中的手牌、出牌权或合法动作。", "Hand Layout changes only the local visual grouping. It never changes the engine hand, turn ownership, or legal actions."],
  ["使用一副 54 张牌，每人先发 17 张，余下 3 张作为底牌。首叫玩家随机确定，依次可以不叫或叫 1—3 分；后叫必须高于当前叫分，叫到 3 分立即结束，三人均不叫则重新发牌。", "One 54-card deck is used. Each player receives 17 cards and 3 cards remain as the bottom cards. The first bidder is random. Players may pass or bid 1–3; each bid must exceed the current bid. A bid of 3 ends bidding immediately, while three passes trigger a redeal."],
  ["地主确定后公开三张底牌并加入地主手牌。两位农民分别选择“不加倍”或“加倍”；至少一位农民加倍时，地主可选择“再加倍”。这些选择与炸弹、火箭、春天共同影响结算倍数。", "After the landlord is chosen, the three bottom cards are revealed and added to the landlord's hand. Each farmer chooses whether to double. If at least one farmer doubles, the landlord may redouble. These choices, bombs, rockets, and springs affect the final multiplier."],
  ["地主先出牌，后续依次行动。首出不能不出，跟牌可以不出；地主先出完则地主获胜，任一农民先出完则农民方共同获胜。合法牌型与大小比较均由 Dou_platform 规则引擎判定。", "The landlord leads. A leader cannot pass, while followers may pass. The landlord wins by finishing first; the farmers win together when either farmer finishes first. Dou_platform determines all legal hands and comparisons."],
  ["手牌按点数自动分组；理牌只改变展示顺序。每个决策显示 20 秒倒计时，但平台不提供超时代打，只有当前玩家能够提交服务器给出的合法动作。", "Cards are automatically grouped by rank; layout changes only display order. Every decision shows a 20-second timer, but there is no timeout autoplay. Only the current player may submit a server-provided legal action."],
  ["叫分、加倍、再加倍与出牌均使用服务器当前决策中的动作索引，客户端不会自行推断或替换动作。", "Bids, doubles, redoubles, and plays use action indices from the current server decision. The client never infers or substitutes actions."],
]);

const PATTERNS = [
  [/^胜方：([\d、]+)\s*号位队$/u, (_, seats) => `Winners: Seats ${seats.replaceAll("、", ", ")}`],
  [/^([\d、]+)\s*号位队获胜$/u, (_, seats) => `Seats ${seats.replaceAll("、", ", ")} win`],
  [/^第 (\d+) 名$/u, "Place $1"],
  [/^(头游|二游|三游|末游)\s*(\d+)号位$/u, (_, place, seat) => `${EXACT_TEXT.get(place)} Seat ${seat}`],
  [/^第 (\d+) 轮 · (\d+) 次出牌$/u, "Trick $1 · $2 actions"],
  [/^本轮结束，(\d+) 号位先出$/u, "Trick complete · Seat $1 leads"],
  [/^已预选 (\d+) 张，轮到你时校验$/u, "$1 preselected · Checked on your turn"],
  [/^方案 (\d+)\/(\d+) ↻$/u, "Layout $1/$2 ↻"],
  [/^(.+)，方案 (\d+)\/(\d+)，重复点击切换方案$/u, (_, mode, index, count) => `${EXACT_TEXT.get(mode) ?? mode}, Layout ${index}/${count}, activate again to cycle layouts`],
  [/^已选 (\d+) 张 · 不能接当前这手牌$/u, "$1 selected · Cannot beat this hand"],
  [/^已选 (\d+) 张 · 当前可出 ([\d、]+) 张的组合$/u, (_, count, sizes) => `${count} selected · Legal plays contain ${sizes.replaceAll("、", ", ")} cards`],
  [/^已选 (\d+) 张 · 提交时选择牌型$/u, "$1 selected · Choose type on submit"],
  [/^已选 (\d+) 张 · (.+)$/u, (_, count, type) => `${count} selected · ${EXACT_TEXT.get(type) ?? type}`],
  [/^含红心(.+) · 按(.+?)([2-9]|10|J|Q|K|A)判定$/u, (_, level, type, rank) => `Heart ${level} included · Ruled as ${EXACT_TEXT.get(type) ?? type} ${rank}`],
  [/^(单张|对子|三张|顺子|三带二|三连对|钢板|同花顺|炸弹|天王炸) · (.+)$/u, (_, type, rank) => `${EXACT_TEXT.get(type)} · ${EXACT_TEXT.get(rank) ?? rank}`],
  [/^展开选牌，1\s*张手牌$/u, "Expanded selection, 1 card"],
  [/^展开选牌，(\d+)\s*张手牌$/u, "Expanded selection, $1 cards"],
  [/^分组 (\d+) · 1 张$/u, "Group $1 · 1 card"],
  [/^分组 (\d+) · (\d+) 张$/u, "Group $1 · $2 cards"],
  [/^已选 (\d+) 张 · 完成后回到牌桌出牌$/u, "$1 selected · Return to the table to play"],
  [/^选择整组\s*(\d+)，1\s*张，已选\s*(\d+)\s*张$/u, "Select group $1, 1 card, $2 selected"],
  [/^选择整组\s*(\d+)，(\d+)\s*张$/u, "Select group $1, $2 cards"],
  [/^选择整组\s*(\d+)，(\d+)\s*张，已选\s*(\d+)\s*张$/u, "Select group $1, $2 cards, $3 selected"],
  [/^(\d+)\/(\d+)\s*张$/u, "$1/$2 cards"],
  [/^桌号\s*(.+)$/u, "Table $1"],
  [/^(.+)\s*·\s*返回原座位继续$/u, "$1 · Resume your seat"],
  [/^(我方|对方)(双上|一三游|一四游)\s*·\s*([+−]\d+)$/u, (_, side, outcome, delta) => `${side === "我方" ? "Our Team" : "Opponents"} ${outcome === "双上" ? "One-Two Finish" : outcome === "一三游" ? "First-Third Finish" : "First-Last Finish"} · ${delta.replace("−", "-")}`],
  [/^我方\s*(.+?)\s*→\s*(.+?)\s*·\s*对方\s*(.+?)\s*→\s*(.+)$/u, "Our Team $1 → $2 · Opponents $3 → $4"],
  [/^我方\s*(.+?)\s*→\s*(.+)$/u, "Our Team $1 → $2"],
  [/^对方\s*(.+?)\s*→\s*(.+)$/u, "Opponents $1 → $2"],
  [/^下局级牌\s*(.+)$/u, "Next Round Level $1"],
  [/^(掼蛋|斗地主)\s*·\s*(\d+)\s*个人类座位$/u, (_, game, count) => `${game === "掼蛋" ? "GuanDan" : "DouDizhu"} · ${count === "1" ? "Solo" : `${count} players`}`],
  [/^玩家\s*·\s*(\d+)\s*号位$/u, "You · Seat $1"],
  [/^第\s*(\d+)\s*局\s*·\s*级牌\s*(.+)$/u, "Round $1 · Level $2"],
  [/^第\s*(\d+)\s*局$/u, "Round $1"],
  [/^当前级牌：对方\s*(.+)$/u, "Current Level: Opponents $1"],
  [/^当前级牌：我方\s*(.+)$/u, "Current Level: Our Team $1"],
  [/^当前级牌：(.+)$/u, "Current Level: $1"],
  [/^1\s*张(?:手牌)?$/u, "1 card"],
  [/^(\d+)\s*张手牌$/u, "$1 cards"],
  [/^(\d+)\s*张已选$/u, "$1 selected"],
  [/^(\d+)\s*张$/u, "$1 cards"],
  [/^(.+)的本回合剩余秒数$/u, "$1 Remaining Turn Time"],
  [/^策略已加载，但冠军身份未通过校验(.+)$/u, "Policy loaded, but champion identity is unverified$1"],
  [/^冠军权重与运行契约已校验(.+)$/u, "Champion weights and runtime contract verified$1"],
  [/^策略未就绪(.+)$/u, "Policy is not ready$1"],
  [/^本桌合法动作按(.+)判定$/u, "Legal actions use $1"],
  [/^等待\s*(.+?)出牌$/u, "$1's turn"],
  [/^等待\s*(.+?)(叫地主|选择是否加倍|选择是否再加倍)$/u, (_, name, action) => `Waiting for ${name} to ${action === "叫地主" ? "bid" : action === "选择是否加倍" ? "choose double" : "choose redouble"}`],
  [/^叫分记录\s*·\s*(.+)$/u, (_, trail) => `Bid History · ${trail
    .replace(/(\d+)\s*号位/gu, "Seat $1")
    .replace(/(\d+)\s*分/gu, "$1 points")
    .replace(/不叫/gu, "No Bid")}`],
  [/^对局倍率\s*·\s*(.+)$/u, (_, trail) => `Game Multipliers · ${trail.replace(/(\d+)\s*号位/gu, "Seat $1")}`],
  [/^(\d+)\s*号位：不叫$/u, "Seat $1: No Bid"],
  [/^(\d+)\s*号位：(\d+)\s*分$/u, "Seat $1: $2 points"],
  [/^(\d+)\s*号位$/u, "Seat $1"],
  [/^(\d+)\s*号位\s*·\s*(.+)$/u, (_, seat, label) => `Seat ${seat} · ${translateValue(label)}`],
  [/^理牌分组\s*(\d+)，1\s*张$/u, "Hand group $1, 1 card"],
  [/^理牌分组\s*(\d+)，(\d+)\s*张$/u, "Hand group $1, $2 cards"],
  [/^方案\s*(\d+)\/(\d+)；重复点击当前策略可切换。$/u, "Layout $1/$2 · Tap the active mode to cycle."],
  [/^选择\s+(.+)$/u, "Select $1"],
  [/^第\s*(\d+)\s*局结果$/u, "Round $1 Result"],
  [/^(\d+)\s*秒后重连$/u, "Reconnect in $1 seconds"],
  [/^叫\s*(\d+)\s*分$/u, "Bid $1"],
  [/^(\d+)\s*分$/u, "$1 points"],
  [/^桌号\s*(.+)\s*已复制。$/u, "Table $1 copied."],
  [/^换到\s*(\d+)\s*号座位$/u, "Move to Seat $1"],
  [/^观战可见手牌，共\s*(\d+)\s*张$/u, "$1 spectator-visible cards"],
  [/^打开回放\s*(.+)$/u, "Open Replay $1"],
];

function translateValue(value) {
  const raw = String(value ?? "");
  const trimmed = raw.trim();
  if (!trimmed) return raw;
  let translated = EXACT_TEXT.get(trimmed);
  if (translated === undefined) {
    for (const [pattern, replacement] of PATTERNS) {
      if (!pattern.test(trimmed)) continue;
      translated = trimmed.replace(pattern, replacement);
      break;
    }
  }
  if (translated === undefined || translated === trimmed) return raw;
  return `${raw.match(/^\s*/u)?.[0] ?? ""}${translated}${raw.match(/\s*$/u)?.[0] ?? ""}`;
}

function localizeElement(element, language, sourceAttributes) {
  if (element.closest?.("[data-l10n-ignore]")) return;
  for (const attribute of ["aria-label", "title", "placeholder", "data-variant-count"]) {
    if (!element.hasAttribute?.(attribute)) continue;
    let sources = sourceAttributes.get(element);
    if (!sources) {
      sources = new Map();
      sourceAttributes.set(element, sources);
    }
    if (!sources.has(attribute)) sources.set(attribute, element.getAttribute(attribute));
    const source = sources.get(attribute);
    const after = language === "en" ? translateValue(source) : source;
    if (after !== element.getAttribute(attribute)) element.setAttribute(attribute, after);
  }
}

function localizeTextNode(node, language, sourceTexts) {
  if (node.parentElement?.closest?.("[data-l10n-ignore]")) return;
  if (!sourceTexts.has(node)) sourceTexts.set(node, node.nodeValue);
  const source = sourceTexts.get(node);
  const after = language === "en" ? translateValue(source) : source;
  if (after !== node.nodeValue) node.nodeValue = after;
}

function localizeTree(root, documentRef, language, sourceTexts, sourceAttributes) {
  const view = documentRef.defaultView;
  const NodeType = view.Node;
  if (root.nodeType === NodeType.TEXT_NODE) {
    localizeTextNode(root, language, sourceTexts);
    return;
  }
  if (root.nodeType !== NodeType.ELEMENT_NODE && root.nodeType !== NodeType.DOCUMENT_NODE) return;
  if (root.nodeType === NodeType.ELEMENT_NODE && root.closest?.("[data-l10n-ignore]")) return;
  if (root.nodeType === NodeType.ELEMENT_NODE) localizeElement(root, language, sourceAttributes);
  root.querySelectorAll?.("*").forEach((element) => localizeElement(element, language, sourceAttributes));
  const walker = documentRef.createTreeWalker(root, view.NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    localizeTextNode(node, language, sourceTexts);
  }
}

function makeLanguageSwitch(documentRef, onLanguage) {
  const host = documentRef.createElement("div");
  host.className = "solo-language-switch";
  host.dataset.l10nIgnore = "true";
  host.setAttribute("role", "group");
  host.setAttribute("aria-label", "语言 / Language");
  for (const [language, label] of [["zh", "中文"], ["en", "EN"]]) {
    const button = documentRef.createElement("button");
    button.type = "button";
    button.dataset.language = language;
    button.textContent = label;
    button.addEventListener("click", () => onLanguage(language));
    host.append(button);
  }
  const connection = documentRef.getElementById("connectionChip");
  const tools = documentRef.createElement("div");
  tools.className = "solo-topbar-tools";
  connection.parentNode.insertBefore(tools, connection);
  tools.append(host, connection);
  return host;
}

export function installSoloLocale(documentRef = document, storage = undefined) {
  const view = documentRef.defaultView;
  const sourceTexts = new WeakMap();
  const sourceAttributes = new WeakMap();
  const storageKey = "cardks.solo.language.v1";
  let language = "zh";
  try {
    if (storage === undefined) storage = view.localStorage;
    language = storage?.getItem(storageKey) === "en" ? "en" : "zh";
  } catch { /* Locale must remain usable without browser persistence. */ }
  let languageSwitch;

  const applyLanguage = (nextLanguage, persist = true) => {
    language = nextLanguage === "en" ? "en" : "zh";
    documentRef.documentElement.lang = language === "en" ? "en" : "zh-CN";
    documentRef.documentElement.dataset.locale = language;
    localizeTree(documentRef.documentElement, documentRef, language, sourceTexts, sourceAttributes);
    languageSwitch?.querySelectorAll("button").forEach((button) => {
      const active = button.dataset.language === language;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (persist) { try { storage?.setItem(storageKey, language); } catch { /* Keep this page's locale. */ } }
    documentRef.dispatchEvent(new view.CustomEvent("solo-languagechange", { detail: { language } }));
  };

  languageSwitch = makeLanguageSwitch(documentRef, (nextLanguage) => applyLanguage(nextLanguage));
  applyLanguage(language, false);

  const observer = new view.MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "characterData") {
        const node = mutation.target;
        if (node.parentElement?.closest?.("[data-l10n-ignore]")) continue;
        const source = sourceTexts.get(node);
        const expected = source === undefined ? undefined : language === "en" ? translateValue(source) : source;
        if (expected !== undefined && node.nodeValue === expected) continue;
        sourceTexts.set(node, node.nodeValue);
        localizeTextNode(node, language, sourceTexts);
      } else if (mutation.type === "attributes") {
        const element = mutation.target;
        if (element.closest?.("[data-l10n-ignore]")) continue;
        const sources = sourceAttributes.get(element);
        const source = sources?.get(mutation.attributeName);
        const current = element.getAttribute(mutation.attributeName);
        const expected = source === undefined ? undefined : language === "en" ? translateValue(source) : source;
        if (expected !== undefined && current === expected) continue;
        if (!sources) sourceAttributes.set(element, new Map([[mutation.attributeName, current]]));
        else sources.set(mutation.attributeName, current);
        localizeElement(element, language, sourceAttributes);
      } else {
        mutation.addedNodes.forEach((node) => localizeTree(node, documentRef, language, sourceTexts, sourceAttributes));
      }
    }
  });
  observer.observe(documentRef.documentElement, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: ["aria-label", "title", "placeholder", "data-variant-count"],
  });
  return { get language() { return language; }, setLanguage: applyLanguage, observer };
}
