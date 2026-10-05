package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/guandan-service/hand-arranger/common"
)

const (
	presentationVersion = "battle-platform.presentation.v2"
	sourceSHA           = "hand:b91ce83760ef3c60d076e79b6afc6a1a5ed50349ee8c17b97ccc86ef27b6e36b;pattern:2a712692b3e2ab984bba927cb1e2674adb1767fdfba75bf9d257b8c01376aa21;splitter:778438601e6811775a70268292a51c3f5925123ee2682ee95dcd005020f2c325;splitter1:3c8a0d25aa84b3ffc8206b1a72dd08f2bbd6daf16207530eab9d541a58bc6f6a;splitter2:7f751f1ba0b70af30e42e3f2d3b2f0200e0ef9a3bde0e692f56a634fa9506c4b;splitter3:92e64551f258689830c167bec7c23b973df52f40acc0581a7fa16856588c4983;splitter4:3996c711b22352d7554f78786b4a08a282e80d5fc1c96273dd950d24a45f3c68"
)

type request struct {
	ID    string   `json:"id"`
	Game  string   `json:"game"`
	Level string   `json:"level"`
	Mode  int8     `json:"mode"`
	Cards []string `json:"cards"`
}

type group struct {
	Pattern int8     `json:"pattern"`
	Minor   int8     `json:"minor"`
	Value   int8     `json:"value"`
	Cards   []string `json:"cards"`
}

type response struct {
	ID                  string  `json:"id"`
	Version             string  `json:"version"`
	SourceSHA           string  `json:"source_sha"`
	Mode                int8    `json:"mode"`
	Groups              []group `json:"groups"`
	Error               string  `json:"error,omitempty"`
	PresentationVersion string  `json:"presentation_version"`
}

var suitToIndex = map[byte]int8{'S': common.SUIT_SPADE, 'H': common.SUIT_HEART, 'C': common.SUIT_CLUB, 'D': common.SUIT_DIAMOND}
var indexToSuit = map[int8]byte{common.SUIT_SPADE: 'S', common.SUIT_HEART: 'H', common.SUIT_CLUB: 'C', common.SUIT_DIAMOND: 'D'}
var rankToValue = map[string]int8{
	"A": common.VAL_A, "2": common.VAL_2, "3": common.VAL_3, "4": common.VAL_4,
	"5": common.VAL_5, "6": common.VAL_6, "7": common.VAL_7, "8": common.VAL_8,
	"9": common.VAL_9, "T": common.VAL_10, "10": common.VAL_10,
	"J": common.VAL_J, "Q": common.VAL_Q, "K": common.VAL_K,
}
var valueToRank = map[int8]string{
	common.VAL_A: "A", common.VAL_2: "2", common.VAL_3: "3", common.VAL_4: "4",
	common.VAL_5: "5", common.VAL_6: "6", common.VAL_7: "7", common.VAL_8: "8",
	common.VAL_9: "9", common.VAL_10: "T", common.VAL_J: "J", common.VAL_Q: "Q", common.VAL_K: "K",
}

func parseCard(raw string) (int8, error) {
	code := strings.ToUpper(strings.TrimSpace(raw))
	switch code {
	case "SB", "BJ":
		return common.TILE_BJ, nil
	case "HR", "RJ":
		return common.TILE_RJ, nil
	}
	if len(code) < 2 {
		return 0, fmt.Errorf("invalid card %q", raw)
	}
	suit, ok := suitToIndex[code[0]]
	if !ok {
		return 0, fmt.Errorf("invalid card suit %q", raw)
	}
	value, ok := rankToValue[code[1:]]
	if !ok {
		return 0, fmt.Errorf("invalid card rank %q", raw)
	}
	return common.MakeTile(suit, value), nil
}

func formatCard(tile int8) (string, error) {
	if tile == common.TILE_BJ {
		return "SB", nil
	}
	if tile == common.TILE_RJ {
		return "HR", nil
	}
	suit, ok := indexToSuit[common.GetSuit(tile)]
	if !ok {
		return "", fmt.Errorf("invalid output tile 0x%x", tile)
	}
	rank, ok := valueToRank[common.GetValue(tile)]
	if !ok {
		return "", fmt.Errorf("invalid output tile 0x%x", tile)
	}
	return string(suit) + rank, nil
}

func parseLevel(raw string) (int8, error) {
	value, ok := rankToValue[strings.ToUpper(strings.TrimSpace(raw))]
	if !ok || value < common.VAL_A || value > common.VAL_K {
		return 0, fmt.Errorf("invalid level %q", raw)
	}
	return value, nil
}

func arrangeGuandan(req request) ([]group, int8, error) {
	mode := req.Mode

	if !common.IsValidMode(mode) {
		return nil, mode, fmt.Errorf("invalid splitter mode %d", mode)
	}
	if len(req.Cards) > common.HAND_COUNT {
		return nil, mode, fmt.Errorf("hand has %d cards, maximum is %d", len(req.Cards), common.HAND_COUNT)
	}
	level, err := parseLevel(req.Level)
	if err != nil {
		return nil, mode, err
	}
	if len(req.Cards) == 0 {
		return []group{}, mode, nil
	}
	tiles := make([]int8, 0, len(req.Cards))
	var hand common.Hand
	counts := map[int8]int{}
	for _, card := range req.Cards {
		tile, parseErr := parseCard(card)
		if parseErr != nil {
			return nil, mode, parseErr
		}
		counts[tile]++
		if counts[tile] > 2 {
			return nil, mode, fmt.Errorf("card %q occurs more than twice", card)
		}
		tiles = append(tiles, tile)
		hand.Add(tile)
	}
	var splitter common.Splitter
	splitter.Init()
	splitter.SetLevel(level)
	melds := splitter.Split(&hand, mode)
	splitter.Sort(melds)
	result := make([]group, 0, len(melds))
	seen := make([]int8, 0, len(tiles))
	for _, meld := range melds {
		cards := make([]string, 0, len(meld.List()))
		for _, tile := range meld.List() {
			code, formatErr := formatCard(tile)
			if formatErr != nil {
				return nil, mode, formatErr
			}
			cards = append(cards, code)
			seen = append(seen, tile)
		}
		result = append(result, group{
			Pattern: meld.Group.Pattern,
			Minor:   meld.Group.Minor,
			Value:   meld.Group.Value,
			Cards:   cards,
		})
	}
	sort.Slice(tiles, func(i, j int) bool { return tiles[i] < tiles[j] })
	sort.Slice(seen, func(i, j int) bool { return seen[i] < seen[j] })
	if len(tiles) != len(seen) {
		return nil, mode, fmt.Errorf("multiset conservation failed: input=%d output=%d", len(tiles), len(seen))
	}
	for index := range tiles {
		if tiles[index] != seen[index] {
			return nil, mode, fmt.Errorf("multiset conservation failed at index %d", index)
		}
	}
	return result, mode, nil
}

func handle(req request) response {
	out := response{
		ID:                  req.ID,
		Version:             presentationVersion,
		SourceSHA:           sourceSHA,
		PresentationVersion: presentationVersion,
	}
	var err error
	switch strings.ToLower(req.Game) {
	case "guandan":
		out.Groups, out.Mode, err = arrangeGuandan(req)
	default:
		err = fmt.Errorf("unsupported game %q", req.Game)
	}
	if err != nil {
		out.Error = err.Error()
	}
	return out
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4096), 4*1024*1024)
	encoder := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		var req request
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			_ = encoder.Encode(response{
				Version:             presentationVersion,
				SourceSHA:           sourceSHA,
				PresentationVersion: presentationVersion,
				Error:               fmt.Sprintf("invalid request: %v", err),
			})
			continue
		}
		if err := encoder.Encode(handle(req)); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
