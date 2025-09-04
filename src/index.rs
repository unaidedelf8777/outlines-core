//! Building an `Index` to efficiently map vocabulary tokens to state transitions.

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::FxHashMap as HashMap;

use crate::prelude::*;
use crate::vocabulary::Vocabulary;
use crate::vocabulary::trie::TrieNode;
use crate::{Error, Result};

const EMPTY: u32 = u32::MAX;


#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub(crate) struct StateTransitions {
    pub(crate) index: Vec<u32>, // open-address table: slot -> pairs index
    pub(crate) pairs: Vec<u64>, // packed (token | state<<32)
}

impl StateTransitions {
    // Tunable load factor: 90%
    const LOAD_NUM: usize = 9;
    const LOAD_DEN: usize = 10;

    pub fn new() -> Self {
        Self {
            index: Vec::new(),
            pairs: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty() && self.pairs.is_empty()
    }

    #[inline(always)]
    fn pack(token: TokenId, state: StateId) -> u64 { ((state as u64) << 32) | (token as u64) }

    #[inline(always)]
    fn hash(token: u32) -> u32 {
        token.wrapping_mul(0x9E37_79B1)
    }

    #[inline(always)]
    fn unpack_state(pair: u64) -> StateId { (pair >> 32) as u32 }

    #[inline(always)]
    fn unpack_token(pair: u64) -> TokenId { pair as u32 }

    #[inline]
    fn need_grow(&self) -> bool {
        if self.index.is_empty() {
            return true;
        }
        // Will we exceed 90% *after* inserting one more?
        (self.pairs.len() + 1) * Self::LOAD_DEN > self.index.len() * Self::LOAD_NUM
    }

    /// Build from duplicate-free pairs, sizing so that adding *one* more entry
    /// (EOS) still stays at or below a 90% load factor.
    pub fn bulk_build(pairs_in: &[(TokenId, StateId)]) -> Self {
        let entries = pairs_in.len();

        // We guarantee room for one more entry without resize:
        // choose index_len s.t. ceil((entries + 1) / 0.9) is a power of two.
        let target_entries = entries + 1; // account for future EOS insert
        let mut need = (target_entries * Self::LOAD_DEN + (Self::LOAD_NUM - 1)) / Self::LOAD_NUM; // ceil
        need = need.max(2); // at least 2

        let index_len = need.next_power_of_two();
        let mut index = vec![EMPTY; index_len];
        let mut pairs = Vec::with_capacity(target_entries); // exact room for EOS later

        if entries == 0 {
            return Self { index, pairs };
        }

        let mask = (index_len - 1) as u32;

        // Fill once; input has no duplicates.
        for &(token, state) in pairs_in {
            let mut pos = (Self::hash(token) & mask) as usize;
            while index[pos] != EMPTY {
                pos = (pos + 1) & (mask as usize);
            }
            let idx = pairs.len() as u32;
            index[pos] = idx;
            pairs.push(Self::pack(token, state));
        }

        Self { index, pairs }
    }

    pub fn insert(&mut self, token: TokenId, state: StateId) {
        // With 90% LF, this won't trigger when bulk_build sized correctly for one extra.
        if self.index.is_empty() || self.need_grow() {
            self.resize();
        }
        let mask = (self.index.len() - 1) as u32;
        let mut pos = (Self::hash(token) & mask) as usize;

        loop {
            let slot = self.index[pos];
            if slot == EMPTY {
                let idx = self.pairs.len() as u32;
                self.index[pos] = idx;
                self.pairs.push(Self::pack(token, state));
                return;
            }
            let i = slot as usize;
            let pair = self.pairs[i];
            if Self::unpack_token(pair) == token {
                self.pairs[i] = Self::pack(token, state);
                return;
            }
            pos = (pos + 1) & (mask as usize);
        }
    }

    fn resize(&mut self) {
        let new_len = (self.index.len().max(1) * 2).next_power_of_two();
        let mut new_index = vec![EMPTY; new_len];
        let mask = (new_len - 1) as u32;

        for (i, &pair) in self.pairs.iter().enumerate() {
            let token = Self::unpack_token(pair);
            let mut pos = (Self::hash(token) & mask) as usize;
            while new_index[pos] != EMPTY {
                pos = (pos + 1) & (mask as usize);
            }
            new_index[pos] = i as u32;
        }
        self.index = new_index;
    }

    #[inline(always)]
    pub fn get(&self, token: TokenId) -> Option<StateId> {
        let len = self.index.len();
        if len == 0 { return None; }
        let mask = (len - 1) as u32;
        let mut pos = (Self::hash(token) & mask) as usize;

        loop {
            let slot = self.index[pos];
            if slot == EMPTY { return None; }
            let pair = self.pairs[slot as usize];
            if Self::unpack_token(pair) == token {
                return Some(Self::unpack_state(pair));
            }
            pos = (pos + 1) & (mask as usize);
        }
    }

    /// O(1) to create, zero-alloc iterator over all TokenIds.
    pub fn tokens(&self) -> Tokens<'_> {
        Tokens { it: self.pairs.iter() }
    }

    pub fn states(&self) -> States<'_> {
        States { it: self.pairs.iter() }
    }
}

/// Zero-alloc iterator over tokens (views low 32 bits of each pair).
pub struct Tokens<'a> {
    it: std::slice::Iter<'a, u64>,
}
impl<'a> Iterator for Tokens<'a> {
    type Item = TokenId;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|&p| p as u32)
    }
}

/// Zero-alloc iterator over states (high 32 bits of each pair).
pub struct States<'a> {
    it: std::slice::Iter<'a, u64>,
}
impl<'a> Iterator for States<'a> {
    type Item = StateId;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|&p| (p >> 32) as u32)
    }
}

/// `Index` efficiently maps vocabulary tokens to state transitions.
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Index {
    /// The ID of the initial state in the automaton, processing begins from this state.
    initial_state: StateId,
    /// Marker for terminal states, indexed by state id.
    final_states: Vec<bool>,
    /// Transition tables for each state.
    transitions: Vec<StateTransitions>,
    /// The token ID reserved for the "end-of-sequence" token.
    eos_token_id: TokenId,
    /// The size of the vocabulary used to build the index.
    vocab_size: usize,
}
/// The `Index` structure is designed to efficiently map tokens from a given vocabulary
/// to state transitions within a finite-state automaton.
///
/// ## Usage:
/// The `Index` is typically constructed by combining a vocabulary and regular expressions.
/// Once built, it can be used to efficiently evaluate token sequences or to validate input data.
///
/// ## Example:
/// ```rust
/// use outlines_core::prelude::*;
///
/// # fn run() -> Result<(), outlines_core::Error> {
/// let regex = "0|[1-9][0-9]*";
/// let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", None)?;
/// let index = Index::new(regex, &vocabulary)?;
///
/// let initial_state = index.initial_state();
/// println!("Initial state is {}", initial_state);
/// println!("Is initial state a final state? {}", index.is_final_state(initial_state));
///
/// let allowed_tokens = index.allowed_tokens(&initial_state).expect("Some allowed tokens");
/// println!("Allowed tokens at initial state are {:?}", allowed_tokens);
///
/// let token_id = allowed_tokens.first().expect("First token");
/// println!("Next state for the token_id {} is {:?}", token_id, index.next_state(&initial_state, token_id));
///
/// println!("Final states are {:?}", index.final_states().collect::<Vec<_>>());
/// println!("Index has exactly {} transitions", index.transitions().len());
/// # Ok(())
/// # }
///
/// ```
///
/// ## Performance:
/// - **Complexity**:
///   The `Index` can accommodate large vocabularies and complex regular expressions.
///   However, its size may grow significantly with the complexity of the input.
/// - **Construction Cost**:
///   Building the `Index` involves processing the vocabulary and regular expressions,
///   which may require a considerable amount of time and computational resources.
impl Index {
    /// Builds an `Index` from regular expression and vocabulary tokens.
    pub fn new(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
        let eos_token_id = vocabulary.eos_token_id();
        let max_token_id = vocabulary
            .max_token_id()
            .unwrap_or(0)
            .max(eos_token_id);
        let vocab_size = max_token_id as usize + 1;
        let dfa = DFA::new(regex).map_err(Box::new)?;
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::DfaHasNoStartState),
        };

        let mut transitions: Vec<StateTransitions> = vec![StateTransitions::new()];
        let mut final_states: Vec<bool> = vec![false];
        let mut seen: Vec<bool> = vec![false];
        let stride = dfa.stride();

        let mut next_states = vec![start_state];
        // buffer so we can bulk build transitions
        let mut ret = Vec::new();
        let trie = vocabulary.trie();
        while let Some(current_state) = next_states.pop() {
            let current_idx = current_state.as_usize() / stride;

            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.resize(current_idx + 1, false);
                final_states[current_idx] = true;
            }

            // Traverse the trie and DFA simultaneously to find valid tokens from this state.
            {
                let mut stack: Vec<(&TrieNode, AutomataStateId)> = Vec::new();
                // start from trie root and current DFA state
                stack.push((trie.root(), current_state));

                while let Some((node, dfa_state)) = stack.pop() {

                    for n in trie.children(node) {
                        let next_state = dfa.next_state(dfa_state, n.byte());
                        if dfa.is_dead_state(next_state) || dfa.is_quit_state(next_state) {
                            continue;
                        }
                        if !dfa.is_match_state(next_state) || dfa.is_match_state(dfa.next_eoi_state(next_state)) {
                            if let Some(id) = n.token_id() {
                                // make sure current_idx has a entry in state_map
                                let target_idx = (next_state.as_usize() / stride);
                                if target_idx >= transitions.len() {
                                    transitions.resize(target_idx + 1, StateTransitions::new());
                                    final_states.resize(target_idx + 1, false);
                                    seen.resize(target_idx + 1, false);
                                }
                                if !seen[target_idx] {
                                    seen[target_idx] = true;
                                    next_states.push(next_state);
                                }
                                ret.push((id, target_idx as StateId));
                            }
                            stack.push((n, next_state));
                        }
                    }
                }
                if !ret.is_empty() {
                    transitions[current_idx] = StateTransitions::bulk_build(&ret);
                    ret.clear();
                }
            }
        }

        for (state_idx, &is_final) in final_states.iter().enumerate() {
            if is_final {
                if state_idx >= transitions.len() {
                    transitions.resize(state_idx + 1, StateTransitions::new());
                }
                transitions[state_idx].insert(eos_token_id, state_idx as StateId);
            }
        }

        Ok(Self {
            initial_state: (start_state.as_usize() / stride) as StateId,
            final_states,
            transitions,
            eos_token_id,
            vocab_size,
        })
    }

    /// Returns the ID of the initial state in the automaton.
    pub fn initial_state(&self) -> StateId {
        self.initial_state
    }

    /// Returns an iterator over all final states.
    pub fn final_states(&self) -> impl Iterator<Item = StateId> + '_ {
        self.final_states
            .iter()
            .enumerate()
            .filter_map(|(i, &is_final)| is_final.then(|| i as StateId))
    }

    /// Returns the transition table.
    pub(crate) fn transitions(&self) -> &[StateTransitions] {
        &self.transitions
    }

    /// Checks if state is in final states set or not.
    pub fn is_final_state(&self, state: StateId) -> bool {
        self.final_states
            .get(state as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Lists allowed tokens for a given state ID or `None` if it is not found in `Index`.
    pub fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        let t = self.transitions.get(*state as usize);
        match t {
            Some(t) => Some(t.tokens().collect::<Vec<u32>>()),
            None => None,
        }
    }

    pub fn allowed_tokens_iter(&self, state: &StateId) -> Option<impl Iterator<Item = TokenId> + use<'_>> {
        self.transitions
            .get(*state as usize)
            .map(|t| t.tokens())
    }

    /// Returns transition state for a given state and token id or `None` otherwise.
    pub fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if token_id == &self.eos_token_id {
            return None;
        }
        let state_idx = *state as usize;
        self.transitions
            .get(state_idx)
            .and_then(|row| row.get(*token_id))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index object with transitions:")?;
        for (state_id, row) in self.transitions.iter().enumerate() {
            let pairs: Vec<_> = row
                .tokens()
                .zip(row.states())
                .map(|(t, s)| (t, s))
                .collect();
            if !pairs.is_empty() {
                writeln!(f, "{} -> {:?}", state_id, pairs)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn index_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let eos_token_id = 4;
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let initial_state = index.initial_state();
        assert_eq!(initial_state, 0);
        assert_eq!(index.final_states().count(), 3);
        assert!(!index.is_final_state(initial_state));

        let allowed_tokens = index
            .allowed_tokens(&initial_state)
            .expect("No allowed tokens");
        assert!(allowed_tokens.contains(&3));
        let state = index.next_state(&initial_state, &3).unwrap();
        assert!(index.is_final_state(state));

        assert_eq!(index.next_state(&state, &eos_token_id), None);
        assert_eq!(index.next_state(&state, &3), None);
    }

    #[test]
    fn index_from_regex_initital_in_allowed() {
        let regex = "`\\n(\\.\\n)?`\\n";
        let mut vocabulary = Vocabulary::new(104);
        for (token, token_id) in [("\n", 103), (".", 102), ("`", 101)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let allowed = index
            .allowed_tokens(&index.initial_state())
            .expect("No allowed tokens");
        assert!(allowed.contains(&101));
    }

    #[test]
    fn index_from_regex_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let mut vocabulary = Vocabulary::new(8);
        for (token, token_id) in [(" 😍", 5), ("blah", 0), ("😇", 2), ("😈a", 1), ("😍", 3)]
        {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        for (token, token_id) in [
            (vec![32, 240, 159, 152], 7),
            (vec![32, 240, 159, 152, 141], 6),
            (vec![240, 159, 152, 141], 4),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.final_states().count(), 2);
    }
}
