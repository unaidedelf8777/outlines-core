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
    pub(crate) index: Vec<u32>,
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) next_states: Vec<StateId>,
}

impl StateTransitions {
    fn new() -> Self {
        Self {
            index: Vec::new(),
            tokens: Vec::new(),
            next_states: Vec::new(),
        }
    }

    fn insert(&mut self, token: TokenId, state: StateId) {
        if self.tokens.len() * 2 + 1 > self.index.len() {
            self.resize();
        }
        let mask = self.index.len() - 1;
        let mut pos = (token as usize) & mask;
        loop {
            let slot = self.index[pos];
            if slot == EMPTY {
                self.index[pos] = self.tokens.len() as u32;
                self.tokens.push(token);
                self.next_states.push(state);
                break;
            } else if self.tokens[slot as usize] == token {
                self.next_states[slot as usize] = state;
                break;
            } else {
                pos = (pos + 1) & mask;
            }
        }
    }

    fn resize(&mut self) {
        let new_len = (self.index.len().max(1) * 2).next_power_of_two();
        let mut new_index = vec![EMPTY; new_len];
        let mask = new_len - 1;
        for i in 0..self.tokens.len() {
            let token = self.tokens[i];
            let mut pos = (token as usize) & mask;
            while new_index[pos] != EMPTY {
                pos = (pos + 1) & mask;
            }
            new_index[pos] = i as u32;
        }
        self.index = new_index;
    }

    fn get(&self, token: TokenId) -> Option<StateId> {
        if self.index.is_empty() {
            return None;
        }
        let mask = self.index.len() - 1;
        let mut pos = (token as usize) & mask;
        loop {
            let slot = self.index[pos];
            if slot == EMPTY {
                return None;
            }
            if self.tokens[slot as usize] == token {
                return Some(self.next_states[slot as usize]);
            }
            pos = (pos + 1) & mask;
        }
    }

    fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
}

#[inline(never)]
pub fn is_match_state<T>(dfa: &DFA<T>, state: AutomataStateId) -> bool  where T: AsRef<[u32]> {
    dfa.is_match_state(state) && !dfa.is_match_state(dfa.next_eoi_state(state))
}

#[inline(never)]
pub fn next_state<T>(dfa: &DFA<T>, state: AutomataStateId, b: u8) -> AutomataStateId where T: AsRef<[u32]> {
    dfa.next_state(state, b)
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
        let trie = vocabulary.trie();
        let st = std::time::Instant::now();
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
                stack.push((trie.node(trie.root_index()), current_state));
                let mut ret = Vec::new();

                while let Some((node, dfa_state)) = stack.pop() {

                    for (b, child_idx) in node.children() {
                        let next_state = next_state(&dfa, dfa_state, b);
                        if dfa.is_dead_state(next_state) || dfa.is_quit_state(next_state) {
                            continue;
                        }
                        if !is_match_state(&dfa, next_state) || is_match_state(&dfa, dfa.next_eoi_state(next_state)) {
                            if let Some(ids) = trie.node(child_idx).terminal_ids() {
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
                                let target_idx_sid = target_idx as StateId;

                                for id in ids {
                                    ret.push((*id, target_idx_sid));
                                }
                            }
                            stack.push((trie.node(child_idx), next_state));
                        }
                    }
                }
                for (token_id, state_id) in ret {
                    transitions[current_idx].insert(token_id, state_id);
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
    pub fn allowed_tokens(&self, state: &StateId) -> Option<&[TokenId]> {
        self.transitions.get(*state as usize).map(|t| t.tokens())
    }

    pub fn allowed_tokens_iter(&self, state: &StateId) -> Option<impl Iterator<Item = &TokenId>> {
        self.transitions
            .get(*state as usize)
            .map(|t| t.tokens.iter())
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
                .tokens
                .iter()
                .zip(row.next_states.iter())
                .map(|(t, s)| (*t, *s))
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
