//! Sparse-byte trie: each node stores a compact edge list `(byte, next_node)`.
//! This removes the 256-slot array and makes iteration O(#children).
//!
//! Notes:
//! - `children()` iterates a slice of edges (fast).
//! - `get_child(b)` is a tiny linear scan (or binary search if you prefer).
//! - Node indices remain stable; we do not reclaim `nodes` slots on prune.
//! - `SmallVec` is serialized by bridging through `Vec` in wrapper newtypes.

use bincode::{Decode, Encode, BorrowDecode};
use smallvec::SmallVec;
use std::ops::Deref;

use crate::primitives::{Token, TokenId};

const INLINE_IDS: usize = 128;   // tune to 1/2/4 as needed
const INLINE_EDGES: usize = 32; // typical branching factor; tune as needed

// ========================= Ids (SmallVec wrapper) =========================

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Ids(SmallVec<[TokenId; INLINE_IDS]>);

// ---- Encode / Decode (bincode v2) ----

impl Encode for Ids {
    #[inline]
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        // serialize as a slice of TokenId
        self.0.as_slice().encode(encoder)
    }
}

impl<Context> Decode<Context> for Ids
where
    Vec<TokenId>: Decode<Context>,
{
    #[inline]
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let v: Vec<TokenId> = Decode::decode(decoder)?;
        Ok(Ids(SmallVec::from_vec(v)))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Ids
where
    Vec<TokenId>: BorrowDecode<'de, Context>,
{
    #[inline]
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let v: Vec<TokenId> = BorrowDecode::borrow_decode(decoder)?;
        Ok(Ids(SmallVec::from_vec(v)))
    }
}

// ---- Convenience traits so &Ids works like &[TokenId] ----

impl Deref for Ids {
    type Target = [TokenId];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl<'a> IntoIterator for &'a Ids {
    type Item = &'a TokenId;
    type IntoIter = std::slice::Iter<'a, TokenId>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.as_slice().iter()
    }
}

// ---- Helpers used by trie code ----
impl Ids {
    #[inline] pub fn is_empty(&self) -> bool { self.0.is_empty() }
    #[inline] pub fn clear(&mut self) { self.0.clear() }
    #[inline] pub fn push(&mut self, id: TokenId) { self.0.push(id) }
    #[inline] pub fn contains(&self, id: &TokenId) -> bool { self.0.contains(id) }
    #[inline] pub fn len(&self) -> usize { self.0.len() }
    #[inline] pub fn as_slice(&self) -> &[TokenId] { self.0.as_slice() }
    #[inline] pub fn iter(&self) -> std::slice::Iter<'_, TokenId> { self.0.iter() }
}

// ========================= Edges (SmallVec wrapper) =========================

#[derive(Clone, Copy, Debug, PartialEq)]
struct Edge {
    b: u8,
    idx: u32, // child node index (u32 saves space vs usize)
}

// Encode/Decode for Edge via (u8, u32)
impl Encode for Edge {
    #[inline]
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        (self.b, self.idx).encode(encoder)
    }
}
impl<Context> Decode<Context> for Edge {
    #[inline]
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let (b, idx) = <(u8, u32)>::decode(decoder)?;
        Ok(Edge { b, idx })
    }
}
impl<'de, Context> BorrowDecode<'de, Context> for Edge
where
    (u8, u32): BorrowDecode<'de, Context>,
{
    #[inline]
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let (b, idx) = <(u8, u32)>::borrow_decode(decoder)?;
        Ok(Edge { b, idx })
    }
}

// Wrapper around SmallVec so we can implement (Borrow)Encode/Decode.
#[derive(Clone, Debug, PartialEq, Default)]
struct Edges(SmallVec<[Edge; INLINE_EDGES]>);

impl Encode for Edges {
    #[inline]
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        self.0.as_slice().encode(encoder)
    }
}
impl<Context> Decode<Context> for Edges
where
    Vec<Edge>: Decode<Context>,
{
    #[inline]
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let v: Vec<Edge> = Decode::decode(decoder)?;
        Ok(Edges(SmallVec::from_vec(v)))
    }
}
impl<'de, Context> BorrowDecode<'de, Context> for Edges
where
    Vec<Edge>: BorrowDecode<'de, Context>,
{
    #[inline]
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let v: Vec<Edge> = BorrowDecode::borrow_decode(decoder)?;
        Ok(Edges(SmallVec::from_vec(v)))
    }
}

impl Edges {
    #[inline] fn is_empty(&self) -> bool { self.0.is_empty() }
    #[inline] fn len(&self) -> usize { self.0.len() }
    #[inline] fn as_slice(&self) -> &[Edge] { self.0.as_slice() }
    #[inline] fn iter(&self) -> std::slice::Iter<'_, Edge> { self.0.iter() }

    // Keep edges sorted by byte for deterministic order and optional binary_search.
    #[inline]
    fn find_pos(&self, b: u8) -> Result<usize, usize> {
        self.0.binary_search_by_key(&b, |e| e.b)
    }

    #[inline]
    fn get_child(&self, b: u8) -> Option<usize> {
        match self.find_pos(b) {
            Ok(p) => Some(self.0[p].idx as usize),
            Err(_) => None,
        }
    }

    #[inline]
    fn set_child(&mut self, b: u8, idx: usize) {
        match self.find_pos(b) {
            Ok(p) => self.0[p].idx = idx as u32,
            Err(p) => self.0.insert(p, Edge { b, idx: idx as u32 }),
        }
    }

    #[inline]
    fn clear_child(&mut self, b: u8) {
        if let Ok(p) = self.find_pos(b) {
            self.0.remove(p);
        }
    }
}

// ========================= Trie & Nodes =========================

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub(crate) struct TrieNode {
    edges: Edges,
    ids: Ids,
}

impl Default for TrieNode {
    #[inline]
    fn default() -> Self {
        Self { edges: Edges::default(), ids: Ids::default() }
    }
}

pub(crate) struct ChildrenIter<'a> {
    slice: &'a [Edge],
    i: usize,
}

impl<'a> Iterator for ChildrenIter<'a> {
    type Item = (u8, usize);
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i >= self.slice.len() { return None; }
        self.i += 1;
        let e = self.slice[i];
        Some((e.b, e.idx as usize))
    }
}

impl TrieNode {
    /// Iterate existing children as `(byte, child_index)` without allocation (O(#children)).
    #[inline]
    pub(crate) fn children(&self) -> ChildrenIter<'_> {
        ChildrenIter { slice: self.edges.as_slice(), i: 0 }
    }

    #[inline]
    pub(crate) fn terminal_ids(&self) -> Option<&Ids> {
        if self.ids.is_empty() { None } else { Some(&self.ids) }
    }

    #[inline]
    fn get_child(&self, b: u8) -> Option<usize> {
        self.edges.get_child(b)
    }

    #[inline]
    fn set_child(&mut self, b: u8, idx: usize) {
        self.edges.set_child(b, idx);
    }

    #[inline]
    fn clear_child(&mut self, b: u8) {
        self.edges.clear_child(b);
    }

    #[inline]
    fn has_children(&self) -> bool {
        !self.edges.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Encode, Decode)]
pub(crate) struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    pub(crate) fn new() -> Self {
        Self { nodes: vec![TrieNode::default()] }
    }

    #[inline]
    pub(crate) fn root_index(&self) -> usize { 0 }

    #[inline]
    pub(crate) fn node(&self, idx: usize) -> &TrieNode { &self.nodes[idx] }

    /// Look up a token; returns terminal ids if present.
    /// This is a cold path in your design; linear scans per node are fine.
    #[inline]
    pub(crate) fn get(&self, token: impl AsRef<[u8]>) -> Option<&Ids> {
        let mut idx = 0usize;
        for &b in token.as_ref() {
            idx = match self.nodes[idx].get_child(b) {
                Some(n) => n,
                None => return None,
            };
        }
        let ids = &self.nodes[idx].ids;
        if ids.is_empty() { None } else { Some(ids) }
    }

    /// Insert token id; returns true if a new id was added (dedup on ids).
    pub(crate) fn insert(&mut self, token: Token, id: TokenId) -> bool {
        let mut idx = 0usize;
        for b in token.into_iter() {
            let next = match self.nodes[idx].get_child(b) {
                Some(n) => n,
                None => {
                    let new_idx = self.nodes.len();
                    self.nodes[idx].set_child(b, new_idx);
                    self.nodes.push(TrieNode::default());
                    new_idx
                }
            };
            idx = next;
        }
        let ids = &mut self.nodes[idx].ids;
        if ids.contains(&id) { false } else { ids.push(id); true }
    }

    /// Remove a token; returns number of ids removed if any (None if not present).
    pub(crate) fn remove(&mut self, token: impl AsRef<[u8]>) -> Option<usize> {
        // Path for pruning: (parent_index, byte_taken)
        let mut path: Vec<(usize, u8)> = Vec::with_capacity(token.as_ref().len());
        let mut idx = 0usize;
        for &b in token.as_ref() {
            let next = match self.nodes[idx].get_child(b) {
                Some(n) => n,
                None => return None,
            };
            path.push((idx, b));
            idx = next;
        }

        let removed = if self.nodes[idx].ids.is_empty() {
            0
        } else {
            let n = self.nodes[idx].ids.len();
            self.nodes[idx].ids.clear();
            n
        };
        if removed == 0 { return None; }

        // prune back up while node has no children and no ids
        while let Some((parent, byte)) = path.pop() {
            // the child index at that edge may have changed only if previously pruned; re-check
            let child_idx = match self.nodes[parent].get_child(byte) {
                Some(n) => n,
                None => break,
            };
            if !self.nodes[child_idx].has_children() && self.nodes[child_idx].ids.is_empty() {
                // sever the edge; indices remain stable (no node reclamation)
                self.nodes[parent].clear_child(byte);
            } else {
                break;
            }
        }
        Some(removed)
    }

    /// Visit all terminal tokens; `f` receives (bytes, ids).
    pub(crate) fn for_each<F: FnMut(&[u8], &Ids)>(&self, mut f: F) {
        // Single shared buffer to avoid per-child clones.
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(16); // (node_idx, depth)
        let mut buf: Vec<u8> = Vec::with_capacity(32);
        stack.push((0, 0));
        while let Some((idx, depth)) = stack.pop() {
            buf.truncate(depth);
            let node = &self.nodes[idx];

            if !node.ids.is_empty() { f(&buf, &node.ids); }

            // push children in reverse so we visit ascending on pop
            for e in node.edges.iter().rev() {
                buf.push(e.b);
                stack.push((e.idx as usize, depth + 1));
                buf.pop();
            }
        }
    }
}
