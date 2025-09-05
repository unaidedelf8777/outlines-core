// Compact, immutable trie using 8:24 encoding (byte:8, token_id:24) and subtree sizes.
// Each node = 8 bytes: (bits: u32, bits2: u32)
// bits  = [ token_id:24 | byte:8 ], NO_TOKEN=0xFF_FFFF
// bits2 = [ subtree_size:24 | num_parents:8 ]
//
// Build is done through a temporary hash-trie, then serialized into a flat Vec<TrieNode>.
// Runtime traversals are branch-light and cache friendly.

use std::cmp::min;
use bincode::{Decode, Encode, BorrowDecode};

pub type TokenId = u32;

#[derive(Clone, Copy, Decode, Encode, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct TrieNode {
    bits: u32,
    bits2: u32,
}

const NO_TOKEN: u32 = 0xFF_FFFF;

impl TrieNode {
    #[inline(always)]
    pub fn new(byte: u8, token_id: u32, num_parents: u8) -> Self {
        TrieNode {
            bits: ((token_id & NO_TOKEN) << 8) | (byte as u32),
            bits2: num_parents as u32,
        }
    }
    #[inline(always)]
    pub fn byte(&self) -> u8 { (self.bits & 0xFF) as u8 }
    #[inline(always)]
    pub fn token_id(&self) -> Option<u32> {
        let t = self.bits >> 8;
        if t == NO_TOKEN { None } else { Some(t) }
    }
    #[inline(always)]
    pub fn subtree_size(&self) -> usize { (self.bits2 >> 8) as usize }
    #[inline(always)]
    pub fn num_parents(&self) -> usize { (self.bits2 & 0xFF) as usize }
}

#[derive(Clone, Default, Decode, Encode, PartialEq, Eq, Debug)]
pub struct Trie {
    nodes: Vec<TrieNode>, // nodes[0] is the artificial root
}

impl Trie {
    /// Build from a vocabulary where `words[id]` = token bytes for TokenId=id.
    pub fn from_tokens(words: &Vec<Vec<u8>>, eos_token: TokenId) -> Self {
        // Build a temporary hash-trie (sparse children) then serialize.
        let mut builder = HashTrie::new(0xFF); // root byte can be sentinel
        for (id, w) in words.iter().enumerate() {
            if !w.is_empty() {
                builder.insert(w, id as u32);
            } else {
                // Empty token still needs to exist; attach token_id to root.
                builder.token_id = id as u32;
            }
        }
        let mut nodes = Vec::with_capacity(words.len() * 2 + 1);
        builder.serialize(&mut nodes, 0);
        // Make the root always present; if builder placed root at 0, it's already fine.
        let mut trie = Trie { nodes };
        // Basic sanity: eos token should exist unless using a special scheme.
        let _ = eos_token;
        trie
    }

    #[inline] pub fn root(&self) -> &TrieNode { &self.nodes[0] }

    #[inline]
    fn node_offset(&self, n: &TrieNode) -> usize {
        let base = self.root() as *const TrieNode as usize;
        let ptr  = n as *const TrieNode as usize;
        let off = (ptr - base) / std::mem::size_of::<TrieNode>();
        off
    }

    #[inline]
    fn next_node(&self, n: &TrieNode) -> usize {
        self.node_offset(n) + n.subtree_size()
    }

    /// Iterate direct children of `n` by jumping `subtree_size()` per child.
    pub fn children<'a>(&'a self, n: &'a TrieNode) -> NodeChildren<'a> {
        let off = self.node_offset(n);
        NodeChildren {
            trie: self,
            current_offset: off + 1,
            end_offset: off + n.subtree_size(),
        }
    }

    pub fn children_from_to<'a>(&'a self, start: usize, end: usize) -> NodeChildren<'a> {
        NodeChildren {
            trie: self,
            current_offset: start,
            end_offset: end,
        }
    }

    /// Linear scan over children (already contiguous in memory).
    #[inline]
    pub fn child_at_byte<'a>(&'a self, n: &'a TrieNode, byte: u8) -> Option<&'a TrieNode> {
        let off = self.node_offset(n);
        let mut p = off + 1;
        let end = off + n.subtree_size();
        while p < end {
            let c = &self.nodes[p];
            if c.byte() == byte {
                return Some(c);
            }
            p += c.subtree_size();
        }
        None
    }

    /// Descend along a byte slice.
    #[inline]
    pub fn child_at_bytes<'a>(&'a self, mut n: &'a TrieNode, bytes: &[u8]) -> Option<&'a TrieNode> {
        for &b in bytes {
            n = match self.child_at_byte(n, b) {
                Some(c) => c,
                None => return None,
            };
        }
        Some(n)
    }

    /// Returns token id if `bytes` match a terminal exactly.
    #[inline]
    pub fn token_id_at_bytes(&self, bytes: &[u8]) -> Option<TokenId> {
        self.child_at_bytes(self.root(), bytes)
            .and_then(|n| n.token_id())
    }

    /// Compatibility helper for the old API: get a single TokenId for the token bytes.
    #[inline]
    pub fn get(&self, token: impl AsRef<[u8]>) -> Option<TokenId> {
        self.token_id_at_bytes(token.as_ref())
    }

    /// Greedy tokenizer over bytes: returns maximal token sequence.
    pub fn greedy_tokenize(&self, bytes: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::new();
        if bytes.is_empty() { return out; }
        let mut n = self.root();
        let mut last_tok: Option<TokenId> = None;
        let mut last_idx = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            match self.child_at_byte(n, bytes[i]) {
                Some(c) => {
                    if let Some(t) = c.token_id() {
                        last_tok = Some(t);
                        last_idx = i;
                    }
                    n = c;
                }
                None => {
                    out.push(last_tok.expect("input not tokenizable"));
                    i = last_idx;
                    n = self.root();
                }
            }
            i += 1;
        }
        out.push(last_tok.expect("input not tokenizable at end"));
        out
    }

    /// Check if there exists any extension of `start` that yields a token.
    pub fn has_extensions(&self, start: &[u8]) -> bool {
        let Some(n0) = self.child_at_bytes(self.root(), start) else { return false; };
        let off = self.node_offset(n0);
        let mut p = off + 1;
        let end = off + n0.subtree_size();
        while p < end {
            let n = &self.nodes[p];
            if n.token_id().is_some() {
                return true;
            }
            p += n.subtree_size();
        }
        false
    }

    /// Visit all token terminals with their byte strings (preorder, no alloc on hot path).
    pub fn for_each<F: FnMut(&[u8], TokenId)>(&self, mut f: F) {
        let mut bytes = Vec::<u8>::with_capacity(32);
        // Flat array walk: track (offset, next_pop) and mirror `bytes` length manually.
        let root = self.root();
        let off0 = self.node_offset(root);
        let mut p = off0 + 1;
        let end = off0 + root.subtree_size();
        let mut next_pop = 0usize;

        while p < end {
            // pop bytes from previous step as needed
            if next_pop != 0 {
                let keep = bytes.len().saturating_sub(next_pop);
                bytes.truncate(keep);
                next_pop = 0;
            }

            let n = &self.nodes[p];
            bytes.push(n.byte());
            if let Some(t) = n.token_id() {
                f(&bytes, t);
            }
            // If node is a leaf (subtree_size == 1), we need to pop up to its parent count next.
            next_pop = if n.subtree_size() == 1 { n.num_parents() } else { 0 };
            p += 1;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeChildren<'a> {
    trie: &'a Trie,
    current_offset: usize,
    end_offset: usize,
}

impl<'a> NodeChildren<'a> {
    pub fn new(trie: &'a Trie, current_offset: usize, end_offset: usize) -> Self {
        Self { trie, current_offset, end_offset }
    }
}
impl<'a> Iterator for NodeChildren<'a> {
    type Item = &'a TrieNode;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.end_offset { return None; }
        let n = &self.trie.nodes[self.current_offset];
        self.current_offset += n.subtree_size();
        Some(n)
    }
}

#[derive(Clone, Copy)]
pub struct TrieDecodeCtx<'a> {
    pub trie: &'a Trie,
}

impl Encode for NodeChildren<'_> {
    #[inline]
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E)
        -> Result<(), bincode::error::EncodeError>
    {
        // encode as (current_offset, end_offset)
        self.current_offset.encode(encoder)?;
        self.end_offset.encode(encoder)?;
        Ok(())
    }
}

impl<'a, C> Decode<C> for NodeChildren<'a>
where
    C: Copy, // must be TrieDecodeCtx<'a> at call site
{
    #[inline]
    fn decode<D: bincode::de::Decoder<Context = C>>(decoder: &mut D)
        -> Result<Self, bincode::error::DecodeError>
    {
        // pull offsets
        let current_offset = usize::decode(decoder)?;
        let end_offset     = usize::decode(decoder)?;

        // get &Trie from the decoder context
        let ctx: C = *decoder.context();
        // SAFETY: caller promises C == TrieDecodeCtx<'a>
        let ctx: TrieDecodeCtx<'a> = unsafe { std::mem::transmute_copy(&ctx) };

        Ok(NodeChildren { trie: ctx.trie, current_offset, end_offset })
    }
}

impl<'de, 'a, C> BorrowDecode<'de, C> for NodeChildren<'a>
where
    C: Copy, // must be TrieDecodeCtx<'a> at call site
{
    #[inline]
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(decoder: &mut D)
        -> Result<Self, bincode::error::DecodeError>
    {
        // identical to Decode: we don’t actually borrow from the input bytes
        let current_offset = usize::borrow_decode(decoder)?;
        let end_offset     = usize::borrow_decode(decoder)?;

        let ctx: C = *decoder.context();
        let ctx: TrieDecodeCtx<'a> = unsafe { std::mem::transmute_copy(&ctx) };

        Ok(NodeChildren { trie: ctx.trie, current_offset, end_offset })
    }
}

// ---------- Builder (temporary) ----------

#[derive(Clone, Decode, Encode, Debug)]
struct HashTrie {
    token_id: u32,
    byte: u8,
    children: Vec<HashTrie>, // sparse; densified only if it explodes
}

impl HashTrie {
    fn new(byte: u8) -> Self {
        Self { token_id: NO_TOKEN, byte, children: Vec::new() }
    }

    fn insert(&mut self, word: &[u8], token_id: u32) {
        if word.is_empty() {
            self.token_id = token_id; // override duplicates if present
            return;
        }
        if self.children.len() == 0x100 {
            // dense table: byte is the index
            self.children[word[0] as usize].insert(&word[1..], token_id);
            return;
        }
        // find or add child
        for ch in &mut self.children {
            if ch.byte == word[0] {
                ch.insert(&word[1..], token_id);
                return;
            }
        }
        let mut ch = HashTrie::new(word[0]);
        ch.insert(&word[1..], token_id);
        self.children.push(ch);

        // If getting too dense, expand to 256 fanout
        if self.children.len() > 250 {
            let mut v2 = (0u16..=255).map(|b| HashTrie::new(b as u8)).collect::<Vec<_>>();
            for ch in self.children.drain(..) {
                v2[ch.byte as usize] = ch.clone();
            }
            self.children = v2;
        }
    }

    /// Serialize into flat Vec<TrieNode> with subtree sizes and parent counts.
    fn serialize(&mut self, out: &mut Vec<TrieNode>, num_parents: u8) {
        let here = out.len();
        // Root byte can be sentinel; it's never read.
        out.push(TrieNode::new(self.byte, self.token_id, num_parents));

        // Ensure deterministic child order by byte
        self.children.sort_by_key(|c| c.byte);

        // The last child gets num_parents+1 so we can pop back up once for free.
        let mut remaining = self.children.len();
        for ch in &mut self.children {
            remaining -= 1;
            let np = if remaining == 0 { num_parents + 1 } else { 1 };
            ch.serialize(out, np);
        }

        // Patch subtree size (nodes written under this node, including itself)
        let subtree = (out.len() - here) as u32;
        out[here].bits2 |= subtree << 8;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ClassGroups {
    // For node at offset i:
    // - groups for node are at bounds[2*idx .. 2*(idx+cnt)]
    group_idx: Vec<u32>,     // len == trie.nodes.len()
    group_cnt: Vec<u16>,     // len == trie.nodes.len()
    bounds:    Vec<u32>,     // packed [start_off, end_off) pairs
    group_class: Vec<u8>,    // class id per group, same order as bounds pairs
}

impl ClassGroups {
    pub fn build(trie: &Trie, byte_to_class: &[u8; 256]) -> Self {
        let n_nodes = trie.nodes.len();
        let mut group_idx = vec![0u32; n_nodes];
        let mut group_cnt = vec![0u16; n_nodes];
        let mut bounds: Vec<u32> = Vec::new();
        let mut group_class: Vec<u8> = Vec::new();

        // Walk all nodes; children segment is [off+1, off+subtree)
        for node_off in 0..n_nodes {
            let node = &trie.nodes[node_off];
            let start = node_off + 1;
            let end   = node_off + node.subtree_size();

            if start >= end {
                // leaf
                group_idx[node_off] = (bounds.len() / 2) as u32;
                group_cnt[node_off] = 0;
                continue;
            }

            let mut cur = start;
            let idx0 = (bounds.len() / 2) as u32; // start index for this node

            // children are byte-sorted; classes are contiguous ranges
            while cur < end {
                let first = &trie.nodes[cur];
                let cls = byte_to_class[first.byte() as usize];
                let mut p = cur;
                // advance until class changes
                while p < end {
                    let n = &trie.nodes[p];
                    let n_cls = byte_to_class[n.byte() as usize];
                    if n_cls != cls { break; }
                    p += n.subtree_size();
                }
                // record one group
                bounds.push(cur as u32);
                bounds.push(p as u32);
                group_class.push(cls);

                cur = p;
            }

            let cnt = ((bounds.len() / 2) as u32 - idx0) as u16;
            group_idx[node_off] = idx0;
            group_cnt[node_off] = cnt;
        }

        Self { group_idx, group_cnt, bounds, group_class }
    }

    #[inline]
    pub fn groups<'a>(&'a self, trie: &'a Trie, node: &'a TrieNode)
        -> ClassGroupIter<'a>
    {
        let off = trie.node_offset(node);
        let idx = self.group_idx[off] as usize;
        let cnt = self.group_cnt[off] as usize;
        ClassGroupIter {
            trie,
            bounds: &self.bounds[2*idx .. 2*(idx+cnt)],
            classes: &self.group_class[idx .. idx+cnt],
            i: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClassChildGroup<'a> {
    pub rep: &'a TrieNode,     // first child in the group
    pub class: u8,             // byte class id
    pub start_off: usize,      // node offset of first child
    pub end_off: usize,        // exclusive
}

pub struct ClassGroupIter<'a> {
    trie: &'a Trie,
    bounds: &'a [u32],     // len is 2*count
    classes: &'a [u8],     // len is count
    i: usize,
}

impl<'a> Iterator for ClassGroupIter<'a> {
    type Item = ClassChildGroup<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.classes.len() { return None; }
        let b0 = self.bounds[2*self.i] as usize;
        let b1 = self.bounds[2*self.i + 1] as usize;
        let cls = self.classes[self.i];
        let rep = &self.trie.nodes[b0];
        self.i += 1;
        Some(ClassChildGroup { rep, class: cls, start_off: b0, end_off: b1 })
    }
}
