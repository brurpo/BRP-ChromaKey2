//! CLIP byte-level BPE tokenizer, as SAM 3's `tokenizer_ve.SimpleTokenizer` (clean "lower").
//!
//! The merge list is `text/bpe_merges.txt` of the SAM 3.1 pack (upstream's
//! `bpe_simple_vocab_16e6.txt.gz` lines 1..=48894, uncompressed). Not ported: `ftfy.fix_text` and
//! HTML unescaping, which only matter for mojibake or HTML entities in typed prompts.
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

pub struct ClipTokenizer {
    encoder: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    start: i64,
    end: i64,
}

/// `bytes_to_unicode`: printable bytes map to themselves, the rest to 256 + n.
fn bytes_to_unicode() -> ([char; 256], Vec<char>) {
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars = bytes.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    let mut order = Vec::with_capacity(256);
    for (&b, &c) in bytes.iter().zip(&chars) {
        let c = char::from_u32(c).expect("valid code point");
        table[b as usize] = c;
        order.push(c);
    }
    (table, order)
}

impl ClipTokenizer {
    /// `merges`: one "first second" pair per line, in rank order.
    pub fn from_merges(merges: &str) -> Result<Self> {
        let (byte_encoder, order) = bytes_to_unicode();
        let pairs: Vec<(String, String)> = merges
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let mut parts = line.split(' ');
                Ok((
                    parts.next().context("empty BPE merge")?.to_owned(),
                    parts
                        .next()
                        .context("BPE merge needs two symbols")?
                        .to_owned(),
                ))
            })
            .collect::<Result<_>>()?;
        let mut vocab: Vec<String> = order.iter().map(|c| c.to_string()).collect();
        vocab.extend(order.iter().map(|c| format!("{c}</w>")));
        vocab.extend(pairs.iter().map(|(a, b)| format!("{a}{b}")));
        vocab.push("<start_of_text>".into());
        vocab.push("<end_of_text>".into());
        let encoder: HashMap<String, i64> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), i as i64))
            .collect();
        ensure!(
            encoder.len() == vocab.len(),
            "duplicate CLIP vocabulary entries"
        );
        let start = encoder["<start_of_text>"];
        let end = encoder["<end_of_text>"];
        let ranks = pairs
            .into_iter()
            .enumerate()
            .map(|(i, pair)| (pair, i))
            .collect();
        Ok(Self {
            encoder,
            ranks,
            byte_encoder,
            start,
            end,
        })
    }

    /// `[start] + encode(text) + [end]`, truncated (keeping the end token) or zero-padded to `context`.
    pub fn tokenize(&self, text: &str, context: usize) -> Result<Vec<i64>> {
        let mut tokens = vec![self.start];
        tokens.extend(self.encode(text)?);
        tokens.push(self.end);
        if tokens.len() > context {
            tokens.truncate(context);
            tokens[context - 1] = self.end;
        }
        tokens.resize(context, 0);
        Ok(tokens)
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let cleaned = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let mut ids = Vec::new();
        for piece in pre_tokenize(&cleaned) {
            let mapped: String = piece
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            for symbol in self.bpe(&mapped) {
                ids.push(*self.encoder.get(&symbol).with_context(|| {
                    format!("BPE symbol {symbol:?} not in the CLIP vocabulary")
                })?);
            }
        }
        Ok(ids)
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        while word.len() > 1 {
            let best = word
                .windows(2)
                .filter_map(|pair| {
                    self.ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|r| (*r, pair))
                })
                .min_by_key(|(rank, _)| *rank)
                .map(|(_, pair)| (pair[0].clone(), pair[1].clone()));
            let Some((first, second)) = best else {
                break;
            };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }
}

/// `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+` on lowercased text.
fn pre_tokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '\''
            && let Some(suffix) = CONTRACTIONS.iter().find(|s| {
                let s: Vec<char> = s.chars().collect();
                chars.get(i + 1..i + 1 + s.len()) == Some(&s[..])
            })
        {
            out.push(format!("'{suffix}"));
            i += 1 + suffix.chars().count();
            continue;
        }
        let start = i;
        if c.is_alphabetic() {
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
        } else if c.is_numeric() {
            i += 1;
        } else {
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tokenizer_matches_the_clip_pattern() {
        assert_eq!(
            pre_tokenize("red car's wheel!"),
            ["red", "car", "'s", "wheel", "!"]
        );
        assert_eq!(pre_tokenize("two dogs 42"), ["two", "dogs", "4", "2"]);
        assert_eq!(
            pre_tokenize("well-known <x>"),
            ["well", "-", "known", "<", "x", ">"]
        );
        assert_eq!(pre_tokenize("!'s 'x"), ["!'", "s", "'", "x"]);
    }

    #[test]
    fn small_vocabulary_merges_by_rank() {
        // Ranks: "h a" then "ha t</w>" -> "hat</w>" is one token.
        let tokenizer = ClipTokenizer::from_merges("h a\nha t</w>\n").unwrap();
        let tokens = tokenizer.tokenize("Hat", 6).unwrap();
        let hat = tokenizer.encoder["hat</w>"];
        assert_eq!(tokens, vec![tokenizer.start, hat, tokenizer.end, 0, 0, 0]);
        // Truncation keeps the end token.
        let long = tokenizer.tokenize("hat hat hat hat", 4).unwrap();
        assert_eq!(long, vec![tokenizer.start, hat, hat, tokenizer.end]);
    }

    /// Upstream token ids (sam31_text_assets.py) when the development pack is present.
    #[test]
    fn matches_upstream_fixtures_when_available() {
        let dir = std::path::Path::new("D:/BRP_CHROMA_2/.dev/sam3/onnx/text");
        let (Ok(merges), Ok(fixtures)) = (
            std::fs::read_to_string(dir.join("bpe_merges.txt")),
            std::fs::read_to_string(dir.join("token_fixtures.json")),
        ) else {
            eprintln!("SAM 3.1 text assets not present; skipping upstream comparison");
            return;
        };
        let tokenizer = ClipTokenizer::from_merges(&merges).unwrap();
        let fixtures: HashMap<String, Vec<i64>> = serde_json::from_str(&fixtures).unwrap();
        for (text, expected) in fixtures {
            assert_eq!(tokenizer.tokenize(&text, 32).unwrap(), expected, "{text:?}");
        }
    }
}
