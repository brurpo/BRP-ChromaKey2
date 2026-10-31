//! Prompt translation for SAM 3.1 text prompts (its text encoder only understands English).
//!
//! Language detection: lingua with only the English and Portuguese models (a two-way decision;
//! more Romance languages would split short prompts' votes). Portuguese goes through Opus-MT
//! ROMANCE->English (Marian, Apache-2.0), exported to
//! INT8 ONNX by `tools/model_export/export_opus_mt.py` and run on the ONNX Runtime CPU provider
//! with greedy decoding. Short prompts are wrapped as "Na imagem: <prompt>." so the sentence-trained
//! model translates single words reliably ("sapato" -> "shoe", "bengala" -> "cane"); the wrapper
//! is removed from the output, then articles and punctuation are stripped.
//!
//! `ck-native translate --models DIR [--runtime DLL] --text "..."` translates once;
//! `--serve` keeps the models resident and answers one JSON line per input line:
//! `{"text": "..."}` -> `{"text", "english", "language", "translated", "ms"}` or `{"text", "error"}`.

use anyhow::{Context, Result, bail, ensure};
use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use ndarray::{Array2, ArrayD};
use prost::Message;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

const WORD_BOUNDARY: char = '\u{2581}';
const UNKNOWN_PENALTY: f32 = 10.0;
const WRAP_PREFIX: &str = "Na imagem: ";
const ENGLISH_THRESHOLD: f64 = 0.9;
const UNWRAP_PREFIX: &str = "in the image:";

// ----- SentencePiece model (only the fields the unigram encoder needs)

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SentencePiece>,
    #[prost(message, optional, tag = "3")]
    normalizer_spec: Option<NormalizerSpec>,
}

#[derive(Clone, PartialEq, Message)]
struct SentencePiece {
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    #[prost(float, optional, tag = "2")]
    score: Option<f32>,
    /// 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.
    #[prost(int32, optional, tag = "3")]
    kind: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct NormalizerSpec {
    #[prost(string, optional, tag = "1")]
    name: Option<String>,
    #[prost(bool, optional, tag = "3")]
    add_dummy_prefix: Option<bool>,
    #[prost(bool, optional, tag = "4")]
    remove_extra_whitespaces: Option<bool>,
    #[prost(bool, optional, tag = "5")]
    escape_whitespaces: Option<bool>,
}

/// Unigram SentencePiece segmentation (Viterbi), as `SentencePieceProcessor.encode(out_type=str)`.
pub struct Unigram {
    pieces: HashMap<String, f32>,
    max_chars: usize,
    unknown_score: f32,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

impl Unigram {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let model = ModelProto::decode(bytes).context("reading SentencePiece model")?;
        let spec = model.normalizer_spec.unwrap_or_default();
        ensure!(
            matches!(spec.name.as_deref(), None | Some("nmt_nfkc") | Some("nfkc")),
            "unsupported SentencePiece normalizer {:?}",
            spec.name
        );
        let mut pieces = HashMap::new();
        let mut min_score = f32::MAX;
        let mut max_chars = 1;
        for p in &model.pieces {
            let kind = p.kind.unwrap_or(1);
            let (Some(piece), score) = (p.piece.clone(), p.score.unwrap_or(0.0)) else {
                continue;
            };
            if kind == 1 || kind == 4 {
                min_score = min_score.min(score);
                max_chars = max_chars.max(piece.chars().count());
                pieces.insert(piece, score);
            }
        }
        ensure!(!pieces.is_empty(), "SentencePiece model has no pieces");
        Ok(Self {
            pieces,
            max_chars,
            unknown_score: min_score - UNKNOWN_PENALTY,
            add_dummy_prefix: spec.add_dummy_prefix.unwrap_or(true),
            remove_extra_whitespaces: spec.remove_extra_whitespaces.unwrap_or(true),
            escape_whitespaces: spec.escape_whitespaces.unwrap_or(true),
        })
    }

    /// NFKC, whitespace cleanup, dummy prefix and the word-boundary marker (`nmt_nfkc`).
    fn normalize(&self, text: &str) -> String {
        let nfkc: String = text
            .nfkc()
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .filter(|c| !c.is_control())
            .collect();
        let mut s = if self.remove_extra_whitespaces {
            nfkc.split(' ')
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            nfkc
        };
        if s.is_empty() {
            return s;
        }
        if self.add_dummy_prefix {
            s.insert(0, ' ');
        }
        if self.escape_whitespaces {
            s = s.replace(' ', &WORD_BOUNDARY.to_string());
        }
        s
    }

    pub fn encode(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = self.normalize(text).chars().collect();
        let n = chars.len();
        // best[i] = (score, start of the last piece) for the prefix of i characters.
        let mut best: Vec<(f32, usize)> = vec![(f32::NEG_INFINITY, 0); n + 1];
        best[0] = (0.0, 0);
        for start in 0..n {
            let base = best[start].0;
            if base == f32::NEG_INFINITY {
                continue;
            }
            let mut piece = String::new();
            let mut covered = false;
            for end in start + 1..=(start + self.max_chars).min(n) {
                piece.push(chars[end - 1]);
                if let Some(&score) = self.pieces.get(&piece) {
                    covered |= end == start + 1;
                    if base + score > best[end].0 {
                        best[end] = (base + score, start);
                    }
                }
            }
            // A character no piece starts with becomes a single unknown piece.
            if !covered && base + self.unknown_score > best[start + 1].0 {
                best[start + 1] = (base + self.unknown_score, start);
            }
        }
        let mut out = Vec::new();
        let mut end = n;
        while end > 0 {
            let start = best[end].1;
            out.push(chars[start..end].iter().collect());
            end = start;
        }
        out.reverse();
        out
    }
}

// ----- Marian translator

#[derive(Deserialize)]
struct MarianConfig {
    pad: i64,
    eos: i64,
    unk: i64,
    decoder_start: i64,
    max_target_tokens: usize,
}

pub struct Translator {
    source: Unigram,
    vocab: HashMap<String, i64>,
    pieces: Vec<String>,
    config: MarianConfig,
    encoder: ort::session::Session,
    decoder: ort::session::Session,
    detector: LanguageDetector,
    /// Text is left untranslated only when lingua is at least this confident it is English:
    /// a missed Portuguese prompt finds nothing, while translating English is close to identity.
    english_threshold: f64,
}

/// One translated prompt.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Translation {
    pub text: String,
    pub english: String,
    pub language: String,
    /// lingua's confidence that the text is English (0..1).
    pub english_confidence: f64,
    pub translated: bool,
    pub ms: f64,
}

impl Translator {
    /// Loads the detector and the translator (`dir` = the exported Opus-MT folder). ONNX Runtime
    /// must already be initialised.
    pub fn load(dir: &Path, args: &crate::Args) -> Result<Self> {
        let source = Unigram::from_bytes(&std::fs::read(dir.join("source.spm"))?)?;
        let vocab: HashMap<String, i64> =
            serde_json::from_slice(&std::fs::read(dir.join("vocab.json"))?)
                .context("reading the Marian vocabulary")?;
        let size = vocab.values().copied().max().unwrap_or(0) as usize + 1;
        let mut pieces = vec![String::new(); size];
        for (piece, &id) in &vocab {
            pieces[id as usize] = piece.clone();
        }
        let config: MarianConfig =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let cpu = crate::Args {
            provider: "cpu".into(),
            threads: 2,
            ..args.clone()
        };
        let session = |name: &str| -> Result<ort::session::Session> {
            crate::create_session_with(&cpu, &dir.join(name), None, &[])
                .with_context(|| format!("loading translator {name}"))
        };
        let detector =
            LanguageDetectorBuilder::from_languages(&[Language::English, Language::Portuguese])
                .with_preloaded_language_models()
                .build();
        Ok(Self {
            source,
            vocab,
            pieces,
            encoder: session("encoder.onnx")?,
            decoder: session("decoder.onnx")?,
            config,
            detector,
            english_threshold: ENGLISH_THRESHOLD,
        })
    }

    /// Marian source ids: SentencePiece pieces looked up in the shared vocabulary, then `</s>`.
    pub fn source_ids(&self, text: &str) -> Vec<i64> {
        let mut ids: Vec<i64> = self
            .source
            .encode(text)
            .iter()
            .map(|p| self.vocab.get(p).copied().unwrap_or(self.config.unk))
            .collect();
        ids.push(self.config.eos);
        ids
    }

    /// Greedy decoding of `text` (no wrapper, no clean-up).
    pub fn translate_raw(&mut self, text: &str) -> Result<String> {
        use ort::value::Tensor;
        let ids = self.source_ids(text);
        let n = ids.len();
        let input = Array2::from_shape_vec((1, n), ids)?;
        let mask = Array2::<i64>::ones((1, n));
        let hidden: ArrayD<f32> = {
            let out = self.encoder.run(ort::inputs![
                "input_ids" => Tensor::from_array(input)?,
                "attention_mask" => Tensor::from_array(mask.clone())?,
            ])?;
            out["hidden"].try_extract_array::<f32>()?.to_owned()
        };
        let mut target = vec![self.config.decoder_start];
        for _ in 0..self.config.max_target_tokens {
            let t = target.len();
            let out = self.decoder.run(ort::inputs![
                "decoder_input_ids" => Tensor::from_array(Array2::from_shape_vec((1, t), target.clone())?)?,
                "encoder_hidden_states" => Tensor::from_array(hidden.clone())?,
                "encoder_attention_mask" => Tensor::from_array(mask.clone())?,
            ])?;
            let logits = out["logits"].try_extract_array::<f32>()?;
            // The pad token is never a valid output (as in Marian generation).
            let next = logits
                .iter()
                .enumerate()
                .filter(|(i, _)| *i as i64 != self.config.pad)
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as i64)
                .context("empty translator logits")?;
            target.push(next);
            if next == self.config.eos {
                break;
            }
        }
        Ok(self.detokenize(&target[1..]))
    }

    fn detokenize(&self, ids: &[i64]) -> String {
        let special = [self.config.pad, self.config.eos, self.config.unk];
        let text: String = ids
            .iter()
            .filter(|id| !special.contains(id))
            .filter_map(|&id| self.pieces.get(id as usize))
            .map(String::as_str)
            .collect();
        text.replace(WORD_BOUNDARY, " ").trim().to_owned()
    }

    /// Detected language of a prompt: "en", "pt" or "unknown".
    pub fn language(&self, text: &str) -> &'static str {
        match self.detector.detect_language_of(text) {
            Some(Language::English) => "en",
            Some(Language::Portuguese) => "pt",
            _ => "unknown",
        }
    }

    /// English prompt for SAM 3.1: unchanged when the text is English (or has no letters),
    /// otherwise translated with the image wrapper and cleaned up.
    pub fn translate(&mut self, text: &str) -> Result<Translation> {
        self.translate_with(text, false)
    }

    /// `force` translates even confident English (developer checks).
    pub fn translate_with(&mut self, text: &str, force: bool) -> Result<Translation> {
        let started = std::time::Instant::now();
        let trimmed = text.trim();
        let has_letters = trimmed.chars().any(char::is_alphabetic);
        let (language, english_confidence) = if has_letters {
            (
                self.language(trimmed),
                self.detector
                    .compute_language_confidence(trimmed, Language::English),
            )
        } else {
            ("unknown", 0.0)
        };
        let translated = has_letters && (force || english_confidence < self.english_threshold);
        let english = if translated {
            let wrapped = self.translate_raw(&format!("{WRAP_PREFIX}{trimmed}."))?;
            match unwrap_prompt(&wrapped) {
                Some(prompt) => prompt,
                // The model rephrased the wrapper: fall back to the bare prompt.
                None => clean_prompt(&self.translate_raw(trimmed)?),
            }
        } else {
            trimmed.to_owned()
        };
        Ok(Translation {
            text: text.to_owned(),
            english,
            language: language.to_owned(),
            english_confidence,
            translated,
            ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// "In the image: a red car." -> "red car"; `None` when the wrapper is not recognisable.
pub fn unwrap_prompt(output: &str) -> Option<String> {
    let lower = output.trim().to_lowercase();
    let rest = lower.strip_prefix(UNWRAP_PREFIX)?;
    let prompt = clean_prompt(rest);
    (!prompt.is_empty()).then_some(prompt)
}

/// Lowercase, trim punctuation and quotes, drop a leading article.
pub fn clean_prompt(text: &str) -> String {
    let mut t = text
        .trim()
        .to_lowercase()
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace() || "“”«»".contains(c))
        .to_owned();
    for article in ["a ", "an ", "the "] {
        if let Some(rest) = t.strip_prefix(article) {
            t = rest.trim().to_owned();
            break;
        }
    }
    t
}

/// Default translator folder: `models/translate/opus-mt-romance-en` beside the worker.
pub fn default_models() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let dir = exe.parent().context("worker path without folder")?;
    Ok(dir
        .join("models")
        .join("translate")
        .join("opus-mt-romance-en"))
}

pub fn run_cli(raw: &[String]) -> Result<()> {
    let mut models = None;
    let mut runtime = None;
    let mut text = None;
    let mut serve = false;
    let mut it = raw.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--serve" => serve = true,
            "--models" => models = it.next().map(PathBuf::from),
            "--runtime" => runtime = it.next().map(PathBuf::from),
            "--text" => text = it.next().cloned(),
            other => bail!("unknown translate argument {other}"),
        }
    }
    let models = match models {
        Some(m) => m,
        None => default_models()?,
    };
    let args = crate::Args {
        command: "translate".into(),
        runtime,
        provider: "cpu".into(),
        threads: 2,
        ort_optimization: "all".into(),
        ..Default::default()
    };
    crate::initialize_runtime(&args)?;
    let loading = std::time::Instant::now();
    let mut translator = Translator::load(&models, &args)?;
    let load_ms = loading.elapsed().as_secs_f64() * 1000.0;
    if !serve {
        let text = text.context("give --text or --serve")?;
        let mut out = serde_json::to_value(translator.translate(&text)?)?;
        out["load_ms"] = load_ms.into();
        println!("{out}");
        return Ok(());
    }
    let stdout = std::io::stdout();
    writeln!(
        stdout.lock(),
        "{}",
        serde_json::json!({"ready": true, "load_ms": load_ms})
    )?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        // Some shells prefix piped text with a byte-order mark.
        let line = line.trim_start_matches('\u{feff}');
        if line.trim().is_empty() {
            continue;
        }
        #[derive(Deserialize)]
        struct Request {
            text: String,
            #[serde(default)]
            force: bool,
        }
        let reply = match serde_json::from_str::<Request>(line) {
            Ok(request) => match translator.translate_with(&request.text, request.force) {
                Ok(t) => serde_json::to_value(t)?,
                Err(e) => serde_json::json!({"text": request.text, "error": format!("{e:#}")}),
            },
            Err(e) => serde_json::json!({"error": format!("bad request: {e}")}),
        };
        let mut out = stdout.lock();
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "D:/BRP_CHROMA_2/.dev/translate/opus-mt-romance-en";

    #[test]
    fn clean_and_unwrap() {
        assert_eq!(
            unwrap_prompt("In the image: a red car."),
            Some("red car".into())
        );
        assert_eq!(
            unwrap_prompt("In the image: the man on the left."),
            Some("man on the left".into())
        );
        assert_eq!(unwrap_prompt("A picture of a shoe."), None);
        assert_eq!(clean_prompt("  The Colorful Hat! "), "colorful hat");
        assert_eq!(clean_prompt("an umbrella"), "umbrella");
        assert_eq!(clean_prompt("“cane”"), "cane");
    }

    /// Token ids match the Python MarianTokenizer (development fixture, skipped when absent).
    #[test]
    fn source_ids_match_marian_tokenizer() {
        let dir = Path::new(DIR);
        let Ok(fixtures) = std::fs::read(dir.join("token_fixtures.json")) else {
            return;
        };
        #[derive(Deserialize)]
        struct Row {
            text: String,
            ids: Vec<i64>,
        }
        let rows: Vec<Row> = serde_json::from_slice(&fixtures).unwrap();
        let source = Unigram::from_bytes(&std::fs::read(dir.join("source.spm")).unwrap()).unwrap();
        let vocab: HashMap<String, i64> =
            serde_json::from_slice(&std::fs::read(dir.join("vocab.json")).unwrap()).unwrap();
        let mut failures = Vec::new();
        for row in &rows {
            let mut ids: Vec<i64> = source
                .encode(&row.text)
                .iter()
                .map(|p| vocab.get(p).copied().unwrap_or(1))
                .collect();
            ids.push(0);
            if ids != row.ids {
                failures.push((row.text.clone(), ids, row.ids.clone()));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} differ: {failures:?}",
            failures.len(),
            rows.len()
        );
    }
}
