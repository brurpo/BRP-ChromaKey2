//! Transparent prompt translation for SAM 3.1 text prompts.
//!
//! SAM 3.1's text encoder only understands English. A resident worker process
//! (`ck-native translate --serve`: lingua English/Portuguese detection + Opus-MT ROMANCE->English
//! INT8 on the CPU) translates each committed prompt once; jobs receive the English text, the
//! project keeps what the user typed, and the UI only shows a small read-only "-> english" line.
//! The helper starts on first use and exits when the app closes (its stdin closes).

use ck_desktop_core::RuntimePaths;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
pub enum Translated {
    /// English to send to SAM 3.1; `changed` is false when the text already was English.
    Done {
        english: String,
        changed: bool,
    },
    Failed,
}

#[derive(Default)]
struct Shared {
    results: HashMap<String, Translated>,
    pending: HashSet<String>,
    dead: bool,
}

pub struct TranslationService {
    runtime: RuntimePaths,
    shared: Arc<(Mutex<Shared>, Condvar)>,
    process: Option<(Child, ChildStdin)>,
    repaint: Option<eframe::egui::Context>,
}

impl TranslationService {
    pub fn new(runtime: RuntimePaths) -> Self {
        Self {
            runtime,
            shared: Arc::new((Mutex::new(Shared::default()), Condvar::new())),
            process: None,
            repaint: None,
        }
    }

    pub fn set_repaint(&mut self, ctx: &eframe::egui::Context) {
        if self.repaint.is_none() {
            self.repaint = Some(ctx.clone());
        }
    }

    pub fn available(&self) -> bool {
        self.runtime.translator_available() && self.runtime.worker.is_file()
    }

    fn key(text: &str) -> String {
        text.trim().to_owned()
    }

    /// Cached result for `text`, if any.
    pub fn get(&self, text: &str) -> Option<Translated> {
        let key = Self::key(text);
        if key.is_empty() {
            return None;
        }
        self.shared.0.lock().ok()?.results.get(&key).cloned()
    }

    /// Asks for `text` in the background (no-op when cached, pending or unavailable).
    pub fn request(&mut self, text: &str) {
        let key = Self::key(text);
        if key.is_empty() || !self.available() {
            return;
        }
        {
            let Ok(mut shared) = self.shared.0.lock() else {
                return;
            };
            if shared.dead
                || shared.results.contains_key(&key)
                || !shared.pending.insert(key.clone())
            {
                return;
            }
        }
        let line = serde_json::json!({ "text": key }).to_string();
        let sent = self.ensure_process().and_then(|stdin| {
            writeln!(stdin, "{line}")?;
            stdin.flush()?;
            Ok(())
        });
        if sent.is_err() {
            self.process = None;
            if let Ok(mut shared) = self.shared.0.lock() {
                shared.pending.remove(&key);
                shared.results.insert(key, Translated::Failed);
            }
        }
    }

    /// The English for a job: waits briefly for the helper; falls back to the text itself.
    pub fn english_for_job(&mut self, text: &str) -> String {
        let key = Self::key(text);
        if key.is_empty() || !self.available() {
            return key;
        }
        self.request(&key);
        let (lock, ready) = &*self.shared;
        let Ok(mut shared) = lock.lock() else {
            return key;
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while shared.pending.contains(&key) && !shared.dead {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                break;
            }
            shared = match ready.wait_timeout(shared, left) {
                Ok((guard, _)) => guard,
                Err(_) => return key,
            };
        }
        match shared.results.get(&key) {
            Some(Translated::Done { english, .. }) if !english.is_empty() => english.clone(),
            _ => key,
        }
    }

    fn ensure_process(&mut self) -> std::io::Result<&mut ChildStdin> {
        if self.process.is_none() {
            let mut command = Command::new(&self.runtime.worker);
            command
                .arg("translate")
                .arg("--serve")
                .arg("--models")
                .arg(self.runtime.translate_models())
                .arg("--runtime")
                .arg(&self.runtime.ort_library)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
            }
            let mut child = command.spawn()?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| std::io::Error::other("no stdin"))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| std::io::Error::other("no stdout"))?;
            let shared = self.shared.clone();
            let repaint = self.repaint.clone();
            std::thread::spawn(move || read_replies(stdout, &shared, repaint));
            self.process = Some((child, stdin));
        }
        Ok(&mut self.process.as_mut().expect("spawned above").1)
    }
}

fn read_replies(
    stdout: std::process::ChildStdout,
    shared: &Arc<(Mutex<Shared>, Condvar)>,
    repaint: Option<eframe::egui::Context>,
) {
    let (lock, ready) = &**shared;
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        let Ok(reply) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let Some(text) = reply.get("text").and_then(|t| t.as_str()) else {
            continue; // the ready line
        };
        let result = match (
            reply.get("english").and_then(|e| e.as_str()),
            reply.get("error"),
        ) {
            (Some(english), None) => Translated::Done {
                english: english.to_owned(),
                changed: reply
                    .get("translated")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false),
            },
            _ => Translated::Failed,
        };
        if let Ok(mut s) = lock.lock() {
            s.pending.remove(text);
            s.results.insert(text.to_owned(), result);
        }
        ready.notify_all();
        if let Some(ctx) = &repaint {
            ctx.request_repaint();
        }
    }
    // Helper exited: release anyone waiting; later requests fall back to the typed text.
    if let Ok(mut s) = lock.lock() {
        s.dead = true;
        for key in std::mem::take(&mut s.pending) {
            s.results.insert(key, Translated::Failed);
        }
    }
    ready.notify_all();
}

impl Drop for TranslationService {
    fn drop(&mut self) {
        if let Some((mut child, stdin)) = self.process.take() {
            drop(stdin); // EOF ends the helper's loop
            if child.wait_timeout_ok().is_none() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

trait WaitBriefly {
    fn wait_timeout_ok(&mut self) -> Option<std::process::ExitStatus>;
}

impl WaitBriefly for Child {
    fn wait_timeout_ok(&mut self) -> Option<std::process::ExitStatus> {
        for _ in 0..20 {
            if let Ok(Some(status)) = self.try_wait() {
                return Some(status);
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        None
    }
}

/// The read-only line shown under a prompt: `-> english` when it was translated.
pub fn caption(service: &TranslationService, text: &str) -> Option<String> {
    if text.trim().is_empty() || !service.available() {
        return None;
    }
    match service.get(text) {
        // Only when SAM 3.1 receives something other than what was typed.
        Some(Translated::Done {
            english,
            changed: true,
        }) if english.to_lowercase() != text.trim().to_lowercase() => Some(format!("→ {english}")),
        Some(_) => None,
        None => Some("→ …".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_is_hidden_without_a_translator() {
        let mut runtime = RuntimePaths::installed(std::path::Path::new("Z:/missing-app"));
        runtime.worker = "Z:/missing-app/ck-native.exe".into();
        let mut service = TranslationService::new(runtime);
        assert!(!service.available());
        service.request("chapéu");
        assert_eq!(caption(&service, "chapéu"), None);
        assert_eq!(service.english_for_job("  chapéu "), "chapéu");
    }

    /// Uses the development test app's worker and translator pack; skipped when absent.
    #[test]
    fn translates_portuguese_through_the_resident_helper() {
        let app = std::path::Path::new("D:/BRP_CHROMA_2/app-native-test");
        let runtime = RuntimePaths::installed(app);
        if !runtime.translator_available() || !runtime.worker.is_file() {
            return;
        }
        let mut service = TranslationService::new(runtime);
        assert_eq!(service.english_for_job("chapéu colorido"), "colorful hat");
        assert_eq!(service.english_for_job("sapato"), "shoe");
        assert_eq!(service.english_for_job("person"), "person");
        assert_eq!(
            caption(&service, "chapéu colorido").as_deref(),
            Some("→ colorful hat")
        );
        assert_eq!(caption(&service, "person"), None);
    }
}
