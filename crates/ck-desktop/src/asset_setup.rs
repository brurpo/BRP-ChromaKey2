use ck_desktop_core::{
    AssetCancellation, AssetGroup, AssetManager, AssetPhase, AssetProgress, AssetStatus,
};
use eframe::egui::{self, Color32, RichText};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
};

pub const STARTUP_GROUPS: [AssetGroup; 5] = [
    AssetGroup::Core,
    AssetGroup::Runtime,
    AssetGroup::Ck1080,
    AssetGroup::Sam,
    AssetGroup::Transnet,
];

enum Command {
    Inspect(Vec<AssetGroup>),
    Install(Vec<AssetGroup>, AssetCancellation),
}

enum Event {
    Progress(AssetProgress),
    Finished {
        groups: Vec<AssetGroup>,
        result: Result<AssetStatus, String>,
        cancelled: bool,
    },
}

pub struct AssetSetup {
    directory: PathBuf,
    commands: SyncSender<Command>,
    events: Receiver<Event>,
    required: Vec<AssetGroup>,
    verified: Vec<AssetGroup>,
    status: Option<AssetStatus>,
    progress: Option<AssetProgress>,
    cancellation: Option<AssetCancellation>,
    busy: bool,
    downloading: bool,
    inspect_after: bool,
    changed: bool,
    error: Option<String>,
    cancelled: bool,
    pub open: bool,
}

impl AssetSetup {
    pub fn new(directory: PathBuf, repaint: egui::Context) -> Self {
        let (commands, requests) = mpsc::sync_channel::<Command>(1);
        let (sender, events) = mpsc::sync_channel::<Event>(32);
        let folder = directory.clone();
        thread::spawn(move || {
            let mut manager: Option<AssetManager> = None;
            while let Ok(command) = requests.recv() {
                let (groups, cancellation) = match command {
                    Command::Inspect(groups) => (groups, None),
                    Command::Install(groups, cancellation) => (groups, Some(cancellation)),
                };
                let result = (|| -> anyhow::Result<AssetStatus> {
                    if manager.is_none() {
                        manager = Some(AssetManager::from_manifest_file(
                            folder.clone(),
                            folder.join("assets-manifest.json"),
                        )?);
                    }
                    if cancellation.is_some()
                        && let Some(token) =
                            ck_desktop_core::read_huggingface_token().ok().flatten()
                    {
                        manager = manager
                            .take()
                            .map(|value| value.with_huggingface_token(token));
                    }
                    let manager = manager.as_mut().expect("manager was initialized");
                    match &cancellation {
                        None => manager.inspect(&groups),
                        Some(cancellation) => manager.install(&groups, cancellation, |progress| {
                            let _ = sender.try_send(Event::Progress(progress));
                            repaint.request_repaint();
                        }),
                    }
                })()
                .map_err(|error| format!("{error:#}"));
                let cancelled = cancellation
                    .as_ref()
                    .is_some_and(|value| value.is_cancelled());
                if sender
                    .send(Event::Finished {
                        groups,
                        result,
                        cancelled,
                    })
                    .is_err()
                {
                    break;
                }
                repaint.request_repaint();
            }
        });
        let mut setup = Self {
            directory,
            commands,
            events,
            required: STARTUP_GROUPS.to_vec(),
            verified: vec![],
            status: None,
            progress: None,
            cancellation: None,
            busy: false,
            downloading: false,
            inspect_after: false,
            changed: false,
            error: None,
            cancelled: false,
            open: true,
        };
        setup.inspect();
        setup
    }

    pub fn ready_for(&self, groups: &[AssetGroup]) -> bool {
        groups.iter().all(|group| self.verified.contains(group))
    }

    pub fn require(&mut self, groups: &[AssetGroup]) -> bool {
        if self.ready_for(groups) {
            return true;
        }
        let mut added = false;
        for group in groups {
            if !self.required.contains(group) {
                self.required.push(*group);
                added = true;
            }
        }
        self.open = true;
        if added {
            if self.busy {
                self.inspect_after = true;
            } else {
                self.inspect();
            }
        }
        false
    }

    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    pub fn recheck(&mut self) {
        self.open = true;
        self.verified.clear();
        self.inspect();
    }

    fn inspect(&mut self) {
        if self.busy {
            self.inspect_after = true;
            return;
        }
        self.error = None;
        self.cancelled = false;
        self.progress = None;
        match self.commands.try_send(Command::Inspect(self.required.clone())) {
            Ok(()) => { self.busy = true; self.downloading = false; }
            Err(_) => self.error = Some("The file setup service could not be started. Restart the application and try again.".into()),
        }
    }

    fn install(&mut self) {
        if self.busy {
            return;
        }
        if self.requires_reinstall() {
            self.error = Some("Reinstall app files: re-extract the current application package into this folder. Your existing model files can be kept, then choose Recheck app files.".into());
            return;
        }
        let cancellation = AssetCancellation::default();
        self.error = None;
        self.cancelled = false;
        self.progress = None;
        match self.commands.try_send(Command::Install(
            self.required.clone(),
            cancellation.clone(),
        )) {
            Ok(()) => {
                self.cancellation = Some(cancellation);
                self.busy = true;
                self.downloading = true;
            }
            Err(_) => self.error = Some(
                "The download service could not be started. Restart the application and try again."
                    .into(),
            ),
        }
    }

    pub fn poll(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Progress(progress) => self.progress = Some(progress),
                Event::Finished {
                    groups,
                    result,
                    cancelled,
                } => {
                    self.busy = false;
                    self.downloading = false;
                    self.cancellation = None;
                    self.cancelled = cancelled;
                    match result {
                        Ok(status) => {
                            self.verified.retain(|group| !groups.contains(group));
                            for group in groups {
                                if !status.missing.iter().any(|file| file.group == group) {
                                    self.verified.push(group);
                                }
                            }
                            self.changed = true;
                            self.status = Some(status);
                            self.error = None;
                            if self.ready_for(&self.required) {
                                self.open = false;
                            }
                        }
                        Err(error) => {
                            if error.contains("Reinstall app files") {
                                self.inspect_after = true;
                            }
                            self.error = Some(error);
                        }
                    }
                }
            }
        }
        if !self.busy && self.inspect_after {
            self.inspect_after = false;
            self.inspect();
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        let requires_reinstall = self.requires_reinstall();
        let mut open = self.open;
        let mut dismiss = false;
        egui::Window::new("Required files")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(510.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(RichText::new(if self.busy && self.downloading { "Preparing BRP CHROMA" } else if self.busy { "Checking required files" } else if requires_reinstall { "Reinstall app files" } else { "Download required files" }).size(22.0).strong());
                ui.add_space(8.0);
                ui.label("BRP CHROMA includes its native runtime and media tools beside the app. Model downloads start only after your confirmation; verified files are reused automatically when you reopen it.");
                if requires_reinstall {
                    ui.add_space(8.0);
                    ui.label(RichText::new("A bundled application file is missing or modified. Re-extract the current full app package into this folder, keeping your model files, then choose Recheck app files. Native app files are not downloaded from the model host.").strong());
                }
                ui.add_space(12.0);
                if let Some(status) = &self.status {
                    if !status.missing.is_empty() {
                        ui.label(RichText::new("Files to prepare").small().weak());
                    }
                    egui::Grid::new("asset_groups").num_columns(2).spacing([30.0, 8.0]).show(ui, |ui| {
                        for group in &self.required {
                            let bytes: u64 = status.missing.iter().filter(|file| file.group == *group).map(|file| file.size).sum();
                            if bytes > 0 {
                                ui.label(group_label(*group));
                                ui.label(RichText::new(byte_label(bytes)).strong());
                                ui.end_row();
                            }
                        }
                    });
                    let bytes: u64 = status.missing.iter().filter(|file| !file.bundled_only).map(|file| file.size).sum();
                    if bytes > 0 {
                        ui.add_space(10.0);
                        ui.label(RichText::new(format!("Total download: {}", byte_label(status.download_bytes))).strong());
                        if status.reusable_bytes > 0 {
                            ui.label(format!("{} reused from files already on this computer.", byte_label(status.reusable_bytes)));
                        }
                        ui.label(RichText::new("Model weights are downloaded separately from the app.").small().weak());
                    }
                }
                ui.add_space(10.0);
                ui.label(RichText::new("Application folder").small().weak());
                ui.add(egui::Label::new(self.directory.display().to_string()).wrap());
                ui.add_space(12.0);
                if self.busy {
                    if let Some(progress) = &self.progress {
                        ui.label(format!("{}: {}", group_label(progress.group), phase_label(progress.phase)));
                        let fraction = if progress.total == 0 { 0.0 } else { progress.downloaded as f32 / progress.total as f32 };
                        ui.add(egui::ProgressBar::new(fraction.clamp(0.0, 1.0)).fill(crate::theme::ACCENT).desired_width(ui.available_width()).show_percentage());
                        ui.label(RichText::new(format!("File {} of {}  ·  {} / {}", progress.file_index, progress.file_count, byte_label(progress.downloaded), byte_label(progress.total))).small().weak());
                    } else {
                        ui.horizontal(|ui| { ui.spinner(); ui.label(if self.downloading { "Starting download..." } else { "Verifying files already on this computer..." }); });
                    }
                    if let Some(cancellation) = &self.cancellation {
                        if cancellation.is_cancelled() { ui.label("Cancelling download..."); }
                        else if ui.button("Cancel download").clicked() { cancellation.cancel(); }
                    }
                    ctx.request_repaint_after(std::time::Duration::from_millis(40));
                } else {
                    if self.cancelled {
                        ui.label("Download cancelled. Completed files are kept; retry resumes unfinished downloads.");
                    } else if self.error.is_some() && !requires_reinstall {
                        ui.label(RichText::new("The required files could not be prepared. Check your connection and try again, or open Details.").color(Color32::from_rgb(239, 172, 92)));
                    }
                    if let Some(error) = &self.error {
                        egui::CollapsingHeader::new("Details").show(ui, |ui| {
                            egui::ScrollArea::vertical().max_height(170.0).show(ui, |ui| { ui.add(egui::Label::new(error).wrap()); });
                        });
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let has_missing = self.status.as_ref().is_some_and(|status| !status.missing.is_empty());
                        if requires_reinstall {
                            ui.add_enabled(false, egui::Button::new("Download models")).on_hover_text("Restore the bundled app files before downloading models.");
                            if ui.button("Recheck app files").clicked() { self.inspect(); }
                        } else if has_missing {
                            if ui.button(if self.error.is_some() || self.cancelled { "Retry download" } else { "Download" }).clicked() { self.install(); }
                        } else if ui.button("Retry check").clicked() { self.inspect(); }
                        if ui.button("Not now").clicked() { dismiss = true; }
                    });
                    if !requires_reinstall { ui.label(RichText::new("Downloads start only after you choose Download.").small().weak()); }
                }
            });
        self.open = open && !dismiss;
    }

    fn requires_reinstall(&self) -> bool {
        self.status
            .as_ref()
            .is_some_and(|status| status.missing.iter().any(|file| file.bundled_only))
    }
}

impl Drop for AssetSetup {
    fn drop(&mut self) {
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
        }
    }
}

fn group_label(group: AssetGroup) -> &'static str {
    match group {
        AssetGroup::Core => "Native runtime and media tools",
        AssetGroup::Ck1080 => "CorridorKey",
        AssetGroup::Sam => "SAM video",
        AssetGroup::Transnet => "Scene detection",
        AssetGroup::Ck4k => "Legacy CorridorKey assets",
        AssetGroup::Runtime => "Inference runtime",
    }
}

fn phase_label(phase: AssetPhase) -> &'static str {
    match phase {
        AssetPhase::Checking => "Checking",
        AssetPhase::Downloading => "Downloading",
        AssetPhase::Verifying => "Verifying download",
        AssetPhase::Activating => "Finishing installation",
    }
}

fn byte_label(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_desktop_core::{AssetFile, AssetManifest};
    use std::{
        fs,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    fn wait_for_check(setup: &mut AssetSetup) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while setup.busy && Instant::now() < deadline {
            setup.poll();
            thread::sleep(Duration::from_millis(2));
        }
        assert!(!setup.busy, "asset inspection did not finish");
        assert!(setup.error.is_none(), "{:?}", setup.error);
    }

    fn fixture_folder() -> PathBuf {
        // Parallel tests can observe the same Windows clock tick. Keep their
        // manifests isolated even when SystemTime::now() returns equal values.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let folder = std::env::temp_dir().join(format!(
            "brp-setup-ui-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(folder.join("managed")).unwrap();
        let resolved = folder.canonicalize().unwrap();
        assert_eq!(
            resolved.parent(),
            Some(std::env::temp_dir().canonicalize().unwrap().as_path())
        );
        assert!(
            resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("brp-setup-ui-")
        );
        let files = [
            AssetGroup::Core,
            AssetGroup::Ck1080,
            AssetGroup::Sam,
            AssetGroup::Transnet,
            AssetGroup::Runtime,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, group)| AssetFile {
            group,
            path: PathBuf::from(format!("managed/{index}.bin")),
            // This fixture is intentionally never downloaded.
            url: "https://example.invalid/test-only/asset".into(),
            size: 1,
            sha256: "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881".into(),
            bundled_only: false,
        })
        .collect();
        fs::write(
            folder.join("assets-manifest.json"),
            serde_json::to_vec(&AssetManifest {
                schema_version: 1,
                id: "ui-consent-fixture".into(),
                version: "1".into(),
                files,
            })
            .unwrap(),
        )
        .unwrap();
        folder
    }

    #[test]
    fn reopening_uses_verified_app_folder_assets_and_startup_never_installs() {
        let folder = fixture_folder();
        fs::write(folder.join("managed/0.bin"), b"x").unwrap();
        let mut setup = AssetSetup::new(folder.clone(), egui::Context::default());
        wait_for_check(&mut setup);
        assert!(setup.ready_for(&[AssetGroup::Core]));
        assert!(!setup.ready_for(&STARTUP_GROUPS));
        assert_eq!(setup.status.as_ref().unwrap().missing.len(), 4);
        for index in 1..=4 {
            assert!(!folder.join(format!("managed/{index}.bin")).exists());
        }

        // Existing/adopted models are discovered without changing any path setting.
        for index in 1..=4 {
            fs::write(folder.join(format!("managed/{index}.bin")), b"x").unwrap();
        }
        setup.recheck();
        wait_for_check(&mut setup);
        assert!(setup.ready_for(&STARTUP_GROUPS));
        assert!(!setup.open);
        drop(setup);
        let mut reopened = AssetSetup::new(folder.clone(), egui::Context::default());
        wait_for_check(&mut reopened);
        assert!(reopened.ready_for(&STARTUP_GROUPS));
        assert!(!reopened.open);
        drop(reopened);
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn missing_bundled_runtime_keeps_media_ready_but_blocks_inference() {
        let folder = fixture_folder();
        for index in 0..=3 {
            fs::write(folder.join(format!("managed/{index}.bin")), b"x").unwrap();
        }
        let mut setup = AssetSetup::new(folder.clone(), egui::Context::default());
        wait_for_check(&mut setup);
        assert_eq!(setup.status.as_ref().unwrap().missing.len(), 1);
        assert!(setup.require(&[AssetGroup::Core]));
        assert!(!setup.require(&[AssetGroup::Core, AssetGroup::Runtime, AssetGroup::Sam]));
        assert!(!folder.join("managed/4.bin").exists());
        drop(setup);
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn missing_bundled_native_shows_reinstall_state_and_does_not_enqueue_downloads() {
        let folder = fixture_folder();
        let path = folder.join("assets-manifest.json");
        let mut manifest: AssetManifest =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest.files[0].bundled_only = true;
        manifest.files[0].url.clear();
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let mut setup = AssetSetup::new(folder.clone(), egui::Context::default());
        wait_for_check(&mut setup);
        assert!(setup.requires_reinstall());
        setup.install();
        assert!(!setup.busy && !setup.downloading && setup.cancellation.is_none());
        assert!(
            setup
                .error
                .as_ref()
                .is_some_and(|error| error.contains("Reinstall app files"))
        );
        assert!(!folder.join(".asset-staging").exists());
        fs::write(folder.join("managed/0.bin"), b"x").unwrap();
        setup.recheck();
        wait_for_check(&mut setup);
        assert!(!setup.requires_reinstall());
        assert!(setup.ready_for(&[AssetGroup::Core]));
        assert!(!setup.ready_for(&STARTUP_GROUPS)); // Models still need consent.
        drop(setup);
        fs::remove_dir_all(folder).unwrap();
    }
}
