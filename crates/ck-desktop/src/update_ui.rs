//! Update check on start, title-bar button and update window (see `ck_desktop_core::updater`).

use crate::theme;
use ck_desktop_core::updater::{self, UpdateCancel, UpdatePlan};
use eframe::egui::{self, RichText};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
enum State {
    Checking,
    UpToDate,
    Available(UpdatePlan),
    Downloading {
        plan: UpdatePlan,
        done: u64,
        total: u64,
    },
    Ready(UpdatePlan),
    /// The check failed (offline, no public release): stays quiet.
    CheckFailed(String),
    /// Download or install failed: shown in the window.
    Failed(UpdatePlan, String),
}

pub struct Updater {
    app_dir: PathBuf,
    state: Arc<Mutex<State>>,
    cancel: UpdateCancel,
    pub open: bool,
}

impl Updater {
    /// Starts the background check unless disabled (`BRP_CHROMA_NO_UPDATE_CHECK=1`, screenshot
    /// or interface-only sessions).
    pub fn start(app_dir: PathBuf, enabled: bool, repaint: egui::Context) -> Self {
        updater::cleanup_previous(&app_dir);
        let enabled =
            enabled && !std::env::var("BRP_CHROMA_NO_UPDATE_CHECK").is_ok_and(|v| v == "1");
        let state = Arc::new(Mutex::new(if enabled {
            State::Checking
        } else {
            State::UpToDate
        }));
        let this = Self {
            app_dir: app_dir.clone(),
            state: state.clone(),
            cancel: UpdateCancel::default(),
            open: false,
        };
        if enabled {
            std::thread::spawn(move || {
                let current = updater::installed_version(&app_dir);
                let next = match updater::check_latest(updater::UPDATE_REPOSITORY, &current) {
                    Ok(Some(plan)) => State::Available(plan),
                    Ok(_) => State::UpToDate,
                    Err(error) => State::CheckFailed(format!("{error:#}")),
                };
                if let Ok(mut state) = state.lock() {
                    *state = next;
                }
                repaint.request_repaint();
            });
        }
        this
    }

    fn state(&self) -> State {
        self.state
            .lock()
            .map(|s| s.clone())
            .unwrap_or(State::UpToDate)
    }

    /// Title-bar button while an update is available, downloading or waiting for a restart.
    pub fn title_button(&mut self, ui: &mut egui::Ui) {
        let (label, hover) = match self.state() {
            State::Available(plan) => (
                "Update".to_owned(),
                format!(
                    "Version {} is available ({} to download)",
                    plan.version,
                    size(plan.bytes)
                ),
            ),
            State::Downloading { done, total, .. } => (
                format!("Updating {}%", percent(done, total)),
                "Downloading the update".to_owned(),
            ),
            State::Ready(plan) => (
                "Restart".to_owned(),
                format!("Version {} is installed; restart to use it", plan.version),
            ),
            State::Failed(..) => ("Update failed".to_owned(), "Open for details".to_owned()),
            _ => return,
        };
        let button = egui::Button::new(RichText::new(label).small().color(egui::Color32::WHITE))
            .fill(theme::ACCENT)
            .corner_radius(6);
        if ui.add(button).on_hover_text(hover).clicked() {
            self.open = true;
        }
    }

    /// The update window. `idle` = no calculation running (updates replace the worker).
    pub fn window(&mut self, ctx: &egui::Context, idle: bool) {
        if !self.open {
            return;
        }
        let state = self.state();
        let mut open = self.open;
        let mut start = false;
        let mut cancel = false;
        let mut restart = false;
        egui::Window::new("Update")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(420.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| match &state {
                State::Available(plan) | State::Failed(plan, _) => {
                    ui.label(
                        RichText::new(format!("BRP CHROMA {} is available", plan.version)).strong(),
                    );
                    ui.label(format!(
                        "Installed: {}. Download: {}.",
                        installed_label(&updater::installed_version(&self.app_dir)),
                        size(plan.bytes)
                    ));
                    if !plan.notes.trim().is_empty() {
                        ui.add_space(6.0);
                        egui::ScrollArea::vertical()
                            .max_height(180.0)
                            .show(ui, |ui| {
                                ui.label(RichText::new(plan.notes.trim()).small());
                            });
                    }
                    if let State::Failed(_, error) = &state {
                        ui.add_space(6.0);
                        ui.colored_label(theme::BAD, error);
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        start = ui
                            .add_enabled(idle, theme::primary_button("Update now", 140.0))
                            .on_disabled_hover_text("Wait for the current calculation to finish.")
                            .clicked();
                        if !plan.page.is_empty() {
                            ui.hyperlink_to("Release page", &plan.page);
                        }
                    });
                }
                State::Downloading { plan, done, total } => {
                    ui.label(format!("Downloading {}…", plan.version));
                    ui.add(
                        egui::ProgressBar::new(if *total > 0 {
                            *done as f32 / *total as f32
                        } else {
                            0.0
                        })
                        .fill(theme::ACCENT)
                        .text(format!(
                            "{} of {}",
                            size(*done),
                            size(*total)
                        )),
                    );
                    cancel = ui.button("Cancel").clicked();
                }
                State::Ready(plan) => {
                    ui.label(
                        RichText::new(format!("Version {} is installed.", plan.version)).strong(),
                    );
                    ui.label("Restart BRP CHROMA to use it. Save your project first.");
                    ui.add_space(8.0);
                    restart = ui
                        .add(theme::primary_button("Restart now", 140.0))
                        .clicked();
                }
                State::Checking => {
                    ui.label("Checking for updates…");
                }
                State::UpToDate => {
                    ui.label("BRP CHROMA is up to date.");
                }
                State::CheckFailed(error) => {
                    ui.label("Could not check for updates.");
                    ui.label(RichText::new(error).small());
                }
            });
        self.open = open;
        if start && let State::Available(plan) | State::Failed(plan, _) = state.clone() {
            self.download_and_install(plan, ctx.clone());
        }
        if cancel {
            self.cancel.cancel();
        }
        if restart {
            if let Ok(exe) = std::env::current_exe() {
                let _ = std::process::Command::new(exe).spawn();
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn download_and_install(&mut self, plan: UpdatePlan, repaint: egui::Context) {
        self.cancel = UpdateCancel::default();
        let cancel = self.cancel.clone();
        let state = self.state.clone();
        let app_dir = self.app_dir.clone();
        if let Ok(mut s) = state.lock() {
            *s = State::Downloading {
                plan: plan.clone(),
                done: 0,
                total: plan.bytes,
            };
        }
        std::thread::spawn(move || {
            let progress_state = state.clone();
            let progress_plan = plan.clone();
            let progress_repaint = repaint.clone();
            let result = updater::download(&app_dir, &plan, &cancel, move |done, total| {
                if let Ok(mut s) = progress_state.lock() {
                    *s = State::Downloading {
                        plan: progress_plan.clone(),
                        done,
                        total,
                    };
                }
                progress_repaint.request_repaint();
            })
            .and_then(|staging| updater::apply(&app_dir, &plan, &staging));
            if let Ok(mut s) = state.lock() {
                *s = match result {
                    Ok(()) => State::Ready(plan),
                    Err(error) => State::Failed(plan, format!("{error:#}")),
                };
            }
            repaint.request_repaint();
        });
    }
}

fn installed_label(version: &str) -> String {
    if version == "0" {
        "unknown".to_owned()
    } else {
        version.to_owned()
    }
}

fn percent(done: u64, total: u64) -> u64 {
    (done * 100).checked_div(total).unwrap_or(0)
}

fn size(bytes: u64) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    if mb >= 1024.0 {
        format!("{:.2} GB", mb / 1024.0)
    } else if mb >= 0.1 {
        format!("{mb:.1} MB")
    } else {
        format!("{} KB", bytes.div_ceil(1024))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_percent() {
        assert_eq!(size(512), "1 KB");
        assert_eq!(size(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(size(3 * 1024 * 1024 * 1024), "3.00 GB");
        assert_eq!(percent(50, 200), 25);
        assert_eq!(percent(1, 0), 0);
    }

    #[test]
    fn disabled_updater_stays_quiet() {
        let dir = std::env::temp_dir().join(format!("brp-update-ui-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let updater = Updater::start(dir.clone(), false, egui::Context::default());
        assert!(matches!(updater.state(), State::UpToDate));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
