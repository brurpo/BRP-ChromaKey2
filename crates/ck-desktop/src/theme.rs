//! Neutral dark studio chrome and joined panel tabs.
//!
//! Base surfaces, borders and text are achromatic. Color is reserved for
//! explicit scene/mask semantics rather than active panel or widget states.

use eframe::egui::{self, Color32, FontId, Stroke, TextStyle, Vec2};

pub const BG_DEEP: Color32 = Color32::from_gray(16);
pub const BG_APP: Color32 = Color32::from_gray(22);
pub const BG_PANEL: Color32 = Color32::from_gray(29);
pub const BG_PANEL_ALT: Color32 = Color32::from_gray(38);
pub const BG_INPUT: Color32 = Color32::from_gray(25);
pub const BORDER: Color32 = Color32::from_gray(58);
pub const BORDER_SOFT: Color32 = Color32::from_gray(43);
pub const TEXT: Color32 = Color32::from_gray(226);
pub const TEXT_DIM: Color32 = Color32::from_gray(148);
pub const ACCENT: Color32 = Color32::from_rgb(0x58, 0xb6, 0xf2);
// Legacy chrome-highlight names remain neutral for existing button callers.
pub const ACCENT_BRIGHT: Color32 = Color32::from_gray(232);
pub const ACCENT_DARK: Color32 = Color32::from_gray(72);
pub const ACCENT_DIM: Color32 = Color32::from_gray(50);
pub const GOOD: Color32 = Color32::from_rgb(0x4a, 0xde, 0x80);
pub const WARN: Color32 = Color32::from_rgb(0xfb, 0xbf, 0x24);
pub const BAD: Color32 = Color32::from_rgb(0xf8, 0x71, 0x71);

pub fn configure(ctx: &egui::Context) {
    // Keep the studio dark when the operating system changes its own theme.
    ctx.set_theme(egui::ThemePreference::Dark);
    let mut style = (*ctx.style()).clone();
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG_APP;
    visuals.window_fill = BG_PANEL;
    visuals.window_stroke = Stroke::new(1.0_f32, BORDER);
    visuals.extreme_bg_color = BG_INPUT;
    visuals.text_edit_bg_color = Some(BG_INPUT);
    visuals.faint_bg_color = BG_PANEL_ALT;
    visuals.code_bg_color = BG_DEEP;
    visuals.weak_text_color = Some(TEXT_DIM);
    visuals.hyperlink_color = ACCENT_BRIGHT;
    visuals.warn_fg_color = TEXT;
    visuals.error_fg_color = TEXT;
    visuals.selection.bg_fill = ACCENT_DIM;
    visuals.selection.stroke = Stroke::new(1.0_f32, TEXT);
    visuals.text_cursor.stroke = Stroke::new(1.0_f32, TEXT);
    visuals.window_corner_radius = 8.into();
    visuals.menu_corner_radius = 4.into();
    visuals.disabled_alpha = 0.45;
    visuals.slider_trailing_fill = true;

    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER_SOFT);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.noninteractive.corner_radius = 4.into();

    visuals.widgets.inactive.bg_fill = BG_PANEL_ALT;
    visuals.widgets.inactive.weak_bg_fill = BG_PANEL_ALT;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.inactive.corner_radius = 4.into();

    visuals.widgets.hovered.bg_fill = Color32::from_gray(47);
    visuals.widgets.hovered.weak_bg_fill = visuals.widgets.hovered.bg_fill;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_gray(79));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.hovered.corner_radius = 4.into();
    visuals.widgets.hovered.expansion = 0.0;

    visuals.widgets.active.bg_fill = ACCENT_DARK;
    visuals.widgets.active.weak_bg_fill = ACCENT_DARK;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, Color32::from_gray(120));
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.active.corner_radius = 4.into();
    visuals.widgets.active.expansion = 0.0;

    visuals.widgets.open.bg_fill = ACCENT_DIM;
    visuals.widgets.open.weak_bg_fill = ACCENT_DIM;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, Color32::from_gray(100));
    visuals.widgets.open.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.open.corner_radius = 4.into();

    style.visuals = visuals;
    style.spacing.item_spacing = Vec2::new(8.0, 6.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.interact_size = Vec2::new(32.0, 28.0);
    style.spacing.icon_width = 16.0;
    style.spacing.icon_width_inner = 10.0;
    style.spacing.icon_spacing = 6.0;
    style.spacing.window_margin = egui::Margin::same(12);
    style.spacing.menu_margin = egui::Margin::symmetric(8, 6);
    style.spacing.slider_width = 130.0;
    style.spacing.combo_width = 120.0;
    style.spacing.scroll.bar_width = 10.0;
    style.drag_value_text_style = TextStyle::Body;
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(13.0));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(13.0));
    style
        .text_styles
        .insert(TextStyle::Small, FontId::proportional(11.0));
    style
        .text_styles
        .insert(TextStyle::Heading, FontId::proportional(18.0));
    style
        .text_styles
        .insert(TextStyle::Monospace, FontId::monospace(13.0));
    ctx.set_style(style);
    install_platform_font(ctx);
}

// ---------------------------------------------------------------- grouped-card layout
// Inspector panels group related settings in neutral cards: a short title, rows with the
// label on the left and the control on the right, muted help text below.

/// Card surface, one step above the panel.
pub const CARD_FILL: Color32 = Color32::from_gray(34);
pub const CARD_BORDER: Color32 = Color32::from_gray(46);
/// The one accent-filled action per panel (Run).
pub const PRIMARY_FILL: Color32 = Color32::from_rgb(38, 99, 145);
pub const PRIMARY_HOVER: Color32 = Color32::from_rgb(46, 115, 166);
/// Selected segment of a segmented control.
pub const SEGMENT_ON: Color32 = Color32::from_rgb(33, 66, 92);
/// Vertical gap between cards.
pub const CARD_GAP: f32 = 8.0;

/// A titled card. `trailing` draws a small muted badge at the right of the title row.
pub fn card<R>(
    ui: &mut egui::Ui,
    title: &str,
    trailing: Option<&str>,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let inner = egui::Frame::new()
        .fill(CARD_FILL)
        .stroke(Stroke::new(1.0_f32, CARD_BORDER))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 8.0;
            if !title.is_empty() || trailing.is_some() {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(title).size(13.5).strong().color(TEXT));
                    if let Some(trailing) = trailing {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            badge(ui, trailing);
                        });
                    }
                });
            }
            add(ui)
        })
        .inner;
    ui.add_space(CARD_GAP);
    inner
}

/// Label on the left, control right-aligned on the same line.
pub fn field_row<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(TEXT_DIM));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), add)
            .inner
    })
    .inner
}

/// Muted, wrapping help text.
pub fn hint(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    ui.add(egui::Label::new(egui::RichText::new(text).size(11.5).color(TEXT_DIM)).wrap())
}

/// Small rounded pill with muted text (counts, frame ranges, status).
pub fn badge(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), FontId::proportional(11.0), TEXT_DIM);
    let size = galley.size() + Vec2::new(14.0, 6.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, 9.0, BG_PANEL_ALT);
    ui.painter()
        .galley(rect.center() - galley.size() / 2.0, galley, TEXT_DIM);
    response
}

/// Joined buttons choosing one of a few options; returns whether the value changed.
pub fn segmented<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    value: &mut T,
    options: &[(T, &str, &str)],
) -> bool {
    let font = FontId::proportional(12.5);
    let widths: Vec<f32> = options
        .iter()
        .map(|(_, label, _)| {
            ui.painter()
                .layout_no_wrap((*label).to_owned(), font.clone(), TEXT)
                .size()
                .x
                + 20.0
        })
        .collect();
    segmented_with_widths(ui, value, options, &widths)
}

/// [`segmented`] spanning the full available width with equal segments.
pub fn segmented_fill<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    value: &mut T,
    options: &[(T, &str, &str)],
) -> bool {
    let each = ui.available_width() / options.len().max(1) as f32;
    let widths = vec![each; options.len()];
    segmented_with_widths(ui, value, options, &widths)
}

fn segmented_with_widths<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    value: &mut T,
    options: &[(T, &str, &str)],
    widths: &[f32],
) -> bool {
    let font = FontId::proportional(12.5);
    let height = 26.0;
    // Segment ids derive from this strip's own (unique) id: several segmented controls in one
    // `Ui` must not share interaction ids, or a click lands on (and changes) all of them.
    let (strip, strip_response) = ui.allocate_exact_size(
        Vec2::new(widths.iter().sum::<f32>(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(strip.expand(1.0));
    painter.rect(
        strip,
        6.0,
        BG_INPUT,
        Stroke::new(1.0_f32, BORDER),
        egui::StrokeKind::Inside,
    );
    let mut changed = false;
    let mut left = strip.left();
    for (index, ((option, label, hover), width)) in options.iter().zip(widths).enumerate() {
        let rect =
            egui::Rect::from_min_size(egui::pos2(left, strip.top()), Vec2::new(*width, height));
        left += width;
        let response = ui
            .interact(
                rect,
                strip_response.id.with(("segment", index)),
                egui::Sense::click(),
            )
            .on_hover_text(*hover);
        if response.clicked() && *value != *option {
            *value = *option;
            changed = true;
        }
        let selected = *value == *option;
        if selected || response.hovered() {
            painter.rect_filled(
                rect.shrink(2.0),
                4.0,
                if selected { SEGMENT_ON } else { BG_PANEL_ALT },
            );
        }
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            *label,
            font.clone(),
            if selected { TEXT } else { TEXT_DIM },
        );
    }
    changed
}

/// Full-width primary action (one per panel).
pub fn primary_button(text: impl Into<egui::WidgetText>, width: f32) -> egui::Button<'static> {
    egui::Button::new(text)
        .fill(PRIMARY_FILL)
        .stroke(Stroke::new(1.0_f32, PRIMARY_HOVER))
        .corner_radius(6)
        .min_size(Vec2::new(width, 34.0))
}

/// Full-width secondary action.
pub fn secondary_button(text: impl Into<egui::WidgetText>, width: f32) -> egui::Button<'static> {
    egui::Button::new(text)
        .corner_radius(6)
        .min_size(Vec2::new(width, 32.0))
}

/// A joined tab strip: the selected tab opens into the panel below it.
/// The caller should not add another separator immediately below this strip.
pub fn tab_bar(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    labels: &[&str],
    selected: usize,
) -> usize {
    if labels.is_empty() {
        return 0;
    }
    ui.push_id(id_salt, |ui| {
        let selected = selected.min(labels.len() - 1);
        let mut next = selected;
        let font = FontId::proportional(13.0);
        let widths: Vec<f32> = labels
            .iter()
            .map(|label| {
                (ui.painter()
                    .layout_no_wrap((*label).to_owned(), font.clone(), TEXT)
                    .size()
                    .x
                    + 24.0)
                    .max(60.0)
            })
            .collect();
        let available = ui.available_width().max(1.0);
        let scale = (available / widths.iter().sum::<f32>()).min(1.0);
        let (strip, _) = ui.allocate_exact_size(Vec2::new(available, 32.0), egui::Sense::hover());
        let painter = ui.painter_at(strip);
        painter.rect_filled(strip, 0.0, BG_APP);
        let baseline = strip.bottom() - 1.0;
        painter.line_segment(
            [
                egui::pos2(strip.left(), baseline),
                egui::pos2(strip.right(), baseline),
            ],
            Stroke::new(1.0_f32, BORDER),
        );
        let mut left = strip.left();
        let mut tabs = Vec::with_capacity(labels.len());
        for (index, width) in widths.into_iter().enumerate() {
            let rect = egui::Rect::from_min_max(
                egui::pos2(left, strip.top() + 2.0),
                egui::pos2(left + width * scale, strip.bottom()),
            );
            let mut response = ui.interact(rect, ui.id().with(index), egui::Sense::click());
            if response.clicked() {
                next = index;
                response.mark_changed();
            }
            left = rect.right();
            tabs.push((rect, response));
        }
        for (index, (label, (rect, response))) in labels.iter().zip(tabs).enumerate() {
            let active = index == next;
            let fill = if active {
                BG_PANEL
            } else if response.hovered() {
                BG_PANEL_ALT
            } else {
                BG_APP
            };
            painter.rect(
                rect,
                egui::CornerRadius {
                    nw: 4,
                    ne: 4,
                    sw: 0,
                    se: 0,
                },
                fill,
                Stroke::new(1.0_f32, if active { BORDER } else { BORDER_SOFT }),
                egui::StrokeKind::Inside,
            );
            if active {
                // Erase the selected tab's bottom edge, joining it to the panel.
                painter.line_segment(
                    [
                        egui::pos2(rect.left() + 1.0, baseline),
                        egui::pos2(rect.right() - 1.0, baseline),
                    ],
                    Stroke::new(2.0_f32, BG_PANEL),
                );
            }
            painter.with_clip_rect(rect.shrink(2.0)).text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                *label,
                FontId::proportional((13.0 * scale).max(11.0)),
                if active { TEXT } else { TEXT_DIM },
            );
            response.widget_info(|| {
                egui::WidgetInfo::selected(
                    egui::WidgetType::SelectableLabel,
                    ui.is_enabled(),
                    active,
                    *label,
                )
            });
            response.on_hover_text(*label);
        }
        next
    })
    .inner
}

// Match the original app's Segoe UI on Windows using the OS font. Other
// platforms keep egui's bundled fonts, so no extra font file ships with the app.
#[cfg(windows)]
fn install_platform_font(ctx: &egui::Context) {
    let Some(windows) = std::env::var_os("WINDIR") else {
        return;
    };
    let path = std::path::Path::new(&windows).join("Fonts/segoeui.ttf");
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("Segoe UI".into(), egui::FontData::from_owned(bytes).into());
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "Segoe UI".into());
    ctx.set_fonts(fonts);
}

#[cfg(not(windows))]
fn install_platform_font(_ctx: &egui::Context) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_chrome_stays_desaturated_in_every_widget_state() {
        let ctx = egui::Context::default();
        configure(&ctx);
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        let style = ctx.style();
        let visuals = &style.visuals;
        let mut colors = vec![
            visuals.panel_fill,
            visuals.window_fill,
            visuals.window_stroke.color,
            visuals.extreme_bg_color,
            visuals.faint_bg_color,
            visuals.code_bg_color,
            visuals.hyperlink_color,
            visuals.warn_fg_color,
            visuals.error_fg_color,
            visuals.selection.bg_fill,
            visuals.selection.stroke.color,
            visuals.text_cursor.stroke.color,
        ];
        colors.extend(visuals.text_edit_bg_color);
        colors.extend(visuals.weak_text_color);
        for widget in [
            &visuals.widgets.noninteractive,
            &visuals.widgets.inactive,
            &visuals.widgets.hovered,
            &visuals.widgets.active,
            &visuals.widgets.open,
        ] {
            colors.extend([
                widget.bg_fill,
                widget.weak_bg_fill,
                widget.bg_stroke.color,
                widget.fg_stroke.color,
            ]);
        }
        for color in colors {
            assert_eq!(color.r(), color.g(), "non-neutral chrome: {color:?}");
            assert_eq!(color.g(), color.b(), "non-neutral chrome: {color:?}");
        }
    }

    #[test]
    fn joined_tabs_select_on_cpu_pointer_release_and_keep_model_selection() {
        fn frame(ctx: &egui::Context, selected: usize, events: Vec<egui::Event>) -> usize {
            let mut next = selected;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    Vec2::new(240.0, 100.0),
                )),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    next = tab_bar(ui, "test-tabs", &["Media", "Subject"], selected);
                });
            });
            next
        }
        let ctx = egui::Context::default();
        assert_eq!(frame(&ctx, 0, vec![]), 0);
        let pointer = egui::pos2(100.0, 24.0);
        let mut selected = 0;
        for pressed in [true, false] {
            selected = frame(
                &ctx,
                selected,
                vec![
                    egui::Event::PointerMoved(pointer),
                    egui::Event::PointerButton {
                        pos: pointer,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
        assert_eq!(selected, 1);
        assert_eq!(frame(&ctx, selected, vec![]), 1);
        assert_eq!(frame(&ctx, 0, vec![]), 0);
    }

    #[test]
    fn segmented_controls_in_one_ui_change_independently() {
        // Two stacked segmented controls in the same Ui (Window + Hint quality, layer rows).
        fn frame(ctx: &egui::Context, values: &mut [u8; 2], events: Vec<egui::Event>) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    Vec2::new(300.0, 200.0),
                )),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    for value in values.iter_mut() {
                        segmented(
                            ui,
                            value,
                            &[(0, "Aaaa", ""), (1, "Bbbb", ""), (2, "Cccc", "")],
                        );
                    }
                });
            });
        }
        let ctx = egui::Context::default();
        let mut values = [0u8, 0u8];
        frame(&ctx, &mut values, vec![]);
        // Click the middle segment of the second control (rows are 26 pt tall after the
        // panel margin, with the default item spacing between them).
        let spacing = ctx.style().spacing.item_spacing.y;
        let margin = 8.0;
        let pointer = egui::pos2(margin + 60.0 + 30.0, margin + 26.0 + spacing + 13.0);
        for pressed in [true, false] {
            frame(
                &ctx,
                &mut values,
                vec![
                    egui::Event::PointerMoved(pointer),
                    egui::Event::PointerButton {
                        pos: pointer,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
        assert_eq!(values, [0, 1], "only the clicked control changes");
    }
}
