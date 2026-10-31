//! The supplied studio artwork embedded in the native GUI.
//!
//! Only the icons used by the current toolbar and transport are included. The
//! source PNGs stay unchanged; decoding and HiDPI resampling happen on the CPU.

use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions, Vec2};
use image::{RgbaImage, imageops::FilterType};

const LOGICAL_ICON_SIZE: f32 = 16.0;
const EMBEDDED: [(&str, &[u8]); 11] = [
    ("video", include_bytes!("../../../icons/video.png")),
    (
        "folder_expanded",
        include_bytes!("../../../icons/folder_expanded.png"),
    ),
    ("sequence", include_bytes!("../../../icons/sequence.png")),
    ("config", include_bytes!("../../../icons/config.png")),
    ("img", include_bytes!("../../../icons/img.png")),
    ("rewind", include_bytes!("../../../icons/rewind.png")),
    ("step_back", include_bytes!("../../../icons/step_back.png")),
    ("play", include_bytes!("../../../icons/play.png")),
    ("pause", include_bytes!("../../../icons/pause.png")),
    (
        "step_foward",
        include_bytes!("../../../icons/step_foward.png"),
    ),
    ("foward", include_bytes!("../../../icons/foward.png")),
];

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Icon {
    OpenMedia,
    OpenProject,
    /// Supplied stacked-frames artwork (`sequence.png`).
    ExportRange,
    Settings,
    ExportFrame,
    RangeStart,
    StepBack,
    Play,
    Pause,
    StepForward,
    RangeEnd,
    SaveProject,
    /// Code-native artwork: two frames split by a cut (after the checkbox textures).
    DetectScenes = 14,
    Trash = 15,
    SegmentFrame = 16,
    Propagate = 17,
    PropagateAll = 18,
    ExportMask = 19,
}

const TRASH_PNG: &[u8] = include_bytes!("../../../icons/trash.png");
const IMG_PNG: &[u8] = include_bytes!("../../../icons/img.png");
const VIDEO_PNG: &[u8] = include_bytes!("../../../icons/video.png");
const DOCK_PNG: &[u8] = include_bytes!("../../../icons/dock.png");

/// Icons added after `DetectScenes`, in texture order.
fn subject_icons(raster_size: u32) -> [(&'static str, ColorImage); 5] {
    [
        ("trash", decode(TRASH_PNG, raster_size)),
        ("segment-frame", decode(IMG_PNG, raster_size)),
        ("propagate", decode(VIDEO_PNG, raster_size)),
        ("propagate-all", stacked_icon(VIDEO_PNG, raster_size)),
        ("export-mask", decode(DOCK_PNG, raster_size)),
    ]
}

/// Two offset copies of an icon (back one dimmed): "all layers" of an action.
fn stacked_icon(png: &[u8], raster_size: u32) -> ColorImage {
    let source = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .expect("embedded studio icons must contain valid PNG data")
        .to_rgba8();
    let size = raster_size.max(4);
    let small = (size * 3 / 4).max(1);
    let copy = image::imageops::resize(&source, small, small, FilterType::Lanczos3);
    let mut canvas = RgbaImage::new(size, size);
    let offset = size - small;
    for (x, y, pixel) in copy.enumerate_pixels() {
        // Back copy, top right, at 55% opacity.
        let mut back = *pixel;
        back.0[3] = (f32::from(back.0[3]) * 0.55) as u8;
        canvas.put_pixel(x + offset, y, back);
    }
    for (x, y, pixel) in copy.enumerate_pixels() {
        // Front copy, bottom left, over the back one.
        let dst = canvas.get_pixel_mut(x, y + offset);
        let a = f32::from(pixel.0[3]) / 255.0;
        for c in 0..3 {
            dst.0[c] = (f32::from(pixel.0[c]) * a + f32::from(dst.0[c]) * (1.0 - a)) as u8;
        }
        dst.0[3] = (f32::from(pixel.0[3]) + f32::from(dst.0[3]) * (1.0 - a)).min(255.0) as u8;
    }
    ColorImage::from_rgba_unmultiplied([size as usize, size as usize], canvas.as_raw())
}

impl Icon {
    fn label(self) -> &'static str {
        match self {
            Self::OpenMedia => "Open media",
            Self::OpenProject => "Open project",
            Self::ExportRange => "Export range",
            Self::DetectScenes => "Detect scenes",
            Self::Settings => "Settings",
            Self::ExportFrame => "Export frame",
            Self::RangeStart => "Go to start",
            Self::StepBack => "Previous frame",
            Self::Play => "Play",
            Self::Pause => "Pause",
            Self::StepForward => "Next frame",
            Self::RangeEnd => "Go to end",
            Self::SaveProject => "Save project",
            Self::Trash => "Delete",
            Self::SegmentFrame => "Segment frame",
            Self::Propagate => "Propagate layer",
            Self::PropagateAll => "Propagate all layers",
            Self::ExportMask => "Export mask",
        }
    }
}

const CHECKBOX_ON: usize = 12;
const CHECKBOX_OFF: usize = 13;

pub struct Icons {
    // Keep the handles alive: the returned buttons carry only the texture IDs.
    textures: Vec<TextureHandle>,
    raster_size: u32,
}

impl Icons {
    pub fn new(ctx: &egui::Context) -> Self {
        let raster_size = raster_size(ctx);
        let mut textures: Vec<_> = EMBEDDED
            .map(|(name, png)| {
                ctx.load_texture(
                    format!("studio-icon/{name}"),
                    decode(png, raster_size),
                    TextureOptions::LINEAR,
                )
            })
            .into();
        textures.push(ctx.load_texture(
            "studio-icon/save-project",
            save_icon(raster_size),
            TextureOptions::LINEAR,
        ));
        textures.push(ctx.load_texture(
            "studio-icon/checkbox-enabled",
            decode(
                include_bytes!("../../../icons/checkbox_enabled.png"),
                raster_size,
            ),
            TextureOptions::LINEAR,
        ));
        textures.push(ctx.load_texture(
            "studio-icon/checkbox-disabled",
            decode(
                include_bytes!("../../../icons/checkbox_disabled.png"),
                raster_size,
            ),
            TextureOptions::LINEAR,
        ));
        textures.push(ctx.load_texture(
            "studio-icon/detect-scenes",
            scene_cut_icon(raster_size),
            TextureOptions::LINEAR,
        ));
        for (name, image) in subject_icons(raster_size) {
            textures.push(ctx.load_texture(
                format!("studio-icon/{name}"),
                image,
                TextureOptions::LINEAR,
            ));
        }
        Self {
            textures,
            raster_size,
        }
    }

    /// Refresh only when moving between display densities or changing UI zoom.
    pub fn sync_density(&mut self, ctx: &egui::Context) {
        let raster_size = raster_size(ctx);
        if raster_size == self.raster_size {
            return;
        }
        for (texture, (_, png)) in self.textures.iter_mut().zip(EMBEDDED) {
            texture.set(decode(png, raster_size), TextureOptions::LINEAR);
        }
        self.textures[Icon::SaveProject as usize]
            .set(save_icon(raster_size), TextureOptions::LINEAR);
        self.textures[CHECKBOX_ON].set(
            decode(
                include_bytes!("../../../icons/checkbox_enabled.png"),
                raster_size,
            ),
            TextureOptions::LINEAR,
        );
        self.textures[CHECKBOX_OFF].set(
            decode(
                include_bytes!("../../../icons/checkbox_disabled.png"),
                raster_size,
            ),
            TextureOptions::LINEAR,
        );
        self.textures[Icon::DetectScenes as usize]
            .set(scene_cut_icon(raster_size), TextureOptions::LINEAR);
        for (index, (_, image)) in subject_icons(raster_size).into_iter().enumerate() {
            self.textures[Icon::Trash as usize + index].set(image, TextureOptions::LINEAR);
        }
        self.raster_size = raster_size;
    }

    /// Use `None` for transport controls, or a visible label for toolbar actions.
    /// The caller can increase the minimum size for Play and add its tooltip.
    pub fn button(&self, icon: Icon, text: Option<&str>) -> egui::Button<'static> {
        let image = egui::Image::from_texture((
            self.textures[icon as usize].id(),
            Vec2::splat(LOGICAL_ICON_SIZE),
        ))
        .alt_text(icon.label());
        // Button::image[_and_text] caps its image at the text font height.
        // Atoms preserve the requested 16 logical pixels at every DPI instead.
        let button = if let Some(text) = text {
            egui::Button::new((image, text.to_owned()))
        } else {
            egui::Button::new(image)
        };
        button
            .min_size(Vec2::splat(28.0))
            .image_tint_follows_text_color(false)
    }

    /// Texture of the supplied enabled/disabled checkbox artwork.
    /// Compact icon-only toolbar button; the tooltip says what it does.
    pub fn tool_button(&self, icon: Icon) -> egui::Button<'static> {
        let image = egui::Image::from_texture((
            self.textures[icon as usize].id(),
            Vec2::splat(LOGICAL_ICON_SIZE),
        ))
        .alt_text(icon.label());
        egui::Button::new(image)
            .min_size(Vec2::splat(30.0))
            .corner_radius(6)
            .image_tint_follows_text_color(false)
    }

    /// Icon-only button of a given size (tooltips carry the label).
    pub fn sized_button(&self, icon: Icon, size: Vec2) -> egui::Button<'static> {
        let image = egui::Image::from_texture((
            self.textures[icon as usize].id(),
            Vec2::splat(LOGICAL_ICON_SIZE),
        ))
        .alt_text(icon.label());
        egui::Button::new(image)
            .min_size(size)
            .corner_radius(6)
            .image_tint_follows_text_color(false)
    }

    pub fn texture(&self, icon: Icon) -> egui::TextureId {
        self.textures[icon as usize].id()
    }

    pub fn checkbox_texture(&self, checked: bool) -> egui::TextureId {
        self.textures[if checked { CHECKBOX_ON } else { CHECKBOX_OFF }].id()
    }

    /// Checkbox drawn with the supplied artwork instead of egui's painted box.
    pub fn checkbox(&self, ui: &mut egui::Ui, value: &mut bool, text: &str) -> egui::Response {
        checkbox_widget(
            ui,
            value,
            text,
            self.checkbox_texture(true),
            self.checkbox_texture(false),
        )
    }
}

/// Icon checkbox usable from free functions that only carry texture ids.
pub fn checkbox_widget(
    ui: &mut egui::Ui,
    value: &mut bool,
    text: &str,
    checked: egui::TextureId,
    unchecked: egui::TextureId,
) -> egui::Response {
    let texture = if *value { checked } else { unchecked };
    let image =
        egui::Image::from_texture((texture, Vec2::splat(LOGICAL_ICON_SIZE))).alt_text("checkbox");
    let response = ui.add(
        egui::Button::image_and_text(image, text)
            .frame(false)
            .min_size(Vec2::new(0.0, LOGICAL_ICON_SIZE)),
    );
    if response.clicked() {
        *value = !*value;
    }
    response
}

fn raster_size(ctx: &egui::Context) -> u32 {
    let density = ctx.pixels_per_point();
    let density = if density.is_finite() { density } else { 1.0 };
    (LOGICAL_ICON_SIZE * density).round().clamp(1.0, 256.0) as u32
}

fn decode(png: &[u8], raster_size: u32) -> ColorImage {
    let rgba = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .expect("embedded studio icons must contain valid PNG data")
        .to_rgba8();
    resize(rgba, raster_size)
}

fn save_icon(raster_size: u32) -> ColorImage {
    // A small code-native floppy-disk silhouette, matching the supplied
    // monochrome #c8c8c8 artwork. Transparent cut-outs show the actual button
    // background; no icon background, font, SVG runtime or download is needed.
    let source = RgbaImage::from_fn(256, 256, |x, y| {
        let body = (24..232).contains(&x) && (20..236).contains(&y) && (x < 192 || y >= x - 172);
        let top_window = (60..180).contains(&x) && (20..104).contains(&y);
        let disk_slot = (140..160).contains(&x) && (30..86).contains(&y);
        let label = (52..204).contains(&x) && (132..220).contains(&y);
        let label_lines =
            (68..188).contains(&x) && ((150..160).contains(&y) || (182..192).contains(&y));
        let filled = body && (!top_window || disk_slot) && (!label || label_lines);
        image::Rgba([200, 200, 200, if filled { 255 } else { 0 }])
    });
    resize(source, raster_size)
}

fn scene_cut_icon(raster_size: u32) -> ColorImage {
    // Two picture frames separated by a dashed cut line, in the supplied #c8c8c8
    // monochrome style (frame outlines with a small mountain, like img.png).
    let source = RgbaImage::from_fn(256, 256, |x, y| {
        let (x, y) = (x as i32, y as i32);
        let frame = |left: i32, right: i32| {
            let inside = (left..right).contains(&x) && (52..204).contains(&y);
            let hollow = (left + 14..right - 14).contains(&x) && (66..190).contains(&y);
            let peak = (left + right) / 2;
            let mountain = hollow && y >= 190 - (48 - (x - peak).abs()).max(0);
            inside && (!hollow || mountain)
        };
        let cut = (122..134).contains(&x) && (20..236).contains(&y) && (y / 22) % 2 == 0;
        let filled = frame(8, 112) || frame(144, 248) || cut;
        image::Rgba([200, 200, 200, if filled { 255 } else { 0 }])
    });
    resize(source, raster_size)
}

fn resize(mut rgba: RgbaImage, raster_size: u32) -> ColorImage {
    // The glyph in many supplied PNGs lives entirely in alpha; hidden RGB is
    // constant gray or white. Filter premultiplied sRGBA so transparent padding
    // cannot bleed into the edges. Color32 uses the same gamma-premultiplied
    // representation, so no second alpha multiplication is performed.
    for pixel in rgba.pixels_mut() {
        let alpha = u16::from(pixel[3]);
        for channel in &mut pixel.0[..3] {
            *channel = ((u16::from(*channel) * alpha + 127) / 255) as u8;
        }
    }
    let mut resized =
        image::imageops::resize(&rgba, raster_size, raster_size, FilterType::Lanczos3);
    // Lanczos ringing must not turn fully transparent edge pixels additive.
    for pixel in resized.pixels_mut() {
        let alpha = pixel[3];
        for channel in &mut pixel.0[..3] {
            *channel = (*channel).min(alpha);
        }
    }
    ColorImage::from_rgba_premultiplied(
        [raster_size as usize, raster_size as usize],
        resized.as_raw(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_padding_does_not_tint_a_resampled_glyph() {
        let image = RgbaImage::from_raw(2, 1, vec![128, 128, 128, 255, 255, 255, 255, 0])
            .expect("two pixels");
        let result = resize(image, 1);
        let [r, g, b, a] = result.pixels[0].to_array();
        assert!((127..=128).contains(&a));
        assert!((63..=65).contains(&r));
        assert_eq!((r, g), (g, b));
    }

    #[test]
    fn supplied_pngs_decode_at_hidpi_with_transparency_intact() {
        let images = EMBEDDED
            .into_iter()
            .map(|(name, png)| (name, decode(png, 32)))
            .chain(std::iter::once(("save_project", save_icon(32))));
        for (name, image) in images {
            assert_eq!(image.size, [32, 32], "{name}");
            assert!(image.pixels.iter().any(|pixel| pixel.a() > 0), "{name}");
            assert!(image.pixels.iter().any(|pixel| pixel.a() < 255), "{name}");
            assert!(
                image.pixels.iter().all(|pixel| {
                    let [r, g, b, a] = pixel.to_array();
                    r <= a && g <= a && b <= a
                }),
                "{name} has invalid premultiplied edges"
            );
        }
    }
}
