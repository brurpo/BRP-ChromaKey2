//! Viewer zoom is measured in physical display pixels, independently of UI DPI.
use eframe::egui::{self, Pos2, Rect, Vec2};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ViewerZoom {
    /// Follow layout while opening: native pixels if they fit, otherwise Fit.
    #[default]
    Automatic,
    Fit,
    Fixed(f32),
    Custom(f32),
}

impl ViewerZoom {
    pub fn fit_scale(canvas: Vec2, source: Vec2, pixels_per_point: f32) -> f32 {
        let density = pixels_per_point.max(0.01);
        (canvas.x.max(1.0) * density / source.x.max(1.0))
            .min(canvas.y.max(1.0) * density / source.y.max(1.0))
    }

    pub fn physical_scale(self, fit: f32) -> f32 {
        match self {
            Self::Automatic => fit.min(1.0),
            Self::Fit => fit,
            Self::Fixed(scale) | Self::Custom(scale) => scale,
        }
        .clamp(0.005, 64.0)
    }

    pub fn label(self, fit: f32) -> String {
        match self {
            Self::Fit => format!("Fit ({:.0}%)", fit * 100.0),
            Self::Automatic if fit < 1.0 => format!("Fit ({:.0}%)", fit * 100.0),
            Self::Automatic => "100%".into(),
            Self::Fixed(scale) => format!("{:.0}%", scale * 100.0),
            Self::Custom(scale) => format!("Custom ({:.1}%)", scale * 100.0),
        }
    }

    pub fn dropdown(&mut self, ui: &mut egui::Ui, fit: f32) -> bool {
        let previous = *self;
        egui::ComboBox::from_id_salt("viewer_zoom")
            .width(112.0)
            .selected_text(self.label(fit))
            .show_ui(ui, |ui| {
                let fitted = matches!(*self, Self::Fit) || (matches!(*self, Self::Automatic) && fit < 1.0);
                if ui.selectable_label(fitted, format!("Fit ({:.0}%)", fit * 100.0)).clicked() { *self = Self::Fit; }
                for percent in [10, 25, 50, 75, 100, 150, 200, 400, 800] {
                    let value = Self::Fixed(percent as f32 / 100.0);
                    let selected = *self == value || (matches!(*self, Self::Automatic) && fit >= 1.0 && percent == 100);
                    if ui.selectable_label(selected, format!("{percent}%")).clicked() { *self = value; }
                }
                if let Self::Custom(_) = *self {
                    let value = *self;
                    ui.selectable_value(self, value, value.label(fit));
                }
            })
            .response
            .on_hover_text("100% = one source pixel per physical display pixel. Wheel zoom creates a Custom value.");
        previous != *self
    }
}

pub fn display_path(path: &std::path::Path) -> String {
    let text = path.to_string_lossy();
    if let Some(unc) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{unc}")
    } else {
        text.strip_prefix(r"\\?\").unwrap_or(&text).to_owned()
    }
}

pub fn image_rect(center: Pos2, source_pixels: Vec2, physical_scale: f32, density: f32) -> Rect {
    let size = source_pixels * (physical_scale / density);
    let rect = Rect::from_center_size(center, size);
    if (physical_scale - 1.0).abs() < 0.00001 {
        // 1:1 dimensions alone are insufficient: a half-pixel origin would
        // make linear texture filtering blend neighboring source pixels.
        let origin = (rect.min.to_vec2() * density).round() / density;
        Rect::from_min_size(Pos2::new(origin.x, origin.y), size)
    } else {
        rect
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_pixels_are_one_to_one_at_fractional_and_double_dpi() {
        for density in [1.0, 1.25, 1.5, 2.0] {
            let source = Vec2::new(1920.0, 1080.0);
            let logical = source * ViewerZoom::Fixed(1.0).physical_scale(0.5) / density;
            assert_eq!(logical * density, source);
        }
    }

    #[test]
    fn opening_chooses_native_or_fit_from_physical_viewport_size() {
        let canvas = Vec2::new(800.0, 500.0);
        let source = Vec2::new(1280.0, 720.0);
        let fit = ViewerZoom::fit_scale(canvas, source, 1.0);
        assert_eq!(ViewerZoom::Automatic.physical_scale(fit), 0.625);
        let fit = ViewerZoom::fit_scale(canvas, source, 2.0);
        assert_eq!(ViewerZoom::Automatic.physical_scale(fit), 1.0);
        assert_eq!(ViewerZoom::Fit.physical_scale(fit), 1.25);
        assert_eq!(ViewerZoom::Fit.label(fit), "Fit (125%)");
    }

    #[test]
    fn native_zoom_aligns_pixel_origin_as_well_as_dimensions() {
        for density in [1.0, 1.25, 1.5, 2.0] {
            let rect = image_rect(
                Pos2::new(501.3, 300.1),
                Vec2::new(853.0, 479.0),
                1.0,
                density,
            );
            let physical_origin = rect.min.to_vec2() * density;
            assert!((physical_origin - physical_origin.round()).length() < 0.001);
            assert!((rect.size() * density - Vec2::new(853.0, 479.0)).length() < 0.001);
        }
    }

    #[test]
    fn windows_path_display_preserves_drive_and_unc_without_device_prefix() {
        assert_eq!(
            display_path(std::path::Path::new(r"\\?\D:\footage\clip.mov")),
            r"D:\footage\clip.mov"
        );
        assert_eq!(
            display_path(std::path::Path::new(r"\\?\UNC\server\share\clip.mov")),
            r"\\server\share\clip.mov"
        );
        assert_eq!(
            display_path(std::path::Path::new("/home/media/clip.mov")),
            "/home/media/clip.mov"
        );
    }
}
