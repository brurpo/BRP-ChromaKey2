use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimePaths {
    pub worker: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub ort_library: PathBuf,
    pub ck_1080_models: PathBuf,
    pub ck_4k_models: PathBuf,
    pub sam_models: PathBuf,
    pub transnet_model: PathBuf,
    pub cuda_library_directory: Option<PathBuf>,
}

fn first_existing(paths: impl IntoIterator<Item = PathBuf>, fallback: PathBuf) -> PathBuf {
    paths
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or(fallback)
}
fn executable(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.into()
    }
}

impl RuntimePaths {
    /// Optional SAM 3.1 pack, beside the SAM 2.1 pack (`models/sam3.1`).
    pub fn sam31_models(&self) -> PathBuf {
        self.sam_models
            .parent()
            .map_or_else(|| PathBuf::from("sam3.1"), |models| models.join("sam3.1"))
    }

    /// True when the SAM 3.1 pack's tracker settings are present.
    pub fn sam31_available(&self) -> bool {
        self.sam31_models().join("tracker/config.json").is_file()
    }

    /// Optional prompt translator for SAM 3.1 text prompts (`models/translate/opus-mt-romance-en`).
    pub fn translate_models(&self) -> PathBuf {
        self.sam_models.parent().map_or_else(
            || PathBuf::from("translate/opus-mt-romance-en"),
            |models| models.join("translate").join("opus-mt-romance-en"),
        )
    }

    /// True when the prompt translator pack is present.
    pub fn translator_available(&self) -> bool {
        let dir = self.translate_models();
        [
            "encoder.onnx",
            "decoder.onnx",
            "source.spm",
            "vocab.json",
            "config.json",
        ]
        .iter()
        .all(|f| dir.join(f).is_file())
    }

    pub fn application_directory() -> PathBuf {
        std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    /// Managed installations have a single layout relative to the executable.
    /// This deliberately does not depend on the current directory or saved paths.
    pub fn installed(app_directory: impl AsRef<Path>) -> Self {
        let root = if app_directory.as_ref().is_absolute() {
            app_directory.as_ref().to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_default()
                .join(app_directory)
        };
        let library = if cfg!(windows) {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        Self {
            worker: root.join(executable("ck-native")),
            ffmpeg: root.join("media").join(executable("ffmpeg")),
            ffprobe: root.join("media").join(executable("ffprobe")),
            ort_library: root.join(library),
            ck_1080_models: root.join("models/corridorkey/dynamic"),
            ck_4k_models: root.join("models/corridorkey/dynamic"),
            sam_models: root.join("models/sam2"),
            transnet_model: root.join("models/transnet/model.onnx"),
            cuda_library_directory: root
                .join("onnxruntime_providers_cuda.dll")
                .is_file()
                .then_some(root),
        }
    }

    /// Installed files take precedence. Source-checkout fallbacks are used only
    /// when the explicitly supplied workspace contains this Cargo project.
    pub fn discover(workspace: impl AsRef<Path>) -> Self {
        let absolute_workspace = if workspace.as_ref().is_absolute() {
            workspace.as_ref().to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_default().join(workspace)
        };
        let workspace = absolute_workspace.as_path();
        let installed = Self::application_directory();
        let library = if cfg!(windows) {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        let packaged = installed.join("assets-manifest.json").is_file()
            || std::env::current_exe()
                .ok()
                .and_then(|path| path.file_stem().map(|stem| stem.to_os_string()))
                .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case("BRP-Chroma"))
            || installed.join(library).is_file()
                && installed.join(executable("ck-native")).is_file();
        if packaged {
            return Self::installed(installed);
        }
        let development =
            workspace.join("Cargo.toml").is_file() && workspace.join("crates/ck-native").is_dir();
        let base = workspace.to_path_buf();
        let mut result = Self {
            worker: first_existing(
                [
                    installed.join(executable("ck-native")),
                    base.join(executable("ck-native")),
                    base.join("target/release").join(executable("ck-native")),
                ],
                base.join(executable("ck-native")),
            ),
            ffmpeg: first_existing(
                [
                    base.join("media").join(executable("ffmpeg")),
                    base.join(executable("ffmpeg")),
                ],
                base.join("media").join(executable("ffmpeg")),
            ),
            ffprobe: first_existing(
                [
                    base.join("media").join(executable("ffprobe")),
                    base.join(executable("ffprobe")),
                ],
                base.join("media").join(executable("ffprobe")),
            ),
            ort_library: base.join(library),
            ..Self::default()
        };
        if development {
            let ffmpeg_dir = base.join(".tools/ffmpeg/ffmpeg-9.0.2-essentials_build/bin");
            if !result.ffmpeg.is_file() {
                result.ffmpeg = ffmpeg_dir.join(executable("ffmpeg"));
            }
            if !result.ffprobe.is_file() {
                result.ffprobe = ffmpeg_dir.join(executable("ffprobe"));
            }
            if !result.ort_library.is_file() {
                result.ort_library = base
                    .join(".dev/export/Lib/site-packages/onnxruntime/capi")
                    .join(library);
            }
        }
        result.cuda_library_directory = if base.join("onnxruntime_providers_cuda.dll").is_file() {
            Some(base.clone())
        } else if development {
            let source = base
                .parent()
                .unwrap_or(&base)
                .join("BRP_CHROMA/CK/venv/Lib/site-packages/torch/lib");
            source.is_dir().then_some(source)
        } else {
            None
        };
        let models = if development {
            base.join("artifacts")
        } else {
            base.join("models")
        };
        result.set_model_root(&models);
        result
    }

    pub fn set_model_root(&mut self, root: impl AsRef<Path>) {
        let absolute_root = if root.as_ref().is_absolute() {
            root.as_ref().to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_default().join(root)
        };
        let root = absolute_root.as_path();
        // A single-pack selection replaces only that pack, preserving the other
        // configured models. A bundle-root selection resolves all families.
        if root.join("video_config.json").is_file() {
            self.sam_models = root.to_path_buf();
            return;
        }
        let dynamic_pack = std::fs::read(root.join("pack.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .is_some_and(|metadata| metadata["kind"] == "corridorkey_dynamic");
        if dynamic_pack && root.join("stage1/model.onnx").is_file() {
            self.ck_1080_models = root.to_path_buf();
            self.ck_4k_models = root.to_path_buf();
            return;
        }
        if root.join("stage1/model.onnx").is_file() {
            let manifest = std::fs::read(root.join("stage1/manifest.json"))
                .ok()
                .and_then(|data| serde_json::from_slice::<serde_json::Value>(&data).ok());
            let dynamic_spatial = manifest.as_ref().is_some_and(|value| {
                value["dynamic_spatial"] == true
                    || value["inputs"].as_array().is_some_and(|inputs| {
                        inputs.iter().any(|input| {
                            input["name"] == "rgb"
                                && input["shape"].as_array().is_some_and(|shape| {
                                    shape.len() == 5
                                        && shape[3].as_str().is_some_and(|axis| !axis.is_empty())
                                        && shape[4].as_str().is_some_and(|axis| !axis.is_empty())
                                })
                        })
                    })
            });
            if dynamic_spatial {
                self.ck_1080_models = root.to_path_buf();
                self.ck_4k_models = root.to_path_buf();
                return;
            }
            let dimensions = manifest.as_ref().and_then(|value| {
                let input = value["inputs"]
                    .as_array()?
                    .iter()
                    .find(|input| input["name"] == "rgb")?;
                Some((input["shape"][3].as_u64()?, input["shape"][4].as_u64()?))
            });
            match dimensions {
                Some((2176, 3840)) => {
                    self.ck_4k_models = root.to_path_buf();
                    return;
                }
                Some((1152, 1920)) => {
                    self.ck_1080_models = root.to_path_buf();
                    return;
                }
                // Development bundle roots also contain the small synthetic
                // stage1 proof. It is not a desktop profile and must not hide
                // the real CK, SAM-video and TransNet packs beside it.
                _ => {}
            }
        }
        let dynamic = first_existing(
            [root.join("corridorkey/dynamic"), root.join("ck-dynamic")],
            root.join("corridorkey/dynamic"),
        );
        if dynamic.join("stage1/model.onnx").is_file() {
            self.ck_1080_models = dynamic.clone();
            self.ck_4k_models = dynamic;
        } else {
            self.ck_1080_models = first_existing(
                [
                    root.join("corridorkey/1080"),
                    root.join("ck-1080"),
                    root.join("highres-models-1080"),
                ],
                root.join("corridorkey/1080"),
            );
            self.ck_4k_models = first_existing(
                [
                    root.join("corridorkey/4k"),
                    root.join("ck-4k"),
                    root.join("highres-models-4k"),
                ],
                root.join("corridorkey/4k"),
            );
        }
        self.sam_models = [
            root.join("sam2"),
            root.join("sam2_video_cuda"),
            root.join("sam2_video"),
        ]
        .into_iter()
        .find(|path| path.join("video_config.json").is_file())
        .unwrap_or_else(|| root.join("sam2"));
        self.transnet_model = first_existing(
            [
                root.join("transnet/model.onnx"),
                root.join("transnetv2/model.onnx"),
            ],
            root.join("transnet/model.onnx"),
        );
    }

    pub fn ck_models_for(&self, width: u32, height: u32) -> Result<&Path> {
        ensure!(
            width > 0 && height > 0,
            "Source dimensions must be positive"
        );
        // The legacy field name remains compatible with existing job JSON.
        // Managed installations point it to one dynamic spatial model pack.
        Ok(&self.ck_1080_models)
    }
    /// Resolve paths from a user-supplied settings file against that file's
    /// directory before launching a worker from the installed binary folder.
    pub fn resolve_relative_to(&mut self, base: &Path) {
        for path in [
            &mut self.worker,
            &mut self.ffmpeg,
            &mut self.ffprobe,
            &mut self.ort_library,
            &mut self.ck_1080_models,
            &mut self.ck_4k_models,
            &mut self.sam_models,
            &mut self.transnet_model,
        ] {
            if path.is_relative() && !path.as_os_str().is_empty() {
                *path = base.join(&*path);
            }
        }
        if let Some(path) = &mut self.cuda_library_directory
            && path.is_relative()
            && !path.as_os_str().is_empty()
        {
            *path = base.join(&*path);
        }
    }
    pub fn validate_common(&self) -> Result<()> {
        for (name, path) in [
            ("Native worker", &self.worker),
            ("FFmpeg", &self.ffmpeg),
            ("FFprobe", &self.ffprobe),
            ("ONNX Runtime", &self.ort_library),
        ] {
            ensure!(path.is_file(), "{name} was not found: {}", path.display());
        }
        Ok(())
    }
    pub fn missing_components(&self) -> Vec<String> {
        let mut missing = Vec::new();
        for (name, path) in [
            ("Native worker", &self.worker),
            ("FFmpeg", &self.ffmpeg),
            ("FFprobe", &self.ffprobe),
            ("ONNX Runtime", &self.ort_library),
            ("TransNet model", &self.transnet_model),
        ] {
            if !path.is_file() {
                missing.push(name.to_string());
            }
        }
        for (name, path) in [
            ("CorridorKey", self.ck_1080_models.join("stage1/model.onnx")),
            ("SAM video", self.sam_models.join("video_config.json")),
        ] {
            if !path.is_file() {
                missing.push(name.to_string());
            }
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corridor_key_uses_one_dynamic_pack_for_all_positive_source_dimensions() {
        let runtime = RuntimePaths {
            ck_1080_models: PathBuf::from("models/corridorkey/dynamic"),
            ..Default::default()
        };
        for (width, height) in [
            (1280, 720),
            (854, 480),
            (720, 1280),
            (1920, 1080),
            (3840, 2160),
        ] {
            assert_eq!(
                runtime.ck_models_for(width, height).unwrap(),
                runtime.ck_1080_models
            );
        }
        assert!(runtime.ck_models_for(0, 720).is_err());
        assert!(runtime.ck_models_for(1280, 0).is_err());
    }

    #[test]
    fn sam31_pack_sits_beside_the_sam2_pack() {
        let root = std::env::temp_dir().join(format!("brp-sam31-fixture-{}", std::process::id()));
        let runtime = RuntimePaths::installed(&root);
        assert_eq!(runtime.sam31_models(), root.join("models/sam3.1"));
        assert!(!runtime.sam31_available());
        std::fs::create_dir_all(root.join("models/sam3.1/tracker")).unwrap();
        std::fs::write(root.join("models/sam3.1/tracker/config.json"), b"{}").unwrap();
        assert!(runtime.sam31_available());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn managed_installation_uses_one_root_even_before_files_are_downloaded() {
        let root = std::env::temp_dir().join("brp-managed-uninstalled-fixture");
        let first = RuntimePaths::installed(&root);
        let restarted = RuntimePaths::installed(&root);
        assert_eq!(first.worker, root.join(executable("ck-native")));
        assert_eq!(
            first.ffprobe,
            root.join("media").join(executable("ffprobe"))
        );
        assert_eq!(
            first.ck_1080_models,
            root.join("models/corridorkey/dynamic")
        );
        assert_eq!(first.ck_4k_models, first.ck_1080_models);
        assert_eq!(first.sam_models, root.join("models/sam2"));
        assert_eq!(
            first.transnet_model,
            root.join("models/transnet/model.onnx")
        );
        assert_eq!(restarted.ck_1080_models, first.ck_1080_models);
        assert_eq!(restarted.sam_models, first.sam_models);
    }
    #[test]
    fn model_discovery_skips_image_only_sam_and_single_pack_preserves_other_models() {
        let folder = std::env::temp_dir().join(format!(
            "brp-model-discovery-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(folder.join("sam2")).unwrap();
        std::fs::create_dir_all(folder.join("sam2_video_cuda")).unwrap();
        std::fs::write(folder.join("sam2_video_cuda/video_config.json"), b"{}").unwrap();
        let mut runtime = RuntimePaths::default();
        runtime.set_model_root(&folder);
        assert_eq!(runtime.sam_models, folder.join("sam2_video_cuda"));
        let ck = runtime.ck_1080_models.clone();
        runtime.set_model_root(folder.join("sam2_video_cuda"));
        assert_eq!(runtime.ck_1080_models, ck);
        std::fs::remove_file(folder.join("sam2_video_cuda/video_config.json")).unwrap();
        std::fs::remove_dir(folder.join("sam2_video_cuda")).unwrap();
        std::fs::remove_dir(folder.join("sam2")).unwrap();
        std::fs::remove_dir(folder).unwrap();
    }
    #[test]
    fn installed_settings_resolve_relative_paths_without_developer_roots() {
        let base = Path::new("/app");
        let mut runtime = RuntimePaths {
            worker: "ck-native".into(),
            ort_library: "onnxruntime.dll".into(),
            cuda_library_directory: Some("cuda".into()),
            ..RuntimePaths::default()
        };
        runtime.resolve_relative_to(base);
        assert_eq!(runtime.worker, base.join("ck-native"));
        assert_eq!(runtime.cuda_library_directory, Some(base.join("cuda")));
        assert!(runtime.sam_models.as_os_str().is_empty());
    }
    #[test]
    fn synthetic_stage1_beside_real_packs_does_not_hide_bundle_models() {
        let folder = std::env::temp_dir().join(format!(
            "brp-bundle-discovery-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for relative in [
            "stage1",
            "highres-models-1080",
            "highres-models-4k",
            "sam2_video_cuda",
            "transnet",
            "individual/stage1",
        ] {
            std::fs::create_dir_all(folder.join(relative)).unwrap();
        }
        std::fs::write(folder.join("stage1/model.onnx"), b"synthetic").unwrap();
        std::fs::write(
            folder.join("stage1/manifest.json"),
            br#"{"inputs":[{"name":"rgb","shape":[1,2,3,128,128]}]}"#,
        )
        .unwrap();
        std::fs::write(folder.join("sam2_video_cuda/video_config.json"), b"{}").unwrap();
        std::fs::write(folder.join("transnet/model.onnx"), b"model").unwrap();
        let mut runtime = RuntimePaths::default();
        runtime.set_model_root(&folder);
        assert_eq!(runtime.ck_1080_models, folder.join("highres-models-1080"));
        assert_eq!(runtime.ck_4k_models, folder.join("highres-models-4k"));
        assert_eq!(runtime.sam_models, folder.join("sam2_video_cuda"));
        assert_eq!(runtime.transnet_model, folder.join("transnet/model.onnx"));
        std::fs::write(
            folder.join("individual/stage1/model.onnx"),
            b"actual-profile",
        )
        .unwrap();
        std::fs::write(folder.join("individual/stage1/manifest.json"), br#"{"inputs":[{"name":"hint","shape":[1,2,1,1152,1920]},{"name":"rgb","shape":[1,2,3,1152,1920]}]}"#).unwrap();
        runtime.set_model_root(folder.join("individual"));
        assert_eq!(runtime.ck_1080_models, folder.join("individual"));
        assert_eq!(runtime.sam_models, folder.join("sam2_video_cuda"));
        assert_eq!(runtime.transnet_model, folder.join("transnet/model.onnx"));
        for relative in [
            "stage1/model.onnx",
            "stage1/manifest.json",
            "sam2_video_cuda/video_config.json",
            "transnet/model.onnx",
            "individual/stage1/model.onnx",
            "individual/stage1/manifest.json",
        ] {
            std::fs::remove_file(folder.join(relative)).unwrap();
        }
        for relative in [
            "stage1",
            "highres-models-1080",
            "highres-models-4k",
            "sam2_video_cuda",
            "transnet",
            "individual/stage1",
            "individual",
        ] {
            std::fs::remove_dir(folder.join(relative)).unwrap();
        }
        std::fs::remove_dir(folder).unwrap();
    }
}
