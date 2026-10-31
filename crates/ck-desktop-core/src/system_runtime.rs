//! Optional, read-only discovery of Microsoft's installed shared Windows ML.
//! No shell, SDK bootstrap, registration, download, or GPU execution occurs.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedOrtRuntime {
    pub library: PathBuf,
    pub version: String,
}

/// A compatible shared Windows ML library, or None when the native fallback is
/// needed. Cached once per process; restarting detects runtime servicing.
pub fn shared_ort_runtime() -> Option<SharedOrtRuntime> {
    #[cfg(windows)]
    {
        use std::sync::OnceLock;
        static RUNTIME: OnceLock<Option<SharedOrtRuntime>> = OnceLock::new();
        RUNTIME.get_or_init(windows::discover).clone()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

pub fn shared_ort_library() -> Option<PathBuf> {
    shared_ort_runtime().map(|runtime| runtime.library)
}

#[cfg(windows)]
mod windows {
    use super::SharedOrtRuntime;
    use std::{
        ffi::{CStr, OsString, c_char, c_void},
        os::windows::ffi::{OsStrExt, OsStringExt},
        path::{Path, PathBuf},
        ptr,
    };
    use windows_sys::Win32::{
        Foundation::{ERROR_INSUFFICIENT_BUFFER, ERROR_SUCCESS, FreeLibrary, HMODULE},
        Storage::Packaging::Appx::{GetPackagePathByFullName, GetPackagesByPackageFamily},
        System::LibraryLoader::{
            GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
            LoadLibraryExW,
        },
    };

    // Stable Microsoft publisher identity + package families. Installation
    // versions and their protected paths are always obtained from Windows.
    const FAMILIES: [&str; 3] = [
        "Microsoft.WindowsAppRuntime.2_8wekyb3d8bbwe",
        "Microsoft.WindowsAppRuntime.1.8_8wekyb3d8bbwe",
        "Microsoft.WindowsAppRuntime.CBS.2_8wekyb3d8bbwe",
    ];

    pub(super) fn discover() -> Option<SharedOrtRuntime> {
        for family in FAMILIES {
            let mut packages = packages(family);
            packages.sort_by_key(|name| std::cmp::Reverse(package_version(name)));
            for package in packages {
                if !native_architecture(&package) {
                    continue;
                }
                if let Some(path) = package_path(&package)
                    && let Some(runtime) = probe(&path.join("onnxruntime.dll"))
                {
                    return Some(runtime);
                }
            }
        }
        None
    }

    fn packages(family: &str) -> Vec<String> {
        let family: Vec<u16> = family.encode_utf16().chain(Some(0)).collect();
        for _ in 0..3 {
            let mut count = 0;
            let mut length = 0;
            // SAFETY: null output buffers explicitly request sizes; the family
            // is a terminated UTF-16 string and both size pointers are live.
            let status = unsafe {
                GetPackagesByPackageFamily(
                    family.as_ptr(),
                    &mut count,
                    ptr::null_mut(),
                    &mut length,
                    ptr::null_mut(),
                )
            };
            if status != ERROR_INSUFFICIENT_BUFFER
                || count == 0
                || count > 4096
                || length > 4 * 1024 * 1024
            {
                return Vec::new();
            }
            let mut names = vec![ptr::null_mut(); count as usize];
            let mut buffer = vec![0_u16; length as usize];
            // SAFETY: both output buffers exactly match the requested sizes.
            let status = unsafe {
                GetPackagesByPackageFamily(
                    family.as_ptr(),
                    &mut count,
                    names.as_mut_ptr(),
                    &mut length,
                    buffer.as_mut_ptr(),
                )
            };
            if status == ERROR_INSUFFICIENT_BUFFER {
                continue;
            }
            if status != ERROR_SUCCESS {
                return Vec::new();
            }
            return names
                .into_iter()
                .take(count as usize)
                .filter_map(|name| {
                    // Returned pointers refer into our buffer. Check the numeric
                    // range before reading through the owned slice, never raw data.
                    let bytes = (name as usize).checked_sub(buffer.as_ptr() as usize)?;
                    if bytes % 2 != 0 {
                        return None;
                    }
                    let remaining = buffer.get(bytes / 2..)?;
                    let end = remaining.iter().position(|character| *character == 0)?;
                    String::from_utf16(&remaining[..end]).ok()
                })
                .collect();
        }
        Vec::new()
    }

    fn package_path(name: &str) -> Option<PathBuf> {
        let name: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
        for _ in 0..3 {
            let mut length = 0;
            // SAFETY: terminated input; null output requests the required size.
            let status =
                unsafe { GetPackagePathByFullName(name.as_ptr(), &mut length, ptr::null_mut()) };
            if status != ERROR_INSUFFICIENT_BUFFER || length == 0 || length > 32768 {
                return None;
            }
            let mut buffer = vec![0_u16; length as usize];
            // SAFETY: writable buffer has the length required by Windows.
            let status = unsafe {
                GetPackagePathByFullName(name.as_ptr(), &mut length, buffer.as_mut_ptr())
            };
            if status == ERROR_INSUFFICIENT_BUFFER {
                continue;
            }
            if status != ERROR_SUCCESS {
                return None;
            }
            let end = buffer.iter().position(|character| *character == 0)?;
            let path = PathBuf::from(OsString::from_wide(&buffer[..end]));
            return path.is_absolute().then_some(path);
        }
        None
    }

    fn package_version(name: &str) -> [u64; 4] {
        let mut version = [0; 4];
        if let Some(value) = name.split('_').nth(1) {
            for (destination, part) in version.iter_mut().zip(value.split('.')) {
                *destination = part.parse().unwrap_or(0);
            }
        }
        version
    }

    fn native_architecture(name: &str) -> bool {
        let expected = match std::env::consts::ARCH {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            "x86" => "x86",
            _ => return false,
        };
        name.split('_').nth(2) == Some(expected)
    }

    struct Library(HMODULE);
    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: this guard owns one successful LoadLibraryEx reference.
            unsafe {
                FreeLibrary(self.0);
            }
        }
    }

    #[repr(C)]
    struct OrtApiBase {
        get_api: unsafe extern "system" fn(u32) -> *const c_void,
        get_version_string: unsafe extern "system" fn() -> *const c_char,
    }

    pub(super) fn probe(path: &Path) -> Option<SharedOrtRuntime> {
        if !path.is_absolute() || !path.is_file() {
            return None;
        }
        let filename: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: the full package path is terminated UTF-16. Restrict dependency
        // search to the DLL's directory and normal safe loader directories.
        let module = unsafe {
            LoadLibraryExW(
                filename.as_ptr(),
                ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        };
        if module.is_null() {
            return None;
        }
        let module = Library(module);
        // SAFETY: module stays loaded for this scope; these are static export names.
        let get_base = unsafe { GetProcAddress(module.0, c"OrtGetApiBase".as_ptr().cast()) }?;
        let dml = unsafe {
            GetProcAddress(
                module.0,
                c"OrtSessionOptionsAppendExecutionProvider_DML"
                    .as_ptr()
                    .cast(),
            )
        };
        dml?;
        type GetBase = unsafe extern "system" fn() -> *const OrtApiBase;
        // SAFETY: OrtGetApiBase's public C ABI matches this function signature.
        let get_base: GetBase = unsafe { std::mem::transmute(get_base) };
        // SAFETY: documented ORT entry point; no environment/session is created.
        let base = unsafe { get_base().as_ref() }?;
        // SAFETY: version is a library-owned terminated string while module lives.
        let version = unsafe { (base.get_version_string)() };
        if version.is_null() {
            return None;
        }
        let version = unsafe { CStr::from_ptr(version) }.to_str().ok()?.to_owned();
        // Avoid generating ORT error logs for the legacy in-box API17 runtime.
        let mut parts = version.split('.');
        let major = parts.next()?.parse::<u32>().ok()?;
        let minor = parts.next()?.parse::<u32>().ok()?;
        if (major, minor) < (1, 22) {
            return None;
        }
        // SAFETY: querying the public API table does not execute a model or GPU.
        if unsafe { (base.get_api)(22) }.is_null() {
            return None;
        }
        Some(SharedOrtRuntime {
            library: path.to_path_buf(),
            version,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn rejects_missing_and_non_ort_libraries_without_gpu_execution() {
            assert!(probe(Path::new("relative/onnxruntime.dll")).is_none());
            let system = std::env::var_os("SystemRoot")
                .map(PathBuf::from)
                .unwrap()
                .join("System32");
            assert!(probe(&system.join("not-a-real-brp-runtime.dll")).is_none());
            assert!(probe(&system.join("kernel32.dll")).is_none());
        }
        #[test]
        fn package_api_discovery_returns_a_compatible_native_runtime_when_installed() {
            if let Some(runtime) = discover() {
                assert!(runtime.library.is_absolute());
                let checked = probe(&runtime.library).unwrap();
                assert_eq!(checked.version, runtime.version);
                eprintln!(
                    "Shared Windows ML: {} ({})",
                    runtime.library.display(),
                    runtime.version
                );
                assert_eq!(super::super::shared_ort_library(), Some(runtime.library));
            }
            assert_eq!(
                package_version("Microsoft.WindowsAppRuntime.2_2.10.3.0_x64__8wekyb3d8bbwe"),
                [2, 10, 3, 0]
            );
            assert!(!native_architecture(
                "Microsoft.WindowsAppRuntime.2_2.5.1.0_neutral__8wekyb3d8bbwe"
            ));
        }
    }
}
