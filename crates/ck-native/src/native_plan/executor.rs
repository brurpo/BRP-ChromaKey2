//! Safe wrapper over the raw-DirectML plan executor DLL (`ck_plan_executor.dll`, C ABI
//! version 1). One session owns its device, heap, weights and compiled operators; callers
//! serialise GPU work (single GPU owner).

use anyhow::{Context, Result, bail, ensure};
use libloading::{Library, Symbol};
use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;

type Create = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    c_int,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type IoCount = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type Io = unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut c_char, usize, *mut i64) -> c_int;
type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut f64, *mut f64) -> c_int;
type Run = unsafe extern "C" fn(
    *mut c_void,
    *const *const f32,
    *const *mut f32,
    *mut f64,
    *mut c_char,
    usize,
) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void);

#[derive(Clone, Debug)]
pub struct IoInfo {
    pub name: String,
    pub shape: Vec<usize>,
}
impl IoInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SetupStats {
    pub device_bytes: u64,
    pub compile_ms: f64,
    pub bind_ms: f64,
}

pub struct PlanExecutor {
    handle: *mut c_void,
    run: Run,
    destroy: Destroy,
    pub inputs: Vec<IoInfo>,
    pub outputs: Vec<IoInfo>,
    pub stats: SetupStats,
    _lib: Library,
}

fn message(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

impl PlanExecutor {
    /// `adapter` is the DXGI adapter index (the app's selected device id); -1 selects WARP.
    pub fn create(
        executor_dll: &Path,
        directml_dll: &Path,
        plan_dir: &Path,
        adapter: i32,
    ) -> Result<Self> {
        ensure!(
            executor_dll.is_absolute() && directml_dll.is_absolute(),
            "executor and DirectML paths must be absolute"
        );
        // SAFETY: loading an explicit, absolute DLL path that exports the version-1 C ABI.
        let lib = unsafe { Library::new(executor_dll) }
            .with_context(|| format!("load {}", executor_dll.display()))?;
        unsafe {
            let version: Symbol<unsafe extern "C" fn() -> c_int> =
                lib.get(b"ck_exec_abi_version")?;
            ensure!(
                version() == 1,
                "unsupported plan executor ABI {}",
                version()
            );
            // Optional export: older ABI-1 DLLs only read CK_NO_METACOMMANDS at load.
            let disable_metacommands = crate::dml_policy::metacommands_disabled()?;
            match lib.get::<unsafe extern "C" fn(c_int)>(b"ck_exec_set_disable_metacommands") {
                Ok(set) => set(c_int::from(disable_metacommands)),
                Err(_) => ensure!(
                    !disable_metacommands,
                    "plan executor cannot disable DirectML metacommands; update ck_plan_executor.dll"
                ),
            }
            let create: Symbol<Create> = lib.get(b"ck_exec_create")?;
            let io_count: Symbol<IoCount> = lib.get(b"ck_exec_io_count")?;
            let io: Symbol<Io> = lib.get(b"ck_exec_io")?;
            let stats: Symbol<Stats> = lib.get(b"ck_exec_stats")?;
            let run: Run = *lib.get::<Run>(b"ck_exec_run")?;
            let destroy: Destroy = *lib.get::<Destroy>(b"ck_exec_destroy")?;
            let plan = CString::new(plan_dir.to_str().context("plan path is not UTF-8")?)?;
            let dml = CString::new(
                directml_dll
                    .to_str()
                    .context("DirectML path is not UTF-8")?,
            )?;
            let mut err = [0u8; 1024];
            let mut handle = std::ptr::null_mut();
            if create(
                plan.as_ptr(),
                dml.as_ptr(),
                adapter,
                &mut handle,
                err.as_mut_ptr().cast(),
                err.len(),
            ) != 0
            {
                bail!("plan executor setup failed: {}", message(&err));
            }
            let list = |out: c_int| -> Result<Vec<IoInfo>> {
                (0..io_count(handle, out))
                    .map(|i| {
                        let mut name = [0u8; 256];
                        let mut shape = [0i64; 8];
                        let rank = io(
                            handle,
                            out,
                            i,
                            name.as_mut_ptr().cast(),
                            name.len(),
                            shape.as_mut_ptr(),
                        );
                        ensure!(rank >= 0, "bad io index {i}");
                        Ok(IoInfo {
                            name: message(&name),
                            shape: shape[..rank as usize].iter().map(|&d| d as usize).collect(),
                        })
                    })
                    .collect()
            };
            let (inputs, outputs) = match (list(0), list(1)) {
                (Ok(i), Ok(o)) => (i, o),
                (Err(e), _) | (_, Err(e)) => {
                    destroy(handle);
                    return Err(e);
                }
            };
            let mut st = SetupStats {
                device_bytes: 0,
                compile_ms: 0.0,
                bind_ms: 0.0,
            };
            stats(
                handle,
                &mut st.device_bytes,
                &mut st.compile_ms,
                &mut st.bind_ms,
            );
            Ok(Self {
                handle,
                run,
                destroy,
                inputs,
                outputs,
                stats: st,
                _lib: lib,
            })
        }
    }

    /// One forward pass. `inputs`/`outputs` follow `self.inputs`/`self.outputs` order and
    /// sizes. Returns the GPU time in milliseconds.
    pub fn run(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<f64> {
        ensure!(
            inputs.len() == self.inputs.len() && outputs.len() == self.outputs.len(),
            "io count mismatch"
        );
        for (x, info) in inputs.iter().zip(&self.inputs) {
            ensure!(
                x.len() == info.elements(),
                "input {} has {} values, expected {}",
                info.name,
                x.len(),
                info.elements()
            );
        }
        for (x, info) in outputs.iter().zip(&self.outputs) {
            ensure!(
                x.len() == info.elements(),
                "output {} has {} values, expected {}",
                info.name,
                x.len(),
                info.elements()
            );
        }
        let ins: Vec<*const f32> = inputs.iter().map(|x| x.as_ptr()).collect();
        let outs: Vec<*mut f32> = outputs.iter_mut().map(|x| x.as_mut_ptr()).collect();
        let mut err = [0u8; 1024];
        let mut gpu_ms = 0.0;
        // SAFETY: pointer arrays and buffers are sized to the plan's io specs checked above.
        let rc = unsafe {
            (self.run)(
                self.handle,
                ins.as_ptr(),
                outs.as_ptr(),
                &mut gpu_ms,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        ensure!(rc == 0, "plan executor run failed: {}", message(&err));
        Ok(gpu_ms)
    }
}

// SAFETY: the session owns free-threaded D3D12/DirectML objects and is only used through
// `&mut self`, so moving it to another thread (graph loader threads) is sound; it is not Sync.
unsafe impl Send for PlanExecutor {}

impl Drop for PlanExecutor {
    fn drop(&mut self) {
        // SAFETY: handle came from ck_exec_create and is destroyed exactly once.
        unsafe { (self.destroy)(self.handle) }
    }
}
