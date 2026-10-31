use crate::{JobEvent, JobRequest};
use anyhow::{Context, Result, ensure};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    process::{Child, Command, Stdio},
};

#[cfg(windows)]
pub(crate) mod process_tree {
    use anyhow::{Context, Result};
    use std::{os::windows::io::AsRawHandle, process::Child};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, HANDLE},
        System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject, TerminateJobObject,
        },
    };
    pub struct ProcessTree(HANDLE);
    // The uniquely owned kernel handle is thread-independent; no pointer is dereferenced.
    unsafe impl Send for ProcessTree {}
    impl ProcessTree {
        pub fn new() -> Result<Self> {
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(std::io::Error::last_os_error())
                    .context("Creating native worker process group");
            }
            let result = Self(handle);
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if unsafe {
                SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    std::mem::size_of_val(&limits) as u32,
                )
            } == 0
            {
                return Err(std::io::Error::last_os_error())
                    .context("Setting native worker process lifetime");
            }
            Ok(result)
        }
        pub fn assign(&self, child: &Child) -> Result<()> {
            if unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle()) } == 0 {
                return Err(std::io::Error::last_os_error())
                    .context("Attaching native worker process group");
            }
            Ok(())
        }
        pub fn terminate(&self) {
            let _ = unsafe { TerminateJobObject(self.0, 1) };
        }
    }
    impl Drop for ProcessTree {
        fn drop(&mut self) {
            let _ = unsafe { CloseHandle(self.0) };
        }
    }
}

pub struct Worker {
    child: Option<Child>,
    job_id: String,
    events: File,
    offset: u64,
    pending: Vec<u8>,
    terminal_seen: bool,
    exited: bool,
    cancelled: bool,
    cancel_reported: bool,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
    #[cfg(windows)]
    tree: Option<process_tree::ProcessTree>,
}
impl Worker {
    /// Spawn the native worker; model inference never executes on the GUI thread.
    pub fn spawn(request: &JobRequest) -> Result<Self> {
        Self::spawn_on_device(request, 0)
    }

    /// Snapshot a provider-specific device index in this child process only.
    /// DirectML indices are the original DXGI indices returned by enumeration.
    pub fn spawn_on_device(request: &JobRequest, device_id: u32) -> Result<Self> {
        crate::parse_device_id(Some(&device_id.to_string()))?;
        let mut request = request.clone();
        let cwd = std::env::current_dir()?;
        request.runtime.resolve_relative_to(&cwd);
        if request.media.path.is_relative() {
            request.media.path = cwd.join(&request.media.path);
        }
        if request.output_dir.is_relative() {
            request.output_dir = cwd.join(&request.output_dir);
        }
        if let crate::JobKind::CorridorKey { hints, .. } = &mut request.kind {
            match hints {
                crate::HintInput::FirstFrame { mask_file, .. } if mask_file.is_relative() => {
                    *mask_file = cwd.join(&*mask_file);
                }
                crate::HintInput::Video { mask_directory, .. } if mask_directory.is_relative() => {
                    *mask_directory = cwd.join(&*mask_directory);
                }
                _ => {}
            }
        }
        request.validate()?;
        request.runtime.validate_common()?;
        fs::create_dir_all(&request.output_dir)?;
        ensure!(
            fs::read_dir(&request.output_dir)?.next().is_none(),
            "Choose an empty output folder for this job; existing files will not be overwritten"
        );
        let job_file = request.output_dir.join("desktop-job.json");
        let events_file = request.output_dir.join("desktop-events.jsonl");
        let gate = request.output_dir.join("desktop-start.gate");
        ensure!(
            !job_file.exists() && !events_file.exists() && !gate.exists(),
            "Choose a new output folder for each job; an existing job will not be overwritten"
        );
        crate::project::write_json_atomic(&job_file, &request)?;
        File::options()
            .write(true)
            .create_new(true)
            .open(&events_file)?;
        let events = File::open(&events_file)?;
        let stdout_log = request.output_dir.join("worker.stdout.log");
        let stderr_log = request.output_dir.join("worker.stderr.log");
        let stdout = File::options()
            .write(true)
            .create_new(true)
            .open(&stdout_log)?;
        let stderr = File::options()
            .write(true)
            .create_new(true)
            .open(&stderr_log)?;
        let mut command = Command::new(&request.runtime.worker);
        command
            .arg("desktop")
            .arg("--job")
            .arg(&job_file)
            .arg("--events")
            .arg(&events_file)
            .arg("--start-gate")
            .arg(&gate)
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        if let Some(parent) = request.runtime.worker.parent() {
            command.current_dir(parent);
        }
        command.env("ORT_DYLIB_PATH", &request.runtime.ort_library);
        command.env(crate::DEVICE_ID_ENV, device_id.to_string());
        if let Some(cuda) = &request.runtime.cuda_library_directory {
            let mut paths = vec![cuda.clone()];
            if let Some(existing) = std::env::var_os("PATH") {
                paths.extend(std::env::split_paths(&existing));
            }
            command.env("PATH", std::env::join_paths(paths)?);
        }
        crate::media::hidden(&mut command);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        #[cfg(windows)]
        let tree = process_tree::ProcessTree::new()?;
        let mut child = command
            .spawn()
            .context("Starting native inference worker")?;
        #[cfg(windows)]
        if let Err(error) = tree.assign(&child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        // The worker waits before spawning FFmpeg, so the Job Object owns every
        // child before work begins and cancellation cannot leave decoders behind.
        if let Err(error) = fs::write(&gate, b"start\n") {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error.into());
        }
        Ok(Self {
            child: Some(child),
            job_id: request.job_id.clone(),
            events,
            offset: 0,
            pending: Vec::new(),
            terminal_seen: false,
            exited: false,
            cancelled: false,
            cancel_reported: false,
            stdout_log,
            stderr_log,
            #[cfg(windows)]
            tree: Some(tree),
        })
    }

    pub fn is_running(&self) -> bool {
        !self.exited
    }
    pub fn poll(&mut self) -> Result<Vec<JobEvent>> {
        if self.cancelled {
            if self.cancel_reported {
                return Ok(Vec::new());
            }
            self.cancel_reported = true;
            return Ok(vec![JobEvent::Cancelled {
                job_id: self.job_id.clone(),
            }]);
        }
        self.events.seek(SeekFrom::Start(self.offset))?;
        let mut bytes = Vec::new();
        (&mut self.events)
            .take(256 * 1024)
            .read_to_end(&mut bytes)?;
        self.offset += bytes.len() as u64;
        self.pending.extend(bytes);
        ensure!(
            self.pending.len() <= 1024 * 1024,
            "Native worker emitted an oversized progress record"
        );
        let mut result = Vec::new();
        while let Some(end) = self.pending.iter().position(|&b| b == b'\n') {
            let line: Vec<_> = self.pending.drain(..=end).collect();
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let event: JobEvent =
                serde_json::from_slice(&line).context("Reading native worker progress")?;
            ensure!(
                event.job_id() == self.job_id,
                "Native worker returned another job's progress"
            );
            self.terminal_seen |= event.is_terminal();
            result.push(event);
        }
        if !self.exited
            && let Some(status) = self
                .child
                .as_mut()
                .context("Worker child was released")?
                .try_wait()?
        {
            self.exited = true;
            if !self.terminal_seen {
                // Every proper worker writes its terminal event before exit.
                // Read any remaining records on the next GUI tick first.
                if self.events.metadata()?.len() > self.offset {
                    self.exited = false;
                } else {
                    self.terminal_seen = true;
                    result.push(JobEvent::Failed {
                        job_id: self.job_id.clone(),
                        message: format!(
                            "Native worker exited ({status}) without a completion record. See {}",
                            self.stderr_log.display()
                        ),
                    });
                }
            }
        }
        Ok(result)
    }
    pub fn cancel(&mut self) -> Result<()> {
        if self.cancelled {
            return Ok(());
        }
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        if child.try_wait()?.is_none() {
            #[cfg(windows)]
            if let Some(tree) = &self.tree {
                tree.terminate();
            }
            #[cfg(unix)]
            {
                unsafe extern "C" {
                    fn kill(pid: i32, signal: i32) -> i32;
                }
                if let Ok(pid) = i32::try_from(child.id()) {
                    let _ = unsafe { kill(-pid, 9) };
                }
            }
            let _ = child.kill();
            child.wait()?;
        }
        self.cancelled = true;
        self.exited = true;
        self.terminal_seen = true;
        Ok(())
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        if self.terminal_seen && !self.cancelled && !self.exited {
            // Completion can arrive just before the process exits. Keep the Job
            // Object alive while cleanup finishes, without blocking a GUI frame
            // or turning a successful job into a forced-termination exit code.
            if let Some(mut child) = self.child.take() {
                #[cfg(windows)]
                let tree = self.tree.take();
                std::thread::spawn(move || {
                    for _ in 0..150 {
                        match child.try_wait() {
                            Ok(Some(_)) => return,
                            Err(_) => break,
                            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
                        }
                    }
                    #[cfg(windows)]
                    if let Some(tree) = &tree {
                        tree.terminate();
                    }
                    #[cfg(unix)]
                    {
                        unsafe extern "C" {
                            fn kill(pid: i32, signal: i32) -> i32;
                        }
                        if let Ok(pid) = i32::try_from(child.id()) {
                            let _ = unsafe { kill(-pid, 9) };
                        }
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                    #[cfg(windows)]
                    drop(tree);
                });
            }
        } else if !self.exited {
            let _ = self.cancel();
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    #[test]
    fn windows_job_cancellation_terminates_an_owned_native_process() {
        let tree = process_tree::ProcessTree::new().unwrap();
        let mut command = Command::new(
            std::env::var_os("COMSPEC").unwrap_or_else(|| "C:\\Windows\\System32\\cmd.exe".into()),
        );
        command.args(["/D", "/C", "ping -n 30 127.0.0.1 >NUL"]);
        crate::media::hidden(&mut command);
        let mut child = command.spawn().unwrap();
        tree.assign(&child).unwrap();
        assert!(child.try_wait().unwrap().is_none());
        tree.terminate();
        assert!(!child.wait().unwrap().success());
    }
}
