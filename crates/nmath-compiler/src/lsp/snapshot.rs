//! Read-only filesystem overlay for unsaved documents and private
//! configuration.
use std::{collections::HashMap, sync::Mutex, time::SystemTime};

use tsrs_vfs::{Entries, FS, FileInfo, FileMode, bundled, osvfs};

pub struct Snapshot {
    files: HashMap<String, String>,
    disk: Box<dyn FS>,
    reads: Mutex<HashMap<String, Option<String>>>,
}
impl Snapshot {
    pub fn new(files: HashMap<String, String>) -> Self {
        Self {
            files,
            disk: Box::new(bundled::wrap_fs(osvfs::fs())),
            reads: Mutex::new(HashMap::new()),
        }
    }
}
impl FS for Snapshot {
    fn use_case_sensitive_file_names(&self) -> bool {
        self.disk.use_case_sensitive_file_names()
    }

    fn file_exists(&self, path: &str) -> bool {
        self.files.contains_key(path) || self.disk.file_exists(path)
    }

    fn read_file(&self, path: &str) -> Option<String> {
        if let Some(text) = self.files.get(path) {
            return Some(text.clone());
        }
        self.reads
            .lock()
            .unwrap()
            .entry(path.into())
            .or_insert_with(|| self.disk.read_file(path))
            .clone()
    }

    fn directory_exists(&self, path: &str) -> bool {
        let prefix = format!("{}/", path.trim_end_matches('/'));
        self.files.keys().any(|p| p.starts_with(&prefix)) || self.disk.directory_exists(path)
    }

    fn get_accessible_entries(&self, path: &str) -> Entries {
        let mut entries = self.disk.get_accessible_entries(path);
        let prefix = format!("{}/", path.trim_end_matches('/'));
        for file in self.files.keys() {
            if let Some(rest) = file.strip_prefix(&prefix) {
                if let Some((dir, _)) = rest.split_once('/') {
                    if !entries.directories.iter().any(|d| d == dir) {
                        entries.directories.push(dir.into());
                    }
                } else if !entries.files.iter().any(|f| f == rest) {
                    entries.files.push(rest.into());
                }
            }
        }
        entries
    }

    fn realpath(&self, path: &str) -> String {
        if self.files.contains_key(path) {
            path.into()
        } else {
            self.disk.realpath(path)
        }
    }

    fn stat(&self, path: &str) -> Option<FileInfo> {
        if let Some(text) = self.files.get(path) {
            Some(FileInfo {
                name: path.rsplit('/').next().unwrap_or(path).into(),
                size: text.len() as i64,
                mode: FileMode::None,
                mod_time: None,
            })
        } else {
            self.disk.stat(path)
        }
    }

    fn write_file(&self, _: &str, _: &str) -> Result<(), String> {
        Err("Editor snapshots are read-only".into())
    }

    fn append_file(&self, _: &str, _: &str) -> Result<(), String> {
        Err("Editor snapshots are read-only".into())
    }

    fn remove(&self, _: &str) -> Result<(), String> {
        Err("Editor snapshots are read-only".into())
    }

    fn chtimes(&self, _: &str, _: SystemTime, _: SystemTime) -> Result<(), String> {
        Err("Editor snapshots are read-only".into())
    }
}
