//! Reading and writing `bench/results/*.json`.

use std::path::{Path, PathBuf};

use crate::record::Run;

/// Writes `run` as pretty JSON to `dir/<run.file_name()>`, creating `dir`.
pub fn write_run(dir: &Path, run: &Run) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(run.file_name());
    std::fs::write(&path, serde_json::to_string_pretty(run)? + "\n")?;
    Ok(path)
}

/// Loads every `*.json` run in `dir` (non-JSON files are ignored), oldest first.
/// A missing directory yields no runs.
pub fn load_runs(dir: &Path) -> anyhow::Result<Vec<Run>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut runs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let text = std::fs::read_to_string(&path)?;
            let run: Run = serde_json::from_str(&text)
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            runs.push(run);
        }
    }
    runs.sort_by(|a, b| a.date.cmp(&b.date));
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::sample_run;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gpucomp-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn written_runs_load_back_oldest_first() {
        let dir = temp_dir("roundtrip");
        let newer = Run {
            date: "2026-10-08T00:00:00Z".into(),
            ..sample_run()
        };
        write_run(&dir, &newer).unwrap();
        write_run(&dir, &sample_run()).unwrap();
        std::fs::write(dir.join("README.txt"), "not a run").unwrap();
        assert_eq!(load_runs(&dir).unwrap(), [sample_run(), newer]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_run_uses_the_runs_file_name() {
        let dir = temp_dir("name");
        let path = write_run(&dir, &sample_run()).unwrap();
        assert_eq!(path, dir.join(sample_run().file_name()));
        assert!(path.is_file());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_directory_has_no_runs() {
        assert!(load_runs(&temp_dir("missing")).unwrap().is_empty());
    }
}
