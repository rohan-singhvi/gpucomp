//! A description of the host CPU for the run record.

/// The host CPU's model and the number of threads the CPU baselines use
/// (rayon's pool), e.g. `Apple M4 Pro, 12 threads`.
pub fn cpu() -> String {
    describe(brand().as_deref(), rayon::current_num_threads())
}

/// The CPU model as the OS reports it, if it does.
fn brand() -> Option<String> {
    match std::env::consts::OS {
        "macos" => std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned()),
        "windows" => std::env::var("PROCESSOR_IDENTIFIER").ok(),
        _ => std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| brand_from_cpuinfo(&text).map(str::to_string)),
    }
}

/// `<brand>, <threads> threads`.
fn describe(brand: Option<&str>, threads: usize) -> String {
    let brand = brand.map(str::trim).filter(|b| !b.is_empty());
    let unit = if threads == 1 { "thread" } else { "threads" };
    format!("{}, {threads} {unit}", brand.unwrap_or("unknown CPU"))
}

/// The first `model name` in Linux's `/proc/cpuinfo`.
fn brand_from_cpuinfo(text: &str) -> Option<&str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn description_names_the_brand_and_thread_count() {
        assert_eq!(
            describe(Some(" Apple M4 Pro\n"), 12),
            "Apple M4 Pro, 12 threads"
        );
        assert_eq!(describe(None, 1), "unknown CPU, 1 thread");
        assert_eq!(describe(Some(""), 8), "unknown CPU, 8 threads");
    }

    #[test]
    fn cpuinfo_brand_is_the_first_model_name() {
        let text = "processor\t: 0\nvendor_id\t: GenuineIntel\n\
                    model name\t: Intel(R) Core(TM) i9-13900K\nprocessor\t: 1\n\
                    model name\t: Intel(R) Core(TM) i9-13900K\n";
        assert_eq!(
            brand_from_cpuinfo(text),
            Some("Intel(R) Core(TM) i9-13900K")
        );
        assert_eq!(brand_from_cpuinfo("processor\t: 0\n"), None);
    }

    #[test]
    fn this_hosts_description_counts_rayons_threads() {
        let threads = rayon::current_num_threads();
        let suffix = if threads == 1 { "thread" } else { "threads" };
        assert!(
            cpu().ends_with(&format!(", {threads} {suffix}")),
            "{}",
            cpu()
        );
    }
}
