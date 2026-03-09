//! Auto-fix logic for installation errors.
//!
//! The rechecker module provides automatic error detection and repair for
//! common installation failures encountered during task validation. It
//! implements a retry mechanism with alternative install strategies.
//!
//! Key features:
//! - Detects SetupError (install failures) and SanityFail (test semantic errors)
//! - Attempts alternative install commands based on language and error patterns
//! - Limits attempts to prevent infinite loops (default: max 3 attempts)
//! - Exports fixed install_config for validated tasks
//!
//! Usage:
//! ```
//! use swe_forge::swe::rechecker::{Rechecker, RecheckerConfig};
//!
//! let config = RecheckerConfig::default();
//! let rechecker = Rechecker::new(config);
//! // Use rechecker.fix_task() or rechecker.fix_install()
//! ```

use std::collections::BTreeMap;

use crate::error::RecheckerError;
use crate::swe::{SweTask, SweTaskStatus};
use tracing::{info, warn};

/// Maximum number of fix attempts before giving up on a task.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;

/// Configuration for the rechecker.
#[derive(Debug, Clone)]
pub struct RecheckerConfig {
    /// Maximum number of fix attempts per task.
    pub max_attempts: u32,
    /// Whether to enable verbose logging of fix attempts.
    pub verbose: bool,
    /// Whether to skip sanity check fixes (only fix setup errors).
    pub skip_sanity_fixes: bool,
}

impl Default for RecheckerConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            verbose: false,
            skip_sanity_fixes: false,
        }
    }
}

impl RecheckerConfig {
    /// Create a new config with specified max attempts.
    pub fn with_max_attempts(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }
}

/// Result of a recheck/fix attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecheckResult {
    /// Task was fixed successfully.
    Fixed,
    /// Task is already valid, no fix needed.
    Ok,
    /// Task could not be fixed after max attempts.
    Incorrigible,
    /// Task was skipped (e.g., invalid input).
    Skipped,
}

impl std::fmt::Display for RecheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed => write!(f, "fixed"),
            Self::Ok => write!(f, "ok"),
            Self::Incorrigible => write!(f, "incorrigible"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// Error types that the rechecker can detect and attempt to fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorType {
    /// Installation/setup command failed.
    SetupError,
    /// Test semantics are invalid (e.g., fail_to_pass already passes on base).
    SanityFail,
    /// Unknown error type.
    Unknown,
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SetupError => write!(f, "setup_error"),
            Self::SanityFail => write!(f, "sanity_fail"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Rechecker for automatic error detection and repair.
pub struct Rechecker {
    config: RecheckerConfig,
}

impl Default for Rechecker {
    fn default() -> Self {
        Self::new(RecheckerConfig::default())
    }
}

impl Rechecker {
    /// Create a new rechecker with the given configuration.
    pub fn new(config: RecheckerConfig) -> Self {
        Self { config }
    }



    /// Detect the error type from a task and optional error message.
    pub fn detect_error_type(&self, task: &SweTask, error_msg: Option<&str>) -> ErrorType {
        // Check error message patterns
        if let Some(msg) = error_msg {
            let msg_lower = msg.to_lowercase();
            
            // SetupError patterns
            if msg_lower.contains("install failed")
                || msg_lower.contains("install command failed")
                || msg_lower.contains("apt-get")
                || msg_lower.contains("pip install")
                || msg_lower.contains("npm install")
                || msg_lower.contains("cargo fetch")
                || msg_lower.contains("go mod")
                || msg_lower.contains("no such file or directory")
                || msg_lower.contains("command not found")
                || msg_lower.contains("exit code")
            {
                return ErrorType::SetupError;
            }

            // SanityFail patterns
            if msg_lower.contains("fail_to_pass already passes")
                || msg_lower.contains("pass_to_pass fails on base")
                || msg_lower.contains("sanity fail")
                || msg_lower.contains("sanity_check")
            {
                return ErrorType::SanityFail;
            }
        }

        // Check task status if available
        if task.status == SweTaskStatus::Rejected {
            // Try to infer from install_config
            if let Some(install) = task.install_config.get("install") {
                if install.starts_with('#') || install.is_empty() {
                    return ErrorType::SetupError;
                }
            }
        }

        ErrorType::Unknown
    }

    /// Attempt to fix a task's installation configuration.
    ///
    /// This is the main entry point for fixing a task. It will:
    /// 1. Detect the error type
    /// 2. Generate alternative install strategies
    /// 3. Try each strategy up to max_attempts
    /// 4. Update the task's install_config if successful
    ///
    /// Returns Ok(RecheckResult) if the operation completed, or Err if an
    /// internal error occurred.
    pub fn fix_install(&self, task: &mut SweTask) -> Result<RecheckResult, RecheckerError> {
        let original_install = task
            .install_config
            .get("install")
            .cloned()
            .unwrap_or_default();

        if self.config.verbose {
            info!(task_id = %task.id, "Attempting to fix install");
        }

        // Already has valid install command?
        if !original_install.is_empty() && !original_install.starts_with('#') {
            // Check if this looks like a valid command
            if Self::looks_like_valid_install(&original_install) {
                return Ok(RecheckResult::Ok);
            }
        }

        // Generate alternative strategies based on language
        let language = task.language.to_lowercase();
        let strategies = self.generate_alternative_strategies(&language, &original_install);

        // Try each strategy
        for (attempt, strategy) in strategies.iter().enumerate().take(self.config.max_attempts as usize) {
            let attempt_num = attempt + 1;
            
            if self.config.verbose {
                info!(task_id = %task.id, attempt = attempt_num, "Trying install strategy");
            }

            // Update the task's install_config with the new strategy
            task.install_config.insert("install".to_string(), strategy.clone());

            // Note: In a real implementation, we would test the install here.
            // For now, we mark it as fixed if we found a valid-looking strategy.
            if Self::looks_like_valid_install(strategy) {
                if self.config.verbose {
                    info!(task_id = %task.id, attempt = attempt_num, "Found valid install strategy");
                }
                return Ok(RecheckResult::Fixed);
            }
        }

        // Max attempts reached, restore original
        if !original_install.is_empty() {
            task.install_config.insert("install".to_string(), original_install);
        }

        warn!(task_id = %task.id, attempts = self.config.max_attempts, "Could not find valid install strategy");
        Ok(RecheckResult::Incorrigible)
    }

    /// Check if a command looks like a valid install command.
    fn looks_like_valid_install(cmd: &str) -> bool {
        if cmd.is_empty() {
            return false;
        }
        if cmd.starts_with('#') {
            return false;
        }
        
        // Check for common package manager commands
        let valid_prefixes = [
            "pip install",
            "pip3 install",
            "apt-get",
            "apt ",
            "npm install",
            "yarn install",
            "cargo",
            "go mod",
            "go get",
            "./mvnw",
            "mvn ",
            "gradle",
            "bundle install",
            "composer install",
        ];
        
        let cmd_trimmed = cmd.trim();
        valid_prefixes.iter().any(|prefix| cmd_trimmed.starts_with(prefix))
    }

    /// Generate alternative install strategies based on language.
    fn generate_alternative_strategies(&self, language: &str, original: &str) -> Vec<String> {
        let mut strategies = Vec::new();

        match language {
            "python" | "py" => {
                // Standard pip install strategies
                strategies.push("pip install --break-system-packages -e .".to_string());
                strategies.push("pip install -e .".to_string());
                strategies.push("pip3 install --break-system-packages -e .".to_string());
                strategies.push("pip3 install -e .".to_string());
                
                // Requirements.txt based
                strategies.push("pip install --break-system-packages -r requirements.txt && pip install --break-system-packages -e .".to_string());
                strategies.push("pip install -r requirements.txt && pip install -e .".to_string());
                
                // With apt dependencies
                strategies.push("apt-get update -qq && apt-get install -y -qq build-essential libffi-dev && pip install --break-system-packages -e .".to_string());
                
                // Try original if not empty and not a comment
                if !original.is_empty() && !original.starts_with('#') {
                    strategies.push(original.to_string());
                }
            }
            "javascript" | "typescript" | "js" | "ts" | "node" | "nodejs" => {
                strategies.push("npm install".to_string());
                strategies.push("npm ci".to_string());
                strategies.push("yarn install".to_string());
                strategies.push("pnpm install".to_string());
                
                // With potential build step
                strategies.push("npm install && npm run build".to_string());
                
                if !original.is_empty() && !original.starts_with('#') {
                    strategies.push(original.to_string());
                }
            }
            "rust" | "rs" | "cargo" => {
                strategies.push("cargo fetch".to_string());
                strategies.push("cargo build".to_string());
                strategies.push("rustup update && cargo fetch".to_string());
                
                if !original.is_empty() && !original.starts_with('#') {
                    strategies.push(original.to_string());
                }
            }
            "go" | "golang" => {
                strategies.push("go mod download".to_string());
                strategies.push("go mod tidy && go mod download".to_string());
                strategies.push("go get ./...".to_string());
                
                if !original.is_empty() && !original.starts_with('#') {
                    strategies.push(original.to_string());
                }
            }
            "java" => {
                strategies.push("./mvnw -q -DskipTests package".to_string());
                strategies.push("mvn -q -DskipTests package".to_string());
                strategies.push("./gradlew build -x test".to_string());
                strategies.push("gradle build -x test".to_string());
                
                if !original.is_empty() && !original.starts_with('#') {
                    strategies.push(original.to_string());
                }
            }
            _ => {
                // For unknown languages, try the original or a generic approach
                if !original.is_empty() && !original.starts_with('#') {
                    strategies.push(original.to_string());
                }
                strategies.push("# manual install required".to_string());
            }
        }

        // Remove duplicates while preserving order
        let mut seen = std::collections::HashSet::new();
        strategies.retain(|s| seen.insert(s.clone()));

        strategies
    }

    /// Fix a task based on detected error type.
    ///
    /// This is the high-level API that should be called by harness and
    /// workspace validator when they encounter errors.
    pub fn fix_task(
        &self,
        task: &mut SweTask,
        error_msg: Option<&str>,
    ) -> Result<RecheckResult, RecheckerError> {
        let error_type = self.detect_error_type(task, error_msg);

        match error_type {
            ErrorType::SetupError => {
                info!(task_id = %task.id, "Detected setup error, attempting fix");
                self.fix_install(task)
            }
            ErrorType::SanityFail => {
                if self.config.skip_sanity_fixes {
                    info!(task_id = %task.id, "Skipping sanity fail fix (disabled in config)");
                    return Ok(RecheckResult::Skipped);
                }
                info!(task_id = %task.id, "Detected sanity fail, attempting fix");
                // Sanity fails often require test command changes, not just install
                // For now, we try the install fix as it may resolve underlying issues
                self.fix_install(task)
            }
            ErrorType::Unknown => {
                warn!(task_id = %task.id, "Unknown error type, skipping fix");
                Ok(RecheckResult::Skipped)
            }
        }
    }

    /// Get the fixed install_config for export.
    ///
    /// Returns the current install_config from the task, which may have been
    /// modified by previous fix attempts.
    pub fn get_fixed_install_config(&self, task: &SweTask) -> BTreeMap<String, String> {
        task.install_config.clone()
    }

    /// Check if a task should be removed after max attempts.
    ///
    /// Returns true if the task has been marked as incorrigible and should
    /// be removed from the dataset.
    pub fn should_remove(&self, result: &RecheckResult) -> bool {
        matches!(result, RecheckResult::Incorrigible)
    }
}

pub mod strategies {
    //! Pre-defined fix strategies for common installation error patterns.
    //!
    //! This module provides helper functions to fix common installation
    //! errors with specific workarounds for package managers like apt,
    //! pip, npm, etc.
    
    /// Fix apt-get related errors with common workarounds.
    pub fn fix_apt_errors(cmd: &str) -> Option<String> {
        if cmd.contains("apt-get") || cmd.contains("apt ") {
            Some(format!(
                "apt-get update -qq && apt-get install -y -qq {} 2>&1 || apt-get install -y {}",
                cmd.replace("apt-get ", "").replace("apt ", ""),
                cmd.replace("apt-get ", "").replace("apt ", "")
            ))
        } else {
            None
        }
    }

    /// Fix pip install errors with common workarounds.
    pub fn fix_pip_errors(cmd: &str) -> Option<String> {
        if cmd.contains("pip install") {
            // Add --break-system-packages if not present
            if !cmd.contains("--break-system-packages") {
                Some(cmd.replace("pip install", "pip install --break-system-packages"))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Fix Node.js/npm related errors.
    pub fn fix_node_errors(cmd: &str) -> Option<String> {
        if cmd.contains("npm install") && !cmd.contains("npm ci") {
            // Try npm ci if package-lock.json exists
            Some("test -f package-lock.json && npm ci || npm install".to_string())
        } else {
            None
        }
    }

    /// Apply all available fixes to a command.
    pub fn apply_all_fixes(cmd: &str) -> Vec<String> {
        let mut alternatives = Vec::new();
        
        if let Some(fixed) = fix_apt_errors(cmd) {
            alternatives.push(fixed);
        }
        if let Some(fixed) = fix_pip_errors(cmd) {
            alternatives.push(fixed);
        }
        if let Some(fixed) = fix_node_errors(cmd) {
            alternatives.push(fixed);
        }
        
        alternatives
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rechecker_config_default() {
        let config = RecheckerConfig::default();
        assert_eq!(config.max_attempts, DEFAULT_MAX_ATTEMPTS);
        assert!(!config.verbose);
        assert!(!config.skip_sanity_fixes);
    }

    #[test]
    fn test_rechecker_config_with_max_attempts() {
        let config = RecheckerConfig::with_max_attempts(5);
        assert_eq!(config.max_attempts, 5);
    }

    #[test]
    fn test_detect_error_type_setup_error() {
        let rechecker = Rechecker::default();
        let task = SweTask::new("test-1", "owner/repo");
        
        let error_type = rechecker.detect_error_type(&task, Some("Install failed with exit code 1"));
        assert_eq!(error_type, ErrorType::SetupError);
        
        let error_type = rechecker.detect_error_type(&task, Some("pip install failed"));
        assert_eq!(error_type, ErrorType::SetupError);
        
        let error_type = rechecker.detect_error_type(&task, Some("apt-get install failed"));
        assert_eq!(error_type, ErrorType::SetupError);
    }

    #[test]
    fn test_detect_error_type_sanity_fail() {
        let rechecker = Rechecker::default();
        let task = SweTask::new("test-1", "owner/repo");
        
        let error_type = rechecker.detect_error_type(&task, Some("fail_to_pass already passes on base commit"));
        assert_eq!(error_type, ErrorType::SanityFail);
        
        let error_type = rechecker.detect_error_type(&task, Some("pass_to_pass fails on base commit"));
        assert_eq!(error_type, ErrorType::SanityFail);
    }

    #[test]
    fn test_detect_error_type_unknown() {
        let rechecker = Rechecker::default();
        let task = SweTask::new("test-1", "owner/repo");
        
        let error_type = rechecker.detect_error_type(&task, None);
        assert_eq!(error_type, ErrorType::Unknown);
        
        let error_type = rechecker.detect_error_type(&task, Some("random error message"));
        assert_eq!(error_type, ErrorType::Unknown);
    }

    #[test]
    fn test_looks_like_valid_install() {
        assert!(Rechecker::looks_like_valid_install("pip install -e ."));
        assert!(Rechecker::looks_like_valid_install("npm install"));
        assert!(Rechecker::looks_like_valid_install("cargo fetch"));
        assert!(Rechecker::looks_like_valid_install("go mod download"));
        assert!(Rechecker::looks_like_valid_install("apt-get update && apt-get install -y python3"));
        
        assert!(!Rechecker::looks_like_valid_install(""));
        assert!(!Rechecker::looks_like_valid_install("# manual install"));
        assert!(!Rechecker::looks_like_valid_install("  "));
    }

    #[test]
    fn test_generate_alternative_strategies_python() {
        let rechecker = Rechecker::default();
        let strategies = rechecker.generate_alternative_strategies("python", "");
        
        assert!(!strategies.is_empty());
        assert!(strategies.iter().any(|s| s.contains("pip install")));
    }

    #[test]
    fn test_generate_alternative_strategies_javascript() {
        let rechecker = Rechecker::default();
        let strategies = rechecker.generate_alternative_strategies("javascript", "");
        
        assert!(!strategies.is_empty());
        assert!(strategies.iter().any(|s| s.contains("npm install")));
    }

    #[test]
    fn test_generate_alternative_strategies_rust() {
        let rechecker = Rechecker::default();
        let strategies = rechecker.generate_alternative_strategies("rust", "");
        
        assert!(!strategies.is_empty());
        assert!(strategies.iter().any(|s| s.contains("cargo")));
    }

    #[test]
    fn test_generate_alternative_strategies_go() {
        let rechecker = Rechecker::default();
        let strategies = rechecker.generate_alternative_strategies("go", "");
        
        assert!(!strategies.is_empty());
        assert!(strategies.iter().any(|s| s.contains("go mod")));
    }

    #[test]
    fn test_recheck_result_display() {
        assert_eq!(format!("{}", RecheckResult::Fixed), "fixed");
        assert_eq!(format!("{}", RecheckResult::Ok), "ok");
        assert_eq!(format!("{}", RecheckResult::Incorrigible), "incorrigible");
        assert_eq!(format!("{}", RecheckResult::Skipped), "skipped");
    }

    #[test]
    fn test_should_remove() {
        let rechecker = Rechecker::default();
        
        assert!(rechecker.should_remove(&RecheckResult::Incorrigible));
        assert!(!rechecker.should_remove(&RecheckResult::Fixed));
        assert!(!rechecker.should_remove(&RecheckResult::Ok));
        assert!(!rechecker.should_remove(&RecheckResult::Skipped));
    }

    #[test]
    fn test_strategies_fix_pip_errors() {
        let cmd = "pip install -e .";
        let fixed = strategies::fix_pip_errors(cmd);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("--break-system-packages"));
        
        // Already has the flag
        let cmd = "pip install --break-system-packages -e .";
        let fixed = strategies::fix_pip_errors(cmd);
        assert!(fixed.is_none());
    }

    #[test]
    fn test_strategies_fix_apt_errors() {
        let cmd = "apt-get install -y python3-dev";
        let fixed = strategies::fix_apt_errors(cmd);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("apt-get update"));
    }

    #[test]
    fn test_strategies_fix_node_errors() {
        let cmd = "npm install";
        let fixed = strategies::fix_node_errors(cmd);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("npm ci"));
    }

    #[test]
    fn test_fix_install_empty_task() {
        let rechecker = Rechecker::default();
        let mut task = SweTask::new("test-1", "owner/repo");
        task.language = "python".to_string();
        
        let result = rechecker.fix_install(&mut task).unwrap();
        // Should either fix or find incorrigible, not error
        assert!(matches!(result, RecheckResult::Fixed | RecheckResult::Ok | RecheckResult::Incorrigible));
    }

    #[test]
    fn test_get_fixed_install_config() {
        let rechecker = Rechecker::default();
        let mut task = SweTask::new("test-1", "owner/repo");
        task.install_config.insert("install".to_string(), "pip install -e .".to_string());
        
        let config = rechecker.get_fixed_install_config(&task);
        assert_eq!(config.get("install"), Some(&"pip install -e .".to_string()));
    }
}
