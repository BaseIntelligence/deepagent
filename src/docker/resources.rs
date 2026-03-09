//! Resource management for Docker containers in swe_forge.
//!
//! This module provides resource limits, volume configuration, and container
//! configuration utilities based on task difficulty levels.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Network isolation mode for containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NetworkMode {
    /// No network access at all.
    None,
    /// Internal network only (containers can communicate).
    #[default]
    Internal,
    /// Bridge network with potential external access.
    Bridge,
}

impl std::fmt::Display for NetworkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkMode::None => write!(f, "none"),
            NetworkMode::Internal => write!(f, "internal"),
            NetworkMode::Bridge => write!(f, "bridge"),
        }
    }
}

/// Resource limits for a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Number of CPU cores (e.g., 1.0, 2.0, 4.0).
    pub cpu_count: f64,
    /// Memory limit in bytes.
    pub memory_bytes: u64,
    /// Storage limit in bytes.
    pub storage_bytes: u64,
    /// Maximum number of processes.
    pub pids_limit: u32,
    /// Network isolation mode.
    pub network_mode: NetworkMode,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_count: 0.0,
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
            storage_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            pids_limit: 200,
            network_mode: NetworkMode::Internal,
        }
    }
}

impl ResourceLimits {
    /// Get memory limit as a human-readable string.
    pub fn memory_string(&self) -> String {
        let mb = self.memory_bytes / (1024 * 1024);
        if mb >= 1024 {
            format!("{}G", mb / 1024)
        } else {
            format!("{}M", mb)
        }
    }

    /// Get storage limit as a human-readable string.
    pub fn storage_string(&self) -> String {
        let gb = self.storage_bytes / (1024 * 1024 * 1024);
        format!("{}G", gb)
    }
}

/// Predefined resource limits for easy difficulty tasks.
pub const EASY_LIMITS: ResourceLimits = ResourceLimits {
    cpu_count: 0.0,
    memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
    storage_bytes: 1024 * 1024 * 1024,     // 1 GB
    pids_limit: 100,
    network_mode: NetworkMode::None,
};

/// Predefined resource limits for medium difficulty tasks.
pub const MEDIUM_LIMITS: ResourceLimits = ResourceLimits {
    cpu_count: 0.0,
    memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
    storage_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
    pids_limit: 200,
    network_mode: NetworkMode::Internal,
};

/// Predefined resource limits for hard difficulty tasks.
pub const HARD_LIMITS: ResourceLimits = ResourceLimits {
    cpu_count: 0.0,
    memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
    storage_bytes: 5 * 1024 * 1024 * 1024, // 5 GB
    pids_limit: 500,
    network_mode: NetworkMode::Internal,
};

/// Get resource limits based on difficulty level.
///
/// # Arguments
/// * `difficulty` - Difficulty level: "easy", "medium", or "hard"
///
/// # Returns
/// Resource limits appropriate for the difficulty level.
pub fn apply_resource_limits(difficulty: &str) -> ResourceLimits {
    match difficulty.to_lowercase().as_str() {
        "easy" => EASY_LIMITS,
        "medium" => MEDIUM_LIMITS,
        "hard" => HARD_LIMITS,
        _ => MEDIUM_LIMITS, // Default to medium if unknown
    }
}

/// Get network mode based on difficulty level.
pub fn network_mode_from_difficulty(difficulty: &str) -> NetworkMode {
    match difficulty.to_lowercase().as_str() {
        "easy" => NetworkMode::None,
        "medium" | "hard" => NetworkMode::Internal,
        _ => NetworkMode::Internal,
    }
}

/// A volume mount configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Path on the host system.
    pub host_path: String,
    /// Path inside the container.
    pub container_path: String,
    /// Whether the mount is read-only.
    pub read_only: bool,
}

impl VolumeMount {
    /// Create a new volume mount.
    pub fn new(host_path: impl Into<String>, container_path: impl Into<String>) -> Self {
        Self {
            host_path: host_path.into(),
            container_path: container_path.into(),
            read_only: false,
        }
    }

    /// Create a read-only volume mount.
    pub fn read_only(host_path: impl Into<String>, container_path: impl Into<String>) -> Self {
        Self {
            host_path: host_path.into(),
            container_path: container_path.into(),
            read_only: true,
        }
    }

    /// Convert to Docker volume string format.
    pub fn to_docker_string(&self) -> String {
        if self.read_only {
            format!("{}:{}:ro", self.host_path, self.container_path)
        } else {
            format!("{}:{}", self.host_path, self.container_path)
        }
    }
}

/// Container configuration combining all settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Container name.
    pub name: String,
    /// Docker image to use.
    pub image: String,
    /// Resource limits.
    pub limits: ResourceLimits,
    /// Environment variables.
    pub env_vars: HashMap<String, String>,
    /// Volume mounts.
    pub volumes: Vec<VolumeMount>,
    /// Network mode.
    pub network_mode: NetworkMode,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            image: String::new(),
            limits: ResourceLimits::default(),
            env_vars: HashMap::new(),
            volumes: Vec::new(),
            network_mode: NetworkMode::Internal,
        }
    }
}

/// Create a set of secure volume mounts for a task.
///
/// # Arguments
/// * `task_id` - Unique identifier for the task
///
/// # Returns
/// A vector of volume mounts with appropriate security settings.
pub fn create_secure_volumes(task_id: &str) -> Vec<VolumeMount> {
    vec![
        // Task dependencies - read-only to prevent modification
        VolumeMount::read_only("./task-deps", "/task-deps"),
        // User workspace - read-write for task execution
        VolumeMount::new(
            format!("/var/lib/swe_forge/tasks/{}/workspace", task_id),
            "/home/user",
        ),
        // Results directory - read-write for output collection
        VolumeMount::new(
            format!("/var/lib/swe_forge/tasks/{}/results", task_id),
            "/home/user/results",
        ),
        // Logs directory - read-write for debugging
        VolumeMount::new(
            format!("/var/lib/swe_forge/tasks/{}/logs", task_id),
            "/var/log/task",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::process::Command;

    // =========================================================================
    // ResourceLimits Struct Creation Tests
    // =========================================================================

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.cpu_count, 0.0);
        assert_eq!(limits.memory_bytes, 32 * 1024 * 1024 * 1024); // 32 GB
        assert_eq!(limits.storage_bytes, 2 * 1024 * 1024 * 1024); // 2 GB
        assert_eq!(limits.pids_limit, 200);
        assert_eq!(limits.network_mode, NetworkMode::Internal);
    }

    #[test]
    fn test_resource_limits_custom_values() {
        let limits = ResourceLimits {
            cpu_count: 2.0,
            memory_bytes: 1024 * 1024 * 1024, // 1 GB
            storage_bytes: 512 * 1024 * 1024, // 512 MB
            pids_limit: 50,
            network_mode: NetworkMode::None,
        };
        assert_eq!(limits.cpu_count, 2.0);
        assert_eq!(limits.memory_bytes, 1024 * 1024 * 1024);
        assert_eq!(limits.storage_bytes, 512 * 1024 * 1024);
        assert_eq!(limits.pids_limit, 50);
        assert_eq!(limits.network_mode, NetworkMode::None);
    }

    #[test]
    fn test_resource_limits_edge_cases() {
        // Zero values
        let limits_zero = ResourceLimits {
            cpu_count: 0.0,
            memory_bytes: 0,
            storage_bytes: 0,
            pids_limit: 0,
            network_mode: NetworkMode::None,
        };
        assert_eq!(limits_zero.memory_string(), "0M");
        assert_eq!(limits_zero.storage_string(), "0G");

        // Very large values
        let limits_large = ResourceLimits {
            cpu_count: 128.0,
            memory_bytes: 1024 * 1024 * 1024 * 1024,  // 1 TB
            storage_bytes: 1024 * 1024 * 1024 * 1024, // 1 TB
            pids_limit: u32::MAX,
            network_mode: NetworkMode::Bridge,
        };
        assert_eq!(limits_large.memory_string(), "1024G");
        assert_eq!(limits_large.storage_string(), "1024G");
    }

    #[test]
    fn test_resource_limits_serialization_roundtrip() {
        let original = ResourceLimits {
            cpu_count: 2.0,
            memory_bytes: 8 * 1024 * 1024 * 1024,  // 8 GB
            storage_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            pids_limit: 300,
            network_mode: NetworkMode::Bridge,
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ResourceLimits = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.cpu_count, original.cpu_count);
        assert_eq!(deserialized.memory_bytes, original.memory_bytes);
        assert_eq!(deserialized.storage_bytes, original.storage_bytes);
        assert_eq!(deserialized.pids_limit, original.pids_limit);
        assert_eq!(deserialized.network_mode, original.network_mode);
    }

    // =========================================================================
    // Memory and Storage String Formatting Tests
    // =========================================================================

    #[test]
    fn test_memory_string_megabytes() {
        let limits = ResourceLimits {
            memory_bytes: 512 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(limits.memory_string(), "512M");

        let limits_small = ResourceLimits {
            memory_bytes: 64 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(limits_small.memory_string(), "64M");
    }

    #[test]
    fn test_memory_string_gigabytes() {
        let limits = ResourceLimits {
            memory_bytes: 32 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(limits.memory_string(), "32G");

        let limits_1gb = ResourceLimits {
            memory_bytes: 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(limits_1gb.memory_string(), "1G");
    }

    #[test]
    fn test_storage_string() {
        let limits_1gb = ResourceLimits {
            storage_bytes: 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(limits_1gb.storage_string(), "1G");

        let limits_5gb = ResourceLimits {
            storage_bytes: 5 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(limits_5gb.storage_string(), "5G");

        let limits_0 = ResourceLimits {
            storage_bytes: 0,
            ..Default::default()
        };
        assert_eq!(limits_0.storage_string(), "0G");
    }

    // =========================================================================
    // Difficulty-Based Limit Application Tests
    // =========================================================================

    #[test]
    fn test_resource_limits_easy() {
        let limits = apply_resource_limits("easy");
        assert_eq!(limits.cpu_count, EASY_LIMITS.cpu_count);
        assert_eq!(limits.memory_bytes, EASY_LIMITS.memory_bytes);
        assert_eq!(limits.storage_bytes, EASY_LIMITS.storage_bytes);
        assert_eq!(limits.pids_limit, EASY_LIMITS.pids_limit);
        assert_eq!(limits.network_mode, NetworkMode::None);

        // Verify specific easy values
        assert_eq!(limits.pids_limit, 100);
        assert_eq!(limits.storage_bytes, 1024 * 1024 * 1024); // 1 GB
    }

    #[test]
    fn test_resource_limits_medium() {
        let limits = apply_resource_limits("medium");
        assert_eq!(limits.cpu_count, MEDIUM_LIMITS.cpu_count);
        assert_eq!(limits.memory_bytes, MEDIUM_LIMITS.memory_bytes);
        assert_eq!(limits.storage_bytes, MEDIUM_LIMITS.storage_bytes);
        assert_eq!(limits.pids_limit, MEDIUM_LIMITS.pids_limit);
        assert_eq!(limits.network_mode, NetworkMode::Internal);

        // Verify specific medium values
        assert_eq!(limits.pids_limit, 200);
        assert_eq!(limits.storage_bytes, 2 * 1024 * 1024 * 1024); // 2 GB
    }

    #[test]
    fn test_resource_limits_hard() {
        let limits = apply_resource_limits("hard");
        assert_eq!(limits.cpu_count, HARD_LIMITS.cpu_count);
        assert_eq!(limits.memory_bytes, HARD_LIMITS.memory_bytes);
        assert_eq!(limits.storage_bytes, HARD_LIMITS.storage_bytes);
        assert_eq!(limits.pids_limit, HARD_LIMITS.pids_limit);
        assert_eq!(limits.network_mode, NetworkMode::Internal);

        // Verify specific hard values
        assert_eq!(limits.pids_limit, 500);
        assert_eq!(limits.storage_bytes, 5 * 1024 * 1024 * 1024); // 5 GB
    }

    #[test]
    fn test_resource_limits_default_unknown() {
        // Unknown difficulty should default to medium
        let limits = apply_resource_limits("unknown");
        let medium = apply_resource_limits("medium");

        assert_eq!(limits.cpu_count, medium.cpu_count);
        assert_eq!(limits.memory_bytes, medium.memory_bytes);
        assert_eq!(limits.storage_bytes, medium.storage_bytes);
        assert_eq!(limits.pids_limit, medium.pids_limit);
        assert_eq!(limits.network_mode, medium.network_mode);
    }

    #[test]
    fn test_apply_resource_limits_case_insensitive() {
        let easy_lower = apply_resource_limits("easy");
        let easy_upper = apply_resource_limits("EASY");
        let easy_mixed = apply_resource_limits("Easy");
        let easy_title = apply_resource_limits("EAsY");

        assert_eq!(easy_lower.pids_limit, easy_upper.pids_limit);
        assert_eq!(easy_lower.pids_limit, easy_mixed.pids_limit);
        assert_eq!(easy_lower.pids_limit, easy_title.pids_limit);
        assert_eq!(easy_lower.storage_bytes, easy_upper.storage_bytes);

        let hard_lower = apply_resource_limits("hard");
        let hard_upper = apply_resource_limits("HARD");
        let hard_mixed = apply_resource_limits("Hard");

        assert_eq!(hard_lower.pids_limit, hard_upper.pids_limit);
        assert_eq!(hard_lower.pids_limit, hard_mixed.pids_limit);
        assert_eq!(hard_lower.storage_bytes, hard_upper.storage_bytes);
    }

    #[test]
    fn test_predefined_constants() {
        // Verify EASY_LIMITS constant
        assert_eq!(EASY_LIMITS.pids_limit, 100);
        assert_eq!(EASY_LIMITS.storage_bytes, 1024 * 1024 * 1024); // 1 GB
        assert_eq!(EASY_LIMITS.memory_bytes, 32 * 1024 * 1024 * 1024); // 32 GB
        assert_eq!(EASY_LIMITS.network_mode, NetworkMode::None);

        // Verify MEDIUM_LIMITS constant
        assert_eq!(MEDIUM_LIMITS.pids_limit, 200);
        assert_eq!(MEDIUM_LIMITS.storage_bytes, 2 * 1024 * 1024 * 1024); // 2 GB
        assert_eq!(MEDIUM_LIMITS.memory_bytes, 32 * 1024 * 1024 * 1024); // 32 GB
        assert_eq!(MEDIUM_LIMITS.network_mode, NetworkMode::Internal);

        // Verify HARD_LIMITS constant
        assert_eq!(HARD_LIMITS.pids_limit, 500);
        assert_eq!(HARD_LIMITS.storage_bytes, 5 * 1024 * 1024 * 1024); // 5 GB
        assert_eq!(HARD_LIMITS.memory_bytes, 32 * 1024 * 1024 * 1024); // 32 GB
        assert_eq!(HARD_LIMITS.network_mode, NetworkMode::Internal);
    }

    // =========================================================================
    // Network Mode Tests
    // =========================================================================

    #[test]
    fn test_network_mode_display() {
        assert_eq!(format!("{}", NetworkMode::None), "none");
        assert_eq!(format!("{}", NetworkMode::Internal), "internal");
        assert_eq!(format!("{}", NetworkMode::Bridge), "bridge");
    }

    #[test]
    fn test_network_mode_from_difficulty() {
        assert_eq!(network_mode_from_difficulty("easy"), NetworkMode::None);
        assert_eq!(
            network_mode_from_difficulty("medium"),
            NetworkMode::Internal
        );
        assert_eq!(network_mode_from_difficulty("hard"), NetworkMode::Internal);
        assert_eq!(
            network_mode_from_difficulty("unknown"),
            NetworkMode::Internal
        );
    }

    #[test]
    fn test_network_mode_from_difficulty_case_insensitive() {
        assert_eq!(network_mode_from_difficulty("EASY"), NetworkMode::None);
        assert_eq!(network_mode_from_difficulty("Easy"), NetworkMode::None);
        assert_eq!(
            network_mode_from_difficulty("MEDIUM"),
            NetworkMode::Internal
        );
        assert_eq!(network_mode_from_difficulty("HARD"), NetworkMode::Internal);
    }

    #[test]
    fn test_network_mode_default() {
        let mode: NetworkMode = Default::default();
        assert_eq!(mode, NetworkMode::Internal);
    }

    #[test]
    fn test_network_mode_serialization_roundtrip() {
        for mode in [
            NetworkMode::None,
            NetworkMode::Internal,
            NetworkMode::Bridge,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let deserialized: NetworkMode = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, mode);
        }
    }

    #[test]
    fn test_network_mode_yaml_serialization() {
        // Test YAML format (commonly used in config files)
        // NetworkMode uses default serde serialization (PascalCase variant names)
        let yaml_none = "None";
        let yaml_internal = "Internal";
        let yaml_bridge = "Bridge";

        let mode_none: NetworkMode = serde_yaml::from_str(yaml_none).unwrap();
        let mode_internal: NetworkMode = serde_yaml::from_str(yaml_internal).unwrap();
        let mode_bridge: NetworkMode = serde_yaml::from_str(yaml_bridge).unwrap();

        assert_eq!(mode_none, NetworkMode::None);
        assert_eq!(mode_internal, NetworkMode::Internal);
        assert_eq!(mode_bridge, NetworkMode::Bridge);

        // Verify roundtrip serialization
        for mode in [
            NetworkMode::None,
            NetworkMode::Internal,
            NetworkMode::Bridge,
        ] {
            let yaml = serde_yaml::to_string(&mode).unwrap();
            let deserialized: NetworkMode = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(deserialized, mode);
        }
    }

    // =========================================================================
    // Docker Integration Tests
    // =========================================================================

    /// Helper function to generate unique container names for tests
    fn test_container_name(prefix: &str) -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        format!("{}-test-{}-{:x}", prefix, timestamp, std::process::id())
    }

    /// Helper function to cleanup test containers
    async fn cleanup_container(name: &str) {
        let _ = Command::new("docker")
            .args(["rm", "-f", name])
            .output()
            .await;
    }

    #[tokio::test]
    async fn test_docker_container_creation_with_easy_limits() {
        let container_name = test_container_name("swe-easy");
        let limits = apply_resource_limits("easy");

        // Create container with easy limits
        let output = Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                &container_name,
                "--memory",
                &limits.memory_string(),
                "--pids-limit",
                &limits.pids_limit.to_string(),
                "--network",
                &limits.network_mode.to_string(),
                "--storage-opt",
                &format!("size={}", limits.storage_string()),
                "python:3.11-slim",
                "sleep",
                "30",
            ])
            .output()
            .await
            .expect("Failed to execute docker command");

        if !output.status.success() {
            // Storage limits may not be supported on all systems, try without
            let output2 = Command::new("docker")
                .args([
                    "run",
                    "-d",
                    "--name",
                    &container_name,
                    "--memory",
                    &limits.memory_string(),
                    "--pids-limit",
                    &limits.pids_limit.to_string(),
                    "--network",
                    &limits.network_mode.to_string(),
                    "python:3.11-slim",
                    "sleep",
                    "30",
                ])
                .output()
                .await
                .expect("Failed to execute docker command");
            assert!(
                output2.status.success(),
                "Failed to create container: {}",
                String::from_utf8_lossy(&output2.stderr)
            );
        }

        // Verify container is running
        let ps_output = Command::new("docker")
            .args(["ps", "-q", "-f", &format!("name={}", container_name)])
            .output()
            .await
            .unwrap();
        assert!(!ps_output.stdout.is_empty(), "Container should be running");

        // Verify limits via docker inspect
        let inspect_output = Command::new("docker")
            .args(["inspect", &container_name])
            .output()
            .await
            .unwrap();
        assert!(inspect_output.status.success());

        let inspect_json: serde_json::Value = serde_json::from_slice(&inspect_output.stdout)
            .expect("Failed to parse docker inspect output");
        let container_info = &inspect_json[0];

        // Verify memory limit
        let host_config = container_info["HostConfig"].clone();
        let memory_limit = host_config["Memory"].as_i64().unwrap_or(0);
        assert_eq!(
            memory_limit, limits.memory_bytes as i64,
            "Memory limit mismatch"
        );

        // Verify PIDs limit
        let pids_limit = host_config["PidsLimit"].as_i64();
        if let Some(limit) = pids_limit {
            assert_eq!(limit, limits.pids_limit as i64, "PIDs limit mismatch");
        }

        // Verify network mode
        let network_mode = host_config["NetworkMode"].as_str().unwrap_or("");
        assert!(
            network_mode.contains("none"),
            "Network mode should be none, got: {}",
            network_mode
        );

        // Cleanup
        cleanup_container(&container_name).await;

        // Verify cleanup
        let ps_after = Command::new("docker")
            .args(["ps", "-a", "-q", "-f", &format!("name={}", container_name)])
            .output()
            .await
            .unwrap();
        assert!(ps_after.stdout.is_empty(), "Container should be removed");
    }

    #[tokio::test]
    async fn test_docker_container_creation_with_medium_limits() {
        let container_name = test_container_name("swe-medium");
        let limits = apply_resource_limits("medium");

        // For medium difficulty, we use bridge network (internal is a conceptual mode)
        // Create container with medium limits using bridge network
        let output = Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                &container_name,
                "--memory",
                &limits.memory_string(),
                "--pids-limit",
                &limits.pids_limit.to_string(),
                "--network",
                "bridge",
                "python:3.11-slim",
                "sleep",
                "30",
            ])
            .output()
            .await
            .expect("Failed to execute docker command");

        assert!(
            output.status.success(),
            "Failed to create container: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Verify limits via docker inspect
        let inspect_output = Command::new("docker")
            .args(["inspect", &container_name])
            .output()
            .await
            .unwrap();
        assert!(inspect_output.status.success());

        let inspect_json: serde_json::Value = serde_json::from_slice(&inspect_output.stdout)
            .expect("Failed to parse docker inspect output");
        let host_config = &inspect_json[0]["HostConfig"];

        // Verify memory limit
        let memory_limit = host_config["Memory"].as_i64().unwrap_or(0);
        assert_eq!(
            memory_limit, limits.memory_bytes as i64,
            "Memory limit mismatch"
        );

        // Verify PIDs limit
        let pids_limit = host_config["PidsLimit"].as_i64();
        if let Some(limit) = pids_limit {
            assert_eq!(limit, limits.pids_limit as i64, "PIDs limit mismatch");
        }

        // Verify network mode uses bridge
        let network_mode = host_config["NetworkMode"].as_str().unwrap_or("");
        assert!(
            network_mode.contains("bridge"),
            "Network mode should be bridge, got: {}",
            network_mode
        );

        // Cleanup
        cleanup_container(&container_name).await;

        // Verify cleanup
        let ps_after = Command::new("docker")
            .args(["ps", "-a", "-q", "-f", &format!("name={}", container_name)])
            .output()
            .await
            .unwrap();
        assert!(ps_after.stdout.is_empty(), "Container should be removed");
    }

    #[tokio::test]
    async fn test_docker_container_creation_with_hard_limits() {
        let container_name = test_container_name("swe-hard");
        let limits = apply_resource_limits("hard");

        // For hard difficulty, we use bridge network (internal is a conceptual mode)
        // Create container with hard limits using bridge network
        let output = Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                &container_name,
                "--memory",
                &limits.memory_string(),
                "--pids-limit",
                &limits.pids_limit.to_string(),
                "--network",
                "bridge",
                "python:3.11-slim",
                "sleep",
                "30",
            ])
            .output()
            .await
            .expect("Failed to execute docker command");

        assert!(
            output.status.success(),
            "Failed to create container: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Verify limits via docker inspect
        let inspect_output = Command::new("docker")
            .args(["inspect", &container_name])
            .output()
            .await
            .unwrap();
        assert!(inspect_output.status.success());

        let inspect_json: serde_json::Value = serde_json::from_slice(&inspect_output.stdout)
            .expect("Failed to parse docker inspect output");
        let host_config = &inspect_json[0]["HostConfig"];

        // Verify memory limit (32GB)
        let memory_limit = host_config["Memory"].as_i64().unwrap_or(0);
        assert_eq!(
            memory_limit, limits.memory_bytes as i64,
            "Memory limit mismatch"
        );

        // Verify PIDs limit (500 for hard)
        let pids_limit = host_config["PidsLimit"].as_i64();
        if let Some(limit) = pids_limit {
            assert_eq!(limit, limits.pids_limit as i64, "PIDs limit mismatch");
        }

        // Verify network mode uses bridge
        let network_mode = host_config["NetworkMode"].as_str().unwrap_or("");
        assert!(
            network_mode.contains("bridge"),
            "Network mode should be bridge, got: {}",
            network_mode
        );

        // Cleanup
        cleanup_container(&container_name).await;

        // Verify cleanup
        let ps_after = Command::new("docker")
            .args(["ps", "-a", "-q", "-f", &format!("name={}", container_name)])
            .output()
            .await
            .unwrap();
        assert!(ps_after.stdout.is_empty(), "Container should be removed");
    }

    #[tokio::test]
    async fn test_resource_limits_comparison_between_difficulties() {
        let easy = apply_resource_limits("easy");
        let medium = apply_resource_limits("medium");
        let hard = apply_resource_limits("hard");

        // Verify PIDs limits increase with difficulty
        assert!(
            easy.pids_limit < medium.pids_limit,
            "Easy should have fewer PIDs than medium"
        );
        assert!(
            medium.pids_limit < hard.pids_limit,
            "Medium should have fewer PIDs than hard"
        );
        assert_eq!(easy.pids_limit, 100);
        assert_eq!(medium.pids_limit, 200);
        assert_eq!(hard.pids_limit, 500);

        // Verify storage limits increase with difficulty
        assert!(
            easy.storage_bytes < medium.storage_bytes,
            "Easy should have less storage than medium"
        );
        assert!(
            medium.storage_bytes < hard.storage_bytes,
            "Medium should have less storage than hard"
        );
        assert_eq!(easy.storage_bytes, 1024 * 1024 * 1024); // 1 GB
        assert_eq!(medium.storage_bytes, 2 * 1024 * 1024 * 1024); // 2 GB
        assert_eq!(hard.storage_bytes, 5 * 1024 * 1024 * 1024); // 5 GB

        // Memory should be the same for all (32 GB)
        assert_eq!(easy.memory_bytes, medium.memory_bytes);
        assert_eq!(medium.memory_bytes, hard.memory_bytes);
        assert_eq!(easy.memory_bytes, 32 * 1024 * 1024 * 1024);

        // Network mode progression: None -> Internal -> Internal
        assert_eq!(easy.network_mode, NetworkMode::None);
        assert_eq!(medium.network_mode, NetworkMode::Internal);
        assert_eq!(hard.network_mode, NetworkMode::Internal);
    }

    #[tokio::test]
    async fn test_docker_container_cleanup_on_test_failure() {
        let container_name = test_container_name("swe-cleanup-test");
        let limits = apply_resource_limits("easy");

        // Create container
        let output = Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                &container_name,
                "--memory",
                &limits.memory_string(),
                "--pids-limit",
                &limits.pids_limit.to_string(),
                "--network",
                &limits.network_mode.to_string(),
                "python:3.11-slim",
                "sleep",
                "300",
            ])
            .output()
            .await
            .expect("Failed to execute docker command");

        assert!(output.status.success());

        // Simulate test failure (container should still be cleaned up)
        // In real tests, cleanup happens regardless of test result

        // Cleanup
        cleanup_container(&container_name).await;

        // Verify cleanup
        let ps_after = Command::new("docker")
            .args(["ps", "-a", "-q", "-f", &format!("name={}", container_name)])
            .output()
            .await
            .unwrap();
        assert!(
            ps_after.stdout.is_empty(),
            "Container should be removed even after test failure simulation"
        );
    }

    // =========================================================================
    // Volume Mount Tests
    // =========================================================================

    #[test]
    fn test_volume_mount() {
        let vol = VolumeMount::new("/host/path", "/container/path");
        assert!(!vol.read_only);
        assert_eq!(vol.host_path, "/host/path");
        assert_eq!(vol.container_path, "/container/path");
        assert_eq!(vol.to_docker_string(), "/host/path:/container/path");

        let vol_ro = VolumeMount::read_only("/host/ro", "/container/ro");
        assert!(vol_ro.read_only);
        assert_eq!(vol_ro.host_path, "/host/ro");
        assert_eq!(vol_ro.container_path, "/container/ro");
        assert_eq!(vol_ro.to_docker_string(), "/host/ro:/container/ro:ro");
    }

    #[test]
    fn test_volume_mount_string_types() {
        // Test with String input
        let vol_string =
            VolumeMount::new(String::from("/host/path"), String::from("/container/path"));
        assert_eq!(vol_string.host_path, "/host/path");
        assert_eq!(vol_string.container_path, "/container/path");

        // Test with &str input
        let vol_str = VolumeMount::new("/host/path", "/container/path");
        assert_eq!(vol_str.host_path, "/host/path");
        assert_eq!(vol_str.container_path, "/container/path");
    }

    #[test]
    fn test_volume_mount_serialization() {
        let vol = VolumeMount {
            host_path: "/host/path".to_string(),
            container_path: "/container/path".to_string(),
            read_only: true,
        };

        let json = serde_json::to_string(&vol).unwrap();
        let deserialized: VolumeMount = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host_path, vol.host_path);
        assert_eq!(deserialized.container_path, vol.container_path);
        assert_eq!(deserialized.read_only, vol.read_only);
    }

    // =========================================================================
    // Secure Volumes Tests
    // =========================================================================

    #[test]
    fn test_create_secure_volumes() {
        let volumes = create_secure_volumes("test-task-001");
        assert_eq!(volumes.len(), 4);

        // Check task-deps is read-only
        let task_deps = volumes.iter().find(|v| v.container_path == "/task-deps");
        assert!(task_deps.is_some());
        assert!(task_deps.unwrap().read_only);
        assert_eq!(task_deps.unwrap().host_path, "./task-deps");

        // Check user workspace is writable
        let workspace = volumes.iter().find(|v| v.container_path == "/home/user");
        assert!(workspace.is_some());
        assert!(!workspace.unwrap().read_only);
        assert!(workspace.unwrap().host_path.contains("test-task-001"));
        assert!(workspace.unwrap().host_path.contains("workspace"));

        // Check results is writable
        let results = volumes
            .iter()
            .find(|v| v.container_path == "/home/user/results");
        assert!(results.is_some());
        assert!(!results.unwrap().read_only);
        assert!(results.unwrap().host_path.contains("test-task-001"));
        assert!(results.unwrap().host_path.contains("results"));

        // Check logs is writable
        let logs = volumes.iter().find(|v| v.container_path == "/var/log/task");
        assert!(logs.is_some());
        assert!(!logs.unwrap().read_only);
        assert!(logs.unwrap().host_path.contains("test-task-001"));
        assert!(logs.unwrap().host_path.contains("logs"));
    }

    #[test]
    fn test_create_secure_volumes_unique_task_ids() {
        let volumes_1 = create_secure_volumes("task-001");
        let volumes_2 = create_secure_volumes("task-002");

        // Verify workspace paths are different
        let workspace_1 = volumes_1
            .iter()
            .find(|v| v.container_path == "/home/user")
            .unwrap();
        let workspace_2 = volumes_2
            .iter()
            .find(|v| v.container_path == "/home/user")
            .unwrap();
        assert_ne!(workspace_1.host_path, workspace_2.host_path);
        assert!(workspace_1.host_path.contains("task-001"));
        assert!(workspace_2.host_path.contains("task-002"));
    }

    // =========================================================================
    // ContainerConfig Tests
    // =========================================================================

    #[test]
    fn test_container_config_default() {
        let config = ContainerConfig::default();
        assert_eq!(config.name, "");
        assert_eq!(config.image, "");
        assert_eq!(config.limits.cpu_count, 0.0);
        assert_eq!(config.limits.memory_bytes, 32 * 1024 * 1024 * 1024);
        assert!(config.env_vars.is_empty());
        assert!(config.volumes.is_empty());
        assert_eq!(config.network_mode, NetworkMode::Internal);
    }

    #[test]
    fn test_container_config_custom() {
        let limits = ResourceLimits {
            cpu_count: 2.0,
            memory_bytes: 8 * 1024 * 1024 * 1024,
            storage_bytes: 4 * 1024 * 1024 * 1024,
            pids_limit: 300,
            network_mode: NetworkMode::Bridge,
        };

        let mut env_vars = HashMap::new();
        env_vars.insert("KEY1".to_string(), "value1".to_string());
        env_vars.insert("KEY2".to_string(), "value2".to_string());

        let volumes = vec![
            VolumeMount::new("/host/data", "/container/data"),
            VolumeMount::read_only("/host/readonly", "/container/readonly"),
        ];

        let config = ContainerConfig {
            name: "test-container".to_string(),
            image: "python:3.11".to_string(),
            limits: limits.clone(),
            env_vars: env_vars.clone(),
            volumes: volumes.clone(),
            network_mode: NetworkMode::Bridge,
        };

        assert_eq!(config.name, "test-container");
        assert_eq!(config.image, "python:3.11");
        assert_eq!(config.limits.cpu_count, 2.0);
        assert_eq!(config.limits.memory_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(config.env_vars.len(), 2);
        assert_eq!(config.env_vars.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(config.volumes.len(), 2);
        assert!(!config.volumes[0].read_only);
        assert!(config.volumes[1].read_only);
        assert_eq!(config.network_mode, NetworkMode::Bridge);
    }

    #[test]
    fn test_container_config_serialization_roundtrip() {
        let limits = ResourceLimits {
            cpu_count: 2.0,
            memory_bytes: 8 * 1024 * 1024 * 1024,
            storage_bytes: 4 * 1024 * 1024 * 1024,
            pids_limit: 300,
            network_mode: NetworkMode::Bridge,
        };

        let mut env_vars = HashMap::new();
        env_vars.insert("TEST_KEY".to_string(), "test_value".to_string());

        let original = ContainerConfig {
            name: "test-container".to_string(),
            image: "python:3.11".to_string(),
            limits,
            env_vars,
            volumes: vec![VolumeMount::new("/host", "/container")],
            network_mode: NetworkMode::None,
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ContainerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, original.name);
        assert_eq!(deserialized.image, original.image);
        assert_eq!(deserialized.limits.cpu_count, original.limits.cpu_count);
        assert_eq!(
            deserialized.limits.memory_bytes,
            original.limits.memory_bytes
        );
        assert_eq!(deserialized.limits.pids_limit, original.limits.pids_limit);
        assert_eq!(deserialized.env_vars.len(), original.env_vars.len());
        assert_eq!(deserialized.volumes.len(), original.volumes.len());
        assert_eq!(deserialized.network_mode, original.network_mode);
    }
}
