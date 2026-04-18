//! ComfyUI API-format workflow construction.
//!
//! Generates the flat `{"node_id": {"class_type": "...", "inputs": {...}}}` format
//! that ComfyUI's POST /prompt endpoint accepts.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A ComfyUI API-format workflow — flat map of node IDs to node definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    #[serde(flatten)]
    pub nodes: HashMap<String, Node>,
}

/// A single node in the workflow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub class_type: String,
    pub inputs: HashMap<String, Value>,
}

/// Reference to another node's output: `["node_id", output_index]`.
pub fn link(node_id: &str, output_index: u32) -> Value {
    Value::Array(vec![
        Value::String(node_id.to_string()),
        Value::Number(output_index.into()),
    ])
}

/// Builder for constructing workflows incrementally.
pub struct WorkflowBuilder {
    nodes: HashMap<String, Node>,
    next_id: u32,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a node and return its ID.
    pub fn add(&mut self, class_type: &str, inputs: HashMap<String, Value>) -> String {
        let id = self.next_id.to_string();
        self.next_id += 1;
        self.nodes.insert(
            id.clone(),
            Node {
                class_type: class_type.to_string(),
                inputs,
            },
        );
        id
    }

    /// Convenience: add a node from a list of key-value pairs.
    pub fn node(&mut self, class_type: &str, inputs: &[(&str, Value)]) -> String {
        let map: HashMap<String, Value> = inputs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        self.add(class_type, map)
    }

    pub fn build(self) -> Workflow {
        Workflow { nodes: self.nodes }
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Workflow {
    /// Serialize to the JSON body for POST /prompt.
    pub fn to_prompt_body(&self, client_id: Option<&str>) -> Value {
        let mut body = serde_json::json!({
            "prompt": self.nodes,
        });
        if let Some(cid) = client_id {
            body["client_id"] = Value::String(cid.to_string());
        }
        body
    }
}
