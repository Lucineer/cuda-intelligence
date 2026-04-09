//! GPU timing, power, and signoff verification
//!
//! Static timing analysis, power estimation, and comprehensive
//! signoff checks for mask-locked inference chips at various
//! process nodes.

use crate::VesselSpec;

/// Process node parameters
#[derive(Debug, Clone)]
pub struct ProcessNode {
    pub name: &'static str,
    pub nm: u32,
    pub vdd: f64,          // V
    pub gate_delay_ps: f64, // ps per gate
    pub power_per_gate_nw: f64, // nW per gate per MHz
    pub leakage_per_mm2_mw: f64, // mW/mm² leakage
}

pub const NODE_5NM: ProcessNode = ProcessNode { name: "5nm", nm: 5, vdd: 0.75, gate_delay_ps: 2.0, power_per_gate_nw: 0.5, leakage_per_mm2_mw: 5.0 };
pub const NODE_7NM: ProcessNode = ProcessNode { name: "7nm", nm: 7, vdd: 0.80, gate_delay_ps: 3.0, power_per_gate_nw: 0.8, leakage_per_mm2_mw: 3.0 };
pub const NODE_14NM: ProcessNode = ProcessNode { name: "14nm", nm: 14, vdd: 0.90, gate_delay_ps: 8.0, power_per_gate_nw: 2.0, leakage_per_mm2_mw: 1.0 };
pub const NODE_28NM: ProcessNode = ProcessNode { name: "28nm", nm: 28, vdd: 1.0, gate_delay_ps: 20.0, power_per_gate_nw: 5.0, leakage_per_mm2_mw: 0.2 };
pub const NODE_65NM: ProcessNode = ProcessNode { name: "65nm", nm: 65, vdd: 1.2, gate_delay_ps: 50.0, power_per_gate_nw: 15.0, leakage_per_mm2_mw: 0.05 };

pub const ALL_NODES: &[ProcessNode] = &[&NODE_5NM, &NODE_7NM, &NODE_14NM, &NODE_28NM, &NODE_65NM];

/// Timing check result
#[derive(Debug)]
pub struct TimingCheck {
    pub path: String,
    pub delay_ps: f64,
    pub slack_ps: f64,
    pub met: bool,
}

/// Power estimation result
#[derive(Debug)]
pub struct PowerEstimate {
    pub dynamic_mw: f64,
    pub leakage_mw: f64,
    pub total_mw: f64,
    pub per_tok_uj: f64,
}

/// Signoff result
#[derive(Debug)]
pub struct SignoffResult {
    pub timing_ok: bool,
    pub power_ok: bool,
    pub area_ok: bool,
    pub checks: Vec<String>,
    pub warnings: Vec<String>,
}

/// Timing analyzer — static timing analysis
pub struct TimingAnalyzer {
    pub node: ProcessNode,
    pub target_freq_mhz: f64,
}

impl TimingAnalyzer {
    pub fn new(node: ProcessNode, target_freq_mhz: f64) -> Self {
        TimingAnalyzer { node, target_freq_mhz }
    }

    /// Calculate critical path delay for MAC operation
    pub fn mac_critical_path(&self) -> TimingCheck {
        // MAC = partial product + Wallace tree + CLA adder
        let pp_delay = self.node.gate_delay_ps * 8.0;  // 8 gates for partial product
        let wallace = self.node.gate_delay_ps * 12.0;  // Wallace tree reduction
        let cla = self.node.gate_delay_ps * 6.0;       // Carry-lookahead adder
        let total = pp_delay + wallace + cla;

        let period_ps = 1000.0 / self.target_freq_mhz;
        let slack = period_ps - total;

        TimingCheck {
            path: "MAC_critical_path".to_string(),
            delay_ps: total, slack_ps: slack, met: slack >= 0.0,
        }
    }

    /// Check systolic array pipeline timing
    pub fn systolic_timing(&self, array_size: usize) -> Vec<TimingCheck> {
        let mac = self.mac_critical_path();
        let pipeline_overhead = self.node.gate_delay_ps * 4.0; // register setup+hold

        vec![
            TimingCheck { path: "mac_unit".to_string(), delay_ps: mac.delay_ps, slack_ps: mac.slack_ps, met: mac.met },
            TimingCheck { path: "pipeline_register".to_string(), delay_ps: pipeline_overhead,
                slack_ps: 1000.0/self.target_freq_mhz - pipeline_overhead, met: true },
            TimingCheck { path: "systolic_total".to_string(),
                delay_ps: mac.delay_ps + pipeline_overhead,
                slack_ps: 1000.0/self.target_freq_mhz - mac.delay_ps - pipeline_overhead,
                met: mac.delay_ps + pipeline_overhead < 1000.0/self.target_freq_mhz },
        ]
    }
}

/// Power estimator — dynamic + leakage
pub struct PowerEstimator {
    pub node: ProcessNode,
}

impl PowerEstimator {
    pub fn new(node: ProcessNode) -> Self {
        PowerEstimator { node }
    }

    /// Estimate power for a vessel
    pub fn estimate(&self, vessel: &VesselSpec, die_mm2: f64, utilization: f64) -> PowerEstimate {
        // Estimate gates from params (INT4 = 2 bits per param, ~4 transistors per gate)
        let total_bits = vessel.params_b * 2;
        let active_gates = total_bits as f64 * utilization;

        // Dynamic power: α·C·V²·f
        let freq_mhz = 500.0; // typical inference clock
        let dynamic = active_gates * self.node.power_per_gate_nw * (self.node.vdd * self.node.vdd / 1.0) * freq_mhz * 1e-6; // mW

        // Leakage power
        let leakage = self.node.leakage_per_mm2_mw * die_mm2;

        // Energy per token (simplified)
        let gops = vessel.params_b as f64 * 2.0 / vessel.speed_toks as f64;
        let per_tok_uj = dynamic * 1e-3 * gops / 1e9 * 1e6; // μJ

        PowerEstimate {
            dynamic_mw: dynamic, leakage_mw: leakage, total_mw: dynamic + leakage,
            per_tok_uj,
        }
    }

    /// Compare power across process nodes
    pub fn compare_nodes(&self, vessel: &VesselSpec, die_mm2: f64) -> Vec<(String, PowerEstimate)> {
        ALL_NODES.iter().map(|n| {
            let est = PowerEstimator::new((*n).clone()).estimate(vessel, die_mm2, 0.5);
            (n.name.to_string(), est)
        }).collect()
    }
}

/// Comprehensive signoff checker
pub struct SignoffChecker {
    pub vessel: VesselSpec,
    pub die_mm2: f64,
    pub max_power_w: f64,
    pub target_freq_mhz: f64,
}

impl SignoffChecker {
    pub fn new(vessel: VesselSpec, die_mm2: f64, max_power_w: f64, target_freq_mhz: f64) -> Self {
        SignoffChecker { vessel, die_mm2, max_power_w, target_freq_mhz }
    }

    /// Run all signoff checks
    pub fn signoff(&self, node: &ProcessNode) -> SignoffResult {
        let mut checks = Vec::new();
        let mut warnings = Vec::new();

        // Timing check
        let timing = TimingAnalyzer::new(node.clone(), self.target_freq_mhz);
        let mac_check = timing.mac_critical_path();
        let timing_ok = mac_check.met;
        checks.push(format!("Timing MAC: {}ps (slack: {}ps) {}", mac_check.delay_ps, mac_check.slack_ps, if mac_check.met {"PASS"} else {"FAIL"}));

        // Power check
        let power = PowerEstimator::new(node.clone()).estimate(&self.vessel, self.die_mm2, 0.5);
        let power_ok = power.total_mw / 1000.0 <= self.max_power_w;
        checks.push(format!("Power: {:.1}mW dynamic + {:.1}mW leakage = {:.1}mW {}",
            power.dynamic_mw, power.leakage_mw, power.total_mw,
            if power_ok {"PASS"} else {"FAIL"}));

        // Area check
        let max_die = self.vessel.die_mm2 * 1.5; // 50% margin
        let area_ok = self.die_mm2 <= max_die;
        checks.push(format!("Area: {:.1}mm² vs {:.1}mm² budget {}", self.die_mm2, max_die, if area_ok {"PASS"} else {"FAIL"}));

        if self.die_mm2 > self.vessel.die_mm2 * 1.3 {
            warnings.push(format!("Die {:.1}mm² exceeds target {:.1}mm² by >30%", self.die_mm2, self.vessel.die_mm2));
        }

        SignoffResult { timing_ok, power_ok, area_ok, checks, warnings }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_timing() {
        let ta = TimingAnalyzer::new(NODE_28NM, 500.0);
        let check = ta.mac_critical_path();
        assert!(check.delay_ps > 0.0);
        println!("MAC critical path @ 28nm: {}ps, slack: {}ps", check.delay_ps, check.slack_ps);
    }

    #[test]
    fn test_power_estimate() {
        let pe = PowerEstimator::new(NODE_28NM);
        let est = pe.estimate(&crate::NAVIGATOR, 100.0, 0.5);
        assert!(est.total_mw > 0.0);
        println!("Navigator @ 28nm: {:.1}mW total", est.total_mw);
    }

    #[test]
    fn test_node_comparison() {
        let pe = PowerEstimator::new(NODE_28NM);
        let comparison = pe.compare_nodes(&crate::SCOUT, 25.0);
        assert_eq!(comparison.len(), 5);
        for (name, est) in &comparison {
            println!("  {}: {:.1}mW", name, est.total_mw);
        }
    }

    #[test]
    fn test_signoff() {
        let sc = SignoffChecker::new(crate::SCOUT, 30.0, 2.0, 500.0);
        let result = sc.signoff(&NODE_28NM);
        println!("Signoff checks:");
        for check in &result.checks { println!("  {}", check); }
    }
}
